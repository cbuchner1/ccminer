/**
 * Blake-256 Cuda Kernel (Tested on SM 5.0)
 *
 * Tanguy Pruvot - Aug. 2014
 */

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
#include <stdint.h>
#include <memory.h>
}

/* threads per block */
#define TPB 128

/* hash by cpu with blake 256 */
extern "C" void blake32hash(void *output, const void *input)
{
	unsigned char hash[64];
	sph_blake256_context ctx;
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);
	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

// in cpu-miner.c
extern bool opt_n_threads;
extern bool opt_benchmark;
//extern bool opt_debug;
extern int device_map[8];

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__constant__
static uint32_t c_Target[8];

__constant__
static uint32_t __align__(32) c_PaddedMessage80[32]; // padded message (80 bytes + padding)

static uint32_t *d_resNounce[8];
static uint32_t *h_resNounce[8];

__constant__
static uint8_t c_sigma[16][16];
const uint8_t host_sigma[16][16] =
{
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
  {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
  {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
  { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

__device__ __constant__
static const uint32_t c_IV256[8] = {
	SPH_C32(0x6A09E667), SPH_C32(0xBB67AE85),
	SPH_C32(0x3C6EF372), SPH_C32(0xA54FF53A),
	SPH_C32(0x510E527F), SPH_C32(0x9B05688C),
	SPH_C32(0x1F83D9AB), SPH_C32(0x5BE0CD19)
};

__device__ __constant__

static const uint32_t c_u256[16] = {
	SPH_C32(0x243F6A88), SPH_C32(0x85A308D3),
	SPH_C32(0x13198A2E), SPH_C32(0x03707344),
	SPH_C32(0xA4093822), SPH_C32(0x299F31D0),
	SPH_C32(0x082EFA98), SPH_C32(0xEC4E6C89),
	SPH_C32(0x452821E6), SPH_C32(0x38D01377),
	SPH_C32(0xBE5466CF), SPH_C32(0x34E90C6C),
	SPH_C32(0xC0AC29B7), SPH_C32(0xC97C50DD),
	SPH_C32(0x3F84D5B5), SPH_C32(0xB5470917)
};

#if 0
#define GS(m0, m1, c0, c1, a, b, c, d)   do { \
		a = SPH_T32(a + b + (m0 ^ c1)); \
		d = SPH_ROTR32(d ^ a, 16); \
		c = SPH_T32(c + d); \
		b = SPH_ROTR32(b ^ c, 12); \
		a = SPH_T32(a + b + (m1 ^ c0)); \
		d = SPH_ROTR32(d ^ a, 8); \
		c = SPH_T32(c + d); \
		b = SPH_ROTR32(b ^ c, 7); \
	} while (0)

#define ROUND_S(r)   do { \
	GS(Mx(r, 0x0), Mx(r, 0x1), CSx(r, 0x0), CSx(r, 0x1), v[0], v[4], v[0x8], v[0xC]); \
	GS(Mx(r, 0x2), Mx(r, 0x3), CSx(r, 0x2), CSx(r, 0x3), v[1], v[5], v[0x9], v[0xD]); \
	GS(Mx(r, 0x4), Mx(r, 0x5), CSx(r, 0x4), CSx(r, 0x5), v[2], v[6], v[0xA], v[0xE]); \
	GS(Mx(r, 0x6), Mx(r, 0x7), CSx(r, 0x6), CSx(r, 0x7), v[3], v[7], v[0xB], v[0xF]); \
	GS(Mx(r, 0x8), Mx(r, 0x9), CSx(r, 0x8), CSx(r, 0x9), v[0], v[5], v[0xA], v[0xF]); \
	GS(Mx(r, 0xA), Mx(r, 0xB), CSx(r, 0xA), CSx(r, 0xB), v[1], v[6], v[0xB], v[0xC]); \
	GS(Mx(r, 0xC), Mx(r, 0xD), CSx(r, 0xC), CSx(r, 0xD), v[2], v[7], v[0x8], v[0xD]); \
	GS(Mx(r, 0xE), Mx(r, 0xF), CSx(r, 0xE), CSx(r, 0xF), v[3], v[4], v[0x9], v[0xE]); \
} while (0)
#endif

#define GS(a,b,c,d,e) { \
	v[a] += (m[sigma[i][e]] ^ u256[sigma[i][e+1]]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[sigma[i][e+1]] ^ u256[sigma[i][e]]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
}

#define BLAKE256_ROUNDS 14

__device__ static
void blake256_compress(uint32_t *h, uint32_t *block, uint8_t ((*sigma)[16]), const uint32_t *u256, const uint32_t T0, uint8_t nullt = 1)
{
	uint32_t /* __align__(8) */ v[16];
	uint32_t /* __align__(8) */ m[16];

	//#pragma unroll
	for (int i = 0; i < 16; ++i) {
		m[i] = block[i];
	}

	//#pragma unroll 8
	for(int i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = u256[0];
	v[ 9] = u256[1];
	v[10] = u256[2];
	v[11] = u256[3];

	v[12] = u256[4] ^ T0;
	v[13] = u256[5] ^ T0;
	v[14] = u256[6];
	v[15] = u256[7];

	//#pragma unroll
	for (int i = 0; i < BLAKE256_ROUNDS; i++) {
		/* column step */
		GS(0, 4, 0x8, 0xC, 0);
		GS(1, 5, 0x9, 0xD, 2);
		GS(2, 6, 0xA, 0xE, 4);
		GS(3, 7, 0xB, 0xF, 6);
		/* diagonal step */
		GS(0, 5, 0xA, 0xF, 0x8);
		GS(1, 6, 0xB, 0xC, 0xA);
		GS(2, 7, 0x8, 0xD, 0xC);
		GS(3, 4, 0x9, 0xE, 0xE);
	}

	//#pragma unroll 16
	for(int i = 0; i < 16; i++)
		h[i % 8] ^= v[i];
}

#if __CUDA_ARCH__ >= 200
/* memory should be aligned to use __nvvm_memset */
#if (__NV_POINTER_SIZE == 64)
# define SZCT uint64_t
#else
# define SZCT uint32_t
#endif
extern __device__ __device_builtin__ void __nvvm_memset(uint8_t *, unsigned char, SZCT, int);
#endif

__global__
void blake256_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint32_t /* __align__(8) */ msg[16];
		uint32_t h[8];

		#pragma unroll
		for(int i=0; i<8; i++)
			h[i] = c_IV256[i];

		blake256_compress(h, c_PaddedMessage80, c_sigma, c_u256, 0x200); /* 512 = 0x200 */

		// ------ Close: Bytes 64 to 80 ------ 

		msg[0] = c_PaddedMessage80[16];
		msg[1] = c_PaddedMessage80[17];
		msg[2] = c_PaddedMessage80[18];
		msg[3] = nounce; /* our tested value */
		msg[4] = 0x80000000UL; //cuda_swab32(0x80U);

		msg[5] = 0;  // uchar[17 to 55]
		msg[6] = 0;
		msg[7] = 0;
		msg[8] = 0;
		msg[9] = 0;
		msg[10] = 0;
		msg[11] = 0;
		msg[12] = 0;

		msg[13] = 1;
		msg[14] = 0;
		msg[15] = 0x280;

		blake256_compress(h, msg, c_sigma, c_u256, 0x280);

		for (int i = 7; i >= 0; i--) {
			uint32_t hash = cuda_swab32(h[i]);
			if (hash > c_Target[i]) {
				return;
			}
			if (hash < c_Target[i]) {
				break;
			}
		}

		/* keep the smallest nounce, hmm... */
		if(resNounce[0] > nounce)
			resNounce[0] = nounce;
	}
}

__host__
uint32_t blake256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce)
{
	const int threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	uint32_t result = 0xffffffffU;
	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	blake256_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_resNounce[thr_id]);
	MyStreamSynchronize(NULL, 1, thr_id);

	if (cudaSuccess == cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		cudaThreadSynchronize();
		result = *h_resNounce[thr_id];
	}
	return result;
}

__host__
void blake256_cpu_setBlock_80(uint32_t *pdata, const void *ptarget)
{
	uint32_t PaddedMessage[32];
	memcpy(PaddedMessage, pdata, 80);
	memset(&PaddedMessage[20], 0, 48);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, sizeof(PaddedMessage), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sigma, host_sigma, sizeof(host_sigma), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Target, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

extern "C" int scanhash_blake32(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	static bool init[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	uint32_t throughput = min(TPB * 2048, max_nonce - first_nonce);
	int rc = 0;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00000f;

	if (!init[thr_id]) {
		if (opt_n_threads > 1) {
			CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		}
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNounce[thr_id], sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNounce[thr_id], sizeof(uint32_t)));
		init[thr_id] = true;
	}

	if (throughput < (TPB * 2048))
		applog(LOG_WARNING, "throughput=%u, start=%x, max=%x", throughput, first_nonce, max_nonce);

	blake256_cpu_setBlock_80(pdata, (void*)ptarget);

	do {
		// GPU HASH
		uint32_t foundNonce = blake256_cpu_hash_80(thr_id, throughput, pdata[19]);
		if (foundNonce != 0xffffffff)
		{
			uint32_t endiandata[20];
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[7];

			for (int k=0; k < 20; k++)
				be32enc(&endiandata[k], pdata[k]);

			if (opt_debug && !opt_quiet) {
				applog(LOG_DEBUG, "throughput=%u, start=%x, max=%x, pdata=%08x...%08x",
					throughput, first_nonce, max_nonce, endiandata[0], endiandata[7]);
				applog_hash((unsigned char *)pdata);
			}

			be32enc(&endiandata[19], foundNonce);

			blake32hash(vhashcpu, endiandata);

			if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				pdata[19] = foundNonce;
				rc = 1;
				goto exit_scan;
			}
			else if (vhashcpu[7] > Htarg) {
				applog(LOG_WARNING, "GPU #%d: result for nounce %08x is not in range: %x > %x", thr_id, foundNonce, vhashcpu[7], Htarg);
			}
			else if (vhashcpu[6] > ptarget[6]) {
				applog(LOG_WARNING, "GPU #%d: hash[6] for nounce %08x is not in range: %x > %x", thr_id, foundNonce, vhashcpu[6], ptarget[6]);
			}
			else {
				applog(LOG_WARNING, "GPU #%d: result for nounce %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

exit_scan:
	*hashes_done = pdata[19] - first_nonce + 1;
	// reset the device to allow multiple instances
	if (opt_n_threads == 1) {
		CUDA_SAFE_CALL(cudaDeviceReset());
		init[thr_id] = false;
	}
	// wait proper end of all threads
	cudaDeviceSynchronize();
	return rc;
}
