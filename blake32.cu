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

#if __CUDA_ARCH__ < 350
	// Kepler (Compute 3.0) + Host
	#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
	// Kepler (Compute 3.5 / 5.0)
	#define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

// in cpu-miner.c
extern bool opt_benchmark;
extern bool opt_debug;
extern int device_map[8];

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// shared for 8 threads of addresses (cudaMalloc)
uint32_t* d_hash[8];

__constant__
static uint32_t pTarget[8];

__constant__
static uint32_t c_PaddedMessage80[32]; // padded message (80 bytes + padding)
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
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[sigma[i][e+1]] ^ u256[sigma[i][e]]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

__device__ static
void blake256_compress(uint32_t *h, uint32_t *block, uint8_t ((*sigma)[16]), const uint32_t *u256, const uint32_t T0, uint8_t nullt = 1)
{
	uint32_t /* __align__(8) */ v[16];
	uint32_t /* __align__(8) */ m[16];

	//#pragma unroll
	for (int i = 0; i < 16; ++i) {
		m[i] = cuda_swab32(block[i]);
		//m[i] = block[i];
	}

	#pragma unroll
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

	// on a 80-bytes null buffer :
	// first : v = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, ...}
	// second : v = {0xb5bfb2f9, 0x14cfcc63, 0xb85c549c, 0xc9b4184e, ..., 0x299f3350, 0x082efa98, 0xec4e6c89}

	//#pragma unroll
	for (int i = 0; i < 14; i++) {
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

	//second H0 = 0x0c7b1594 ... H7 = 0x9051b305
}

#if __CUDA_ARCH__ >= 200
#if (__NV_POINTER_SIZE == 64)
# define SZCT uint64_t
#else
# define SZCT uint32_t
#endif
extern __device__ __device_builtin__ void __nvvm_memset(uint8_t *, unsigned char, SZCT, int);
#endif

__global__
void blake256_gpu_hash_80(int threads, uint32_t startNounce, void *outputHash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t /* __align__(16) */ h[8];
		uint32_t /* __align__(16) */ msg[16];
		const uint32_t nounce = startNounce + thread;

		#pragma unroll
		for(int i=0; i<8; i++)
			h[i] = c_IV256[i];

		blake256_compress(h, c_PaddedMessage80, c_sigma, c_u256, 0x200); /* 512 = 0x200 */

		// ------ Close: Bytes 64 to 80 ------ 

#if 0 /* __CUDA_ARCH__ >= 200 */
		__nvvm_memset((uint8_t*)(&msg[4]), 0, sizeof(msg)-16, 16);
#else
		msg[5] = 0;
		msg[6] = 0;
		msg[7] = 0;
		msg[8] = 0;
		msg[9] = 0;
		msg[10] = 0;
		msg[11] = 0;
		msg[12] = 0;
		msg[14] = 0;
#endif
		msg[0] = c_PaddedMessage80[16];
		msg[1] = c_PaddedMessage80[17];
		msg[2] = c_PaddedMessage80[18];
		msg[3] = cuda_swab32(nounce); // here or at 80 ?

		msg[4] = 0x80; // uchar[16] after buffer
		msg[13] = 0x01000000; //((uint8_t*)msg)[55] = 1; // uchar[17 to 55]
		msg[15] = 0x80020000; // 60-63 0x280

		//h => {0xb5bfb2f9, 0x14cfcc63, 0xb85c549c, 0xc9b4184e, 0x67dfc6ce, 0x29e9904b, 0xd59ee74e, 0xfaa9c653}
		//msg  {0, 0, 0, 0, 0x80, 0...}

		blake256_compress(h, msg, c_sigma, c_u256, 0x280); // or 0x80
		//h => {0x0c7b1594, 0x52328517, 0x463db487, 0xdf5e39b7, 0x1322afaf, 0x14ed562c, 0xe9d18d7d, 0x9051b305}

		uint32_t *outHash = (uint32_t*) outputHash + 16*thread; // 16 = 4 x sizeof(uint32)
		//#pragma unroll
		for (int i=0; i < 8; i++) {
			outHash[i] = cuda_swab32(h[i]);
		}
	}
}

__host__
void blake256_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	blake256_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}

__global__
void gpu_check_hash_64(int threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *inpHash = &g_hash[16 * hashPosition];
		uint32_t hash[8];

		#pragma unroll 8
		for (int i=0; i < 8; i++)
			hash[i] = inpHash[i];

		int i, position = -1;
		bool rc = true;

		#pragma unroll 8
		for (i = 7; i >= 0; i--) {
			if (hash[i] > pTarget[i] && position < i) {
				position = i;
				rc = false;
			}
			if (hash[i] < pTarget[i] && position < i) {
				position = i;
				rc = true;
			}
		}

		if(rc && resNounce[0] > nounce)
			resNounce[0] = nounce;
	}
}

__host__
uint32_t cpu_check_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order)
{
	uint32_t result = 0xffffffff;
	const int threadsperblock = 256;

	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	gpu_check_hash_64 <<<grid, block, shared_size>>>(threads, startNounce, d_nonceVector, d_inputHash, d_resNounce[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	CUDA_SAFE_CALL(cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// cudaMemcpy() is asynch!
	cudaThreadSynchronize();
	result = *h_resNounce[thr_id];

	return result;
}

__host__
void blake256_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sigma, host_sigma, sizeof(host_sigma), 0, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMallocHost(&h_resNounce[thr_id], 1*sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc(&d_resNounce[thr_id], 1*sizeof(uint32_t)));
}

__host__
void blake256_cpu_setBlock_80(uint32_t *pdata, const void *ptarget)
{
	uint32_t PaddedMessage[32];
	memcpy(PaddedMessage, pdata, 80);
	memset(&PaddedMessage[20], 0, 48);
	//for (int i=0; i<20; i++)
	//	PaddedMessage[i] = cuda_swab32(pdata[i]);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, ptarget, 32, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, sizeof(PaddedMessage), 0, cudaMemcpyHostToDevice));
}

#define NULLTEST 0

extern "C" int scanhash_blake32(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t endiandata[20];
	const uint32_t first_nonce = pdata[19];
	const int throughput = 256*256*2;
	static bool init[8] = {0,0,0,0,0,0,0,0};

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00000f;

	uint32_t Htarg = ptarget[7];

	if (!init[thr_id]) {
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));

		blake256_cpu_init(thr_id);

		init[thr_id] = true;
	}

#if NULLTEST
	// dev test with a null buffer 0x00000...
	for (int k = 0; k < 20; k++)
		pdata[k] = 0;
	uint32_t vhash[8];
	blake32hash(vhash, pdata);
#endif

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_cpu_setBlock_80(endiandata, (void*)ptarget);

	do {
		int order = 0;
		uint32_t foundNonce;

		// GPU
		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

#if NULLTEST
		uint32_t buf[8]; memset(buf, 0, sizeof buf);
		CUDA_SAFE_CALL(cudaMemcpy(buf, d_hash[thr_id], sizeof buf, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		//applog_hash((unsigned char*)buf);
#endif
		foundNonce = cpu_check_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if (foundNonce != 0xffffffff)
		{
			uint32_t vhashcpu[8];
			be32enc(&endiandata[19], foundNonce);

			blake32hash(vhashcpu, endiandata);

			if (opt_debug)
				applog(LOG_DEBUG, "foundNonce = %08x",foundNonce);

			if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				pdata[19] = foundNonce;
				*hashes_done = pdata[19] - first_nonce + 1;
				return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}

//#define DEBUG_ALGO

__host__
int scanhash_blake256_cpu(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, uint64_t *hashes_done)
{
	uint32_t n = pdata[19] - 1;
	const uint32_t first_nonce = pdata[19];
	const uint32_t Htarg = ptarget[7];

	uint32_t __align__(32) hash64[8];
	uint32_t endiandata[32];

	uint64_t htmax[] = {
		0,
		0xF,
		0xFF,
		0xFFF,
		0xFFFF,
		0x10000000
	};
	uint32_t masks[] = {
		0xFFFFFFFF,
		0xFFFFFFF0,
		0xFFFFFF00,
		0xFFFFF000,
		0xFFFF0000,
		0
	};

	// we need bigendian data...
	for (int kk=0; kk < 32; kk++) {
		be32enc(&endiandata[kk], ((uint32_t*)pdata)[kk]);
	};
#ifdef DEBUG_ALGO
	if (Htarg != 0)
		printf("[%d] Htarg=%X\n", thr_id, Htarg);
#endif
	for (int m=0; m < 6; m++) {
		if (Htarg <= htmax[m]) {
			uint32_t mask = masks[m];
			do {
				pdata[19] = ++n;
				be32enc(&endiandata[19], n);
				blake32hash(hash64, endiandata);
#ifndef DEBUG_ALGO
				if ((!(hash64[7] & mask)) && fulltest(hash64, ptarget)) {
					*hashes_done = n - first_nonce + 1;
					return true;
				}
#else
				if (!(n % 0x1000) && !thr_id) printf(".");
				if (!(hash64[7] & mask)) {
					printf("[%d]",thr_id);
					if (fulltest(hash64, ptarget)) {
						*hashes_done = n - first_nonce + 1;
						return true;
					}
				}
#endif
			} while (n < max_nonce && !work_restart[thr_id].restart);
			// see blake.c if else to understand the loop on htmax => mask
			break;
		}
	}

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	return 0;
}
