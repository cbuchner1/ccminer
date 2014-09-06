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

/* crc32.c */
extern "C" uint32_t crc32_u32t(const uint32_t *buf, size_t size);

extern "C" int blake256_rounds = 14;

/* hash by cpu with blake 256 */
extern "C" void blake256hash(void *output, const void *input, int rounds = 14)
{
	unsigned char hash[64];
	sph_blake256_context ctx;

	/* in sph_blake.c */
	blake256_rounds = rounds;

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);
	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

#define MAXU 0xffffffffU

// in cpu-miner.c
extern bool opt_n_threads;
extern bool opt_benchmark;
extern int device_map[8];

__constant__
static uint32_t __align__(32) c_Target[8];

__constant__
static uint32_t __align__(32) c_data[20];

static uint32_t *d_resNounce[8];
static uint32_t *h_resNounce[8];
static uint32_t extra_results[2] = { MAXU, MAXU };

#define USE_CACHE 1
#if USE_CACHE
__device__
static uint32_t cache[8];
__device__
static uint32_t prevsum = 0;
#endif

/* prefer uint32_t to prevent size conversions = speed +5/10 % */
__constant__
static uint32_t __align__(32) c_sigma[16][16];
const uint32_t host_sigma[16][16] = {
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
static const uint32_t __align__(32) c_IV256[8] = {
	SPH_C32(0x6A09E667), SPH_C32(0xBB67AE85),
	SPH_C32(0x3C6EF372), SPH_C32(0xA54FF53A),
	SPH_C32(0x510E527F), SPH_C32(0x9B05688C),
	SPH_C32(0x1F83D9AB), SPH_C32(0x5BE0CD19)
};

__device__ __constant__
static const uint32_t __align__(32) c_u256[16] = {
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

#define GS(a,b,c,d,x) { \
	const uint32_t idx1 = c_sigma[i][x]; \
	const uint32_t idx2 = c_sigma[i][x+1]; \
	v[a] += (m[idx1] ^ c_u256[idx2]) + v[b]; \
	v[d] = SPH_ROTL32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ c_u256[idx1]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
}

/* Second part (64-80) msg never change, store it */
__device__ __constant__
static const uint32_t __align__(32) c_Padding[16] = {
	0, 0, 0, 0,
	0x80000000UL, 0, 0, 0,
	0, 0, 0, 0,
	0, 1, 0, 640,
};

__device__ static
void blake256_compress(uint32_t *h, const uint32_t *block, const uint32_t T0, int blakerounds)
{
	uint32_t /* __align__(8) */ m[16];
	uint32_t /* __align__(8) */ v[16];

	m[0] = block[0];
	m[1] = block[1];
	m[2] = block[2];
	m[3] = block[3];

	for (uint32_t i = 4; i < 16; i++) {
		m[i] = (T0 == 0x200) ? block[i] : c_Padding[i];
	}

	//#pragma unroll 8
	for(uint32_t i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = c_u256[0];
	v[ 9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	for (int i = 0; i < blakerounds; i++) {
		/* column step */
		GS(0, 4, 0x8, 0xC, 0x0);
		GS(1, 5, 0x9, 0xD, 0x2);
		GS(2, 6, 0xA, 0xE, 0x4);
		GS(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		GS(0, 5, 0xA, 0xF, 0x8);
		GS(1, 6, 0xB, 0xC, 0xA);
		GS(2, 7, 0x8, 0xD, 0xC);
		GS(3, 4, 0x9, 0xE, 0xE);
	}

	//#pragma unroll 16
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t j = i % 8;
		h[j] ^= v[i];
	}
}

__global__
void blake256_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *resNounce, const int blakerounds, const int crcsum)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint32_t h[8];

		#pragma unroll
		for(int i=0; i<8; i++) {
			h[i] = c_IV256[i];
		}

#if !USE_CACHE
		blake256_compress(h, c_data, 512, blakerounds);
#else
		if (crcsum != prevsum) {
			prevsum = crcsum;
			blake256_compress(h, c_data, 512, blakerounds);
			#pragma unroll
			for(int i=0; i<8; i++) {
				cache[i] = h[i];
			}
		} else {
			#pragma unroll
			for(int i=0; i<8; i++) {
				h[i] = cache[i];
			}
		}
#endif
		// ------ Close: Bytes 64 to 80 ------ 

		uint32_t ending[4];
		ending[0] = c_data[16];
		ending[1] = c_data[17];
		ending[2] = c_data[18];
		ending[3] = nounce; /* our tested value */

		blake256_compress(h, ending, 640, blakerounds);

		for (int i = 7; i >= 0; i--) {
			uint32_t hash = cuda_swab32(h[i]);
			if (hash > c_Target[i]) {
				return;
			}
			if (hash < c_Target[i]) {
				break;
			}
		}

		/* keep the smallest nounce, + extra one if found */
		if (resNounce[0] > nounce) {
			resNounce[1] = resNounce[0];
			resNounce[0] = nounce;
		}
		else
			resNounce[1] = nounce;
	}
}

__host__
uint32_t blake256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, const int blakerounds, const uint32_t crcsum)
{
	const int threadsperblock = TPB;
	uint32_t result = MAXU;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNounce[thr_id], 0xff, 2*sizeof(uint32_t)) != cudaSuccess)
		return result;

	blake256_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_resNounce[thr_id], blakerounds, crcsum);
	cudaDeviceSynchronize();
	if (cudaSuccess == cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		cudaThreadSynchronize();
		result = h_resNounce[thr_id][0];
		extra_results[0] = h_resNounce[thr_id][1];
	}
	return result;
}

__host__
void blake256_cpu_setBlock_80(uint32_t *pdata, const uint32_t *ptarget)
{
	uint32_t data[20];
	memcpy(data, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, data, sizeof(data), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sigma, host_sigma, sizeof(host_sigma), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Target, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

extern "C" int scanhash_blake256(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done, uint32_t blakerounds=14)
{
	const uint32_t first_nonce = pdata[19];
	static bool init[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	uint32_t throughput = min(TPB * 4096, max_nonce - first_nonce);
	uint32_t crcsum = MAXU;
	int rc = 0;

	if (extra_results[0] != MAXU) {
		// possible extra result found in previous call
		if (first_nonce <= extra_results[0] && max_nonce >= extra_results[0]) {
			pdata[19] = extra_results[0];
			*hashes_done = pdata[19] - first_nonce + 1;
			extra_results[0] = MAXU;
			rc = 1;
			goto exit_scan;
		}
	}

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00000f;

	if (!init[thr_id]) {
		if (opt_n_threads > 1) {
			CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		}
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNounce[thr_id], 2*sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNounce[thr_id], 2*sizeof(uint32_t)));
		init[thr_id] = true;
	}

	if (opt_debug && throughput < (TPB * 4096))
		applog(LOG_DEBUG, "throughput=%u, start=%x, max=%x", throughput, first_nonce, max_nonce);

	blake256_cpu_setBlock_80(pdata, ptarget);
#if USE_CACHE
	crcsum = crc32_u32t(pdata, 64);
#endif

	do {
		// GPU HASH
		uint32_t foundNonce = blake256_cpu_hash_80(thr_id, throughput, pdata[19], blakerounds, crcsum);
		if (foundNonce != MAXU)
		{
			uint32_t endiandata[20];
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[7];

			for (int k=0; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			be32enc(&endiandata[19], foundNonce);

			blake256hash(vhashcpu, endiandata, blakerounds);

			if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				pdata[19] = foundNonce;
				rc = 1;

				if (extra_results[0] != MAXU) {
					// Rare but possible if the throughput is big
					be32enc(&endiandata[19], extra_results[0]);
					blake256hash(vhashcpu, endiandata, blakerounds);
					if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget)) {
						applog(LOG_NOTICE, "GPU found more than one result yippee!");
						rc = 2;
					} else {
						extra_results[0] = MAXU;
					}
				}

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

		if ((uint64_t) pdata[19] + throughput > (uint64_t) max_nonce) {
			pdata[19] = max_nonce - first_nonce + 1;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

exit_scan:
	*hashes_done = pdata[19] - first_nonce + 1;
#if 0
	/* reset the device to allow multiple instances
	 * could be made in cpu-miner... check later if required */
	if (opt_n_threads == 1) {
		CUDA_SAFE_CALL(cudaDeviceReset());
		init[thr_id] = false;
	}
#endif
	// wait proper end of all threads
	//cudaDeviceSynchronize();
	return rc;
}
