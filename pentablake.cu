/**
 * Penta Blake-512 Cuda Kernel (Tested on SM 5.0)
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
#define TPB 192

/* hash by cpu with blake 256 */
extern "C" void pentablakehash(void *output, const void *input)
{
	unsigned char hash[128];
	#define hashB hash + 64
	sph_blake512_context ctx;

	sph_blake512_init(&ctx);
	sph_blake512(&ctx, input, 80);
	sph_blake512_close(&ctx, hash);

	sph_blake512(&ctx, hash, 64);
	sph_blake512_close(&ctx, hashB);

	sph_blake512(&ctx, hashB, 64);
	sph_blake512_close(&ctx, hash);

	sph_blake512(&ctx, hash, 64);
	sph_blake512_close(&ctx, hashB);

	sph_blake512(&ctx, hashB, 64);
	sph_blake512_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

__constant__
static uint32_t __align__(32) c_Target[8];

__constant__
static uint64_t __align__(32) c_data[32];

static uint32_t *d_hash[8];
static uint32_t *d_resNounce[8];
static uint32_t *h_resNounce[8];
static uint32_t extra_results[2] = { UINT32_MAX, UINT32_MAX };

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
static const uint64_t __align__(32) c_IV512[8] = {
	0x6a09e667f3bcc908ULL,
	0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL,
	0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL,
	0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL,
	0x5be0cd19137e2179ULL
};

__device__ __constant__
const uint64_t c_u512[16] =
{
	0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
	0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
	0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL,
	0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
	0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL,
	0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
	0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL,
	0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};

#define G(a,b,c,d,x) { \
	uint32_t idx1 = c_sigma[i][x]; \
	uint32_t idx2 = c_sigma[i][x+1]; \
	v[a] += (m[idx1] ^ c_u512[idx2]) + v[b]; \
	v[d] = SWAPDWORDS(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 25); \
	v[a] += (m[idx2] ^ c_u512[idx1]) + v[b]; \
	v[d] = ROTR64(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR64(v[b] ^ v[c], 11); \
}

// Hash-Padding
__device__ __constant__
static const uint64_t d_constHashPadding[8] = {
	0x0000000000000080ull,
	0,
	0,
	0,
	0,
	0x0100000000000000ull,
	0,
	0x0002000000000000ull
};

#if 0

__device__ __constant__
static const uint64_t __align__(32) c_Padding[16] = {
	0, 0, 0, 0,
	0x80000000ULL, 0, 0, 0,
	0, 0, 0, 0,
	0, 1, 0, 640,
};

__device__ static
void pentablake_compress(uint64_t *h, const uint64_t *block, const uint32_t T0)
{
	uint64_t v[16], m[16];

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

	v[ 8] = c_u512[0];
	v[ 9] = c_u512[1];
	v[10] = c_u512[2];
	v[11] = c_u512[3];

	v[12] = xor1(c_u512[4], T0);
	v[13] = xor1(c_u512[5], T0);
	v[14] = c_u512[6];
	v[15] = c_u512[7];

	for (uint32_t i = 0; i < 16; i++) {
		/* column step */
		G(0, 4, 0x8, 0xC, 0x0);
		G(1, 5, 0x9, 0xD, 0x2);
		G(2, 6, 0xA, 0xE, 0x4);
		G(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		G(0, 5, 0xA, 0xF, 0x8);
		G(1, 6, 0xB, 0xC, 0xA);
		G(2, 7, 0x8, 0xD, 0xC);
		G(3, 4, 0x9, 0xE, 0xE);
	}

	//#pragma unroll 16
	for (uint32_t i = 0; i < 16; i++) {
		uint32_t j = i % 8;
		h[j] ^= v[i];
	}
}

__global__
void pentablake_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint64_t h[8];

		#pragma unroll
		for(int i=0; i<8; i++) {
			h[i] = c_IV512[i];
		}

		uint64_t ending[4];
		ending[0] = c_data[16];
		ending[1] = c_data[17];
		ending[2] = c_data[18];
		ending[3] = nounce; /* our tested value */

		pentablake_compress(h, ending, 640);

		// -----------------------------------

		for (int r = 0; r < 4; r++) {
			uint64_t data[8];
			for (int i = 0; i < 7; i++) {
				data[i] = h[i];
			}
			pentablake_compress(h, data, 512); /* todo: use h,h when ok*/
		}
	}
}
#endif

__device__ static
void pentablake_compress(uint64_t *h, const uint64_t *block, const uint64_t T0)
{
	uint64_t v[16], m[16], i;

	#pragma unroll 16
	for(i = 0; i < 16; i++) {
		m[i] = cuda_swab64(block[i]);
	}

	#pragma unroll 8
	for (i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = c_u512[0];
	v[ 9] = c_u512[1];
	v[10] = c_u512[2];
	v[11] = c_u512[3];
	v[12] = c_u512[4] ^ T0;
	v[13] = c_u512[5] ^ T0;
	v[14] = c_u512[6];
	v[15] = c_u512[7];

	//#pragma unroll 16
	for( i = 0; i < 16; i++)
	{
		/* column step */
		G(0, 4, 0x8, 0xC, 0x0);
		G(1, 5, 0x9, 0xD, 0x2);
		G(2, 6, 0xA, 0xE, 0x4);
		G(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		G(0, 5, 0xA, 0xF, 0x8);
		G(1, 6, 0xB, 0xC, 0xA);
		G(2, 7, 0x8, 0xD, 0xC);
		G(3, 4, 0x9, 0xE, 0xE);
	}

	//#pragma unroll 16
	for (i = 0; i < 16; i++) {
		uint32_t idx = i % 8;
		h[idx] ^= v[i];
	}
}

__global__
void pentablake_gpu_hash_80(int threads, const uint32_t startNounce, void *outputHash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t h[8];
		uint64_t buf[16];
		uint32_t nounce = startNounce + thread;

		//#pragma unroll 8
		for(int i=0; i<8; i++)
			h[i] = c_IV512[i];

		//#pragma unroll 16
		for (int i=0; i < 16; i++)
			buf[i] = c_data[i];

		// The test Nonce
		((uint32_t*)buf)[19] = cuda_swab32(nounce);

		pentablake_compress(h, buf, 640ULL);

#if __CUDA_ARCH__ < 300
		uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;
		#pragma unroll 8
		for (uint32_t i=0; i < 8; i++) {
			outHash[2*i]   = cuda_swab32( _HIWORD(h[i]) );
			outHash[2*i+1] = cuda_swab32( _LOWORD(h[i]) );
		}
#else
		uint64_t *outHash = (uint64_t *)outputHash + 8 * thread;
		for (uint32_t i=0; i < 8; i++) {
			outHash[i] = cuda_swab64( h[i] );
		}
#endif

	}
}

__host__
void pentablake_cpu_hash_80(int thr_id, int threads, const uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	pentablake_gpu_hash_80 <<<grid, block, shared_size>>> (threads, startNounce, d_outputHash);

	//MyStreamSynchronize(NULL, order, thr_id);
	cudaDeviceSynchronize();
}


__global__
void pentablake_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint64_t *inpHash = &g_hash[thread<<3]; // hashPosition * 8
		uint64_t buf[16]; // 128 Bytes
		uint64_t h[8]; // State

		#pragma unroll 8
		for (int i=0; i<8; i++)
			h[i] = c_IV512[i];

		// Message for first round
		#pragma unroll 8
		for (int i=0; i < 8; ++i)
			buf[i] = inpHash[i];

		#pragma unroll 8
		for (int i=0; i < 8; i++)
			buf[i+8] = d_constHashPadding[i];

		// Ending round
		pentablake_compress(h, buf, 512);

#if __CUDA_ARCH__ < 300
		uint32_t *outHash = (uint32_t*)&g_hash[thread<<3];
		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			outHash[2*i+0] = cuda_swab32( _HIWORD(h[i]) );
			outHash[2*i+1] = cuda_swab32( _LOWORD(h[i]) );
		}
#else
		uint64_t *outHash = &g_hash[thread<<3];
		for (int i=0; i < 8; i++) {
			outHash[i] = cuda_swab64(h[i]);
		}
#endif
	}
}

__host__
void pentablake_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	pentablake_gpu_hash_64 <<<grid, block, shared_size>>> (threads, startNounce, (uint64_t*)d_outputHash);

	//MyStreamSynchronize(NULL, order, thr_id);
	cudaDeviceSynchronize();
}

#if 0

__host__
uint32_t pentablake_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce)
{
	const int threadsperblock = TPB;
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNounce[thr_id], 0xff, 2*sizeof(uint32_t)) != cudaSuccess)
		return result;

	pentablake_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_resNounce[thr_id]);
	cudaDeviceSynchronize();
	if (cudaSuccess == cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		cudaThreadSynchronize();
		result = h_resNounce[thr_id][0];
		extra_results[0] = h_resNounce[thr_id][1];
	}
	return result;
}
#endif

__global__
void pentablake_gpu_check_hash(uint32_t threads, uint32_t startNounce, uint32_t *g_hash, uint32_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;
		uint32_t *inpHash = &g_hash[thread<<4];
		uint32_t h[8];

		#pragma unroll 8
		for (int i=0; i < 8; i++)
			h[i] = inpHash[i];

		for (int i = 7; i >= 0; i--) {
			uint32_t hash = h[i]; // cuda_swab32(h[i]);
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

__host__ static
uint32_t pentablake_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, int order)
{
	const int threadsperblock = TPB;
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNounce[thr_id], 0xff, 2*sizeof(uint32_t)) != cudaSuccess)
		return result;

	pentablake_gpu_check_hash <<<grid, block, shared_size>>> (threads, startNounce, d_inputHash, d_resNounce[thr_id]);

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	if (cudaSuccess == cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		cudaThreadSynchronize();
		result = h_resNounce[thr_id][0];
		extra_results[0] = h_resNounce[thr_id][1];
	}
	return result;
}


__host__
void pentablake_cpu_setBlock_80(uint32_t *pdata, const uint32_t *ptarget)
{
	uint8_t data[128];
	memcpy((void*) data, (void*) pdata, 80);
	memset(data+80, 0, 48);

	// to swab...
	data[80] = 0x80;
	data[111] = 1;
	data[126] = 0x02;
	data[127] = 0x80;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, data, sizeof(data), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_sigma, host_sigma, sizeof(host_sigma), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_Target, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

static bool init[8] = { 0 };

extern "C" int scanhash_pentablake(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	int rc = 0;
	int throughput = opt_work_size ? opt_work_size : (128 * 2560); // 18.5
	throughput = min(throughput, (int)(max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x000F;

	if (!init[thr_id]) {
		if (active_gpus > 1) {
			cudaSetDevice(device_map[thr_id]);
		}
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 64 * throughput));
		CUDA_SAFE_CALL(cudaMallocHost(&h_resNounce[thr_id], 2*sizeof(uint32_t)));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNounce[thr_id], 2*sizeof(uint32_t)));

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	pentablake_cpu_setBlock_80(endiandata, ptarget);

	do {
		int order = 0;

		// GPU HASH
		pentablake_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		pentablake_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		pentablake_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		pentablake_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		pentablake_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		uint32_t foundNonce = pentablake_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if (foundNonce != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhashcpu[8];

			be32enc(&endiandata[19], foundNonce);
			pentablakehash(vhashcpu, endiandata);

			if (vhashcpu[7] <= Htarg && fulltest(vhashcpu, ptarget)) {
				rc = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (extra_results[0] != UINT32_MAX) {
					// Rare but possible if the throughput is big
					applog(LOG_NOTICE, "GPU found more than one result yippee!");
					pdata[21] = extra_results[0];
					extra_results[0] = UINT32_MAX;
					rc++;
				}
				pdata[19] = foundNonce;
				return rc;
			}
			else if (vhashcpu[7] > Htarg) {
				applog(LOG_WARNING, "GPU #%d: result for nounce %08x is not in range: %x > %x", thr_id, foundNonce, vhashcpu[7], Htarg);
			}
			else {
				applog(LOG_WARNING, "GPU #%d: result for nounce %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	cudaDeviceSynchronize();
	return rc;
}
