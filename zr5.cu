/* Ziftrcoin ZR5 CUDA Implementation, (c) tpruvot 2015 */

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include <stdio.h>
#include <memory.h>

#define ZR_BLAKE   0
#define ZR_GROESTL 1
#define ZR_JH512   2
#define ZR_SKEIN   3

#define POK_BOOL_MASK 0x00008000
#define POK_DATA_MASK 0xFFFF0000

static uint32_t* d_hash[MAX_GPUS];
static uint16_t* d_pokh[MAX_GPUS];
static uint16_t* h_poks[MAX_GPUS];

static uint32_t* d_blake[MAX_GPUS];
static uint32_t* d_groes[MAX_GPUS];
static uint32_t* d_jh512[MAX_GPUS];
static uint32_t* d_skein[MAX_GPUS];

__constant__ uint8_t d_permut[24][4];
static const uint8_t permut[24][4] = {
	{0, 1, 2, 3},
	{0, 1, 3, 2},
	{0, 2, 1, 3},
	{0, 2, 3, 1},
	{0, 3, 1, 2},
	{0, 3, 2, 1},
	{1, 0, 2, 3},
	{1, 0, 3, 2},
	{1, 2, 0, 3},
	{1, 2, 3, 0},
	{1, 3, 0, 2},
	{1, 3, 2, 0},
	{2, 0, 1, 3},
	{2, 0, 3, 1},
	{2, 1, 0, 3},
	{2, 1, 3, 0},
	{2, 3, 0, 1},
	{2, 3, 1, 0},
	{3, 0, 1, 2},
	{3, 0, 2, 1},
	{3, 1, 0, 2},
	{3, 1, 2, 0},
	{3, 2, 0, 1},
	{3, 2, 1, 0}
};

// CPU HASH
extern "C" void zr5hash(void *output, const void *input)
{
	sph_keccak512_context ctx_keccak;
	sph_blake512_context ctx_blake;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_skein512_context ctx_skein;

	uchar _ALIGN(64) hash[64];
	uint32_t *phash = (uint32_t *) hash;
	uint32_t norder;

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*) input, 80);
	sph_keccak512_close(&ctx_keccak, (void*) phash);

	norder = phash[0] % ARRAY_SIZE(permut); /* % 24 */

	for(int i = 0; i < 4; i++)
	{
		switch (permut[norder][i]) {
		case ZR_BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, (const void*) phash, 64);
			sph_blake512_close(&ctx_blake, phash);
			break;
		case ZR_GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, (const void*) phash, 64);
			sph_groestl512_close(&ctx_groestl, phash);
			break;
		case ZR_JH512:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, (const void*) phash, 64);
			sph_jh512_close(&ctx_jh, phash);
			break;
		case ZR_SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, (const void*) phash, 64);
			sph_skein512_close(&ctx_skein, phash);
			break;
		default:
			break;
		}
	}
	memcpy(output, phash, 32);
}

extern "C" void zr5hash_pok(void *output, uint32_t *pdata)
{
	const uint32_t version = pdata[0] & (~POK_DATA_MASK);
	uint32_t _ALIGN(64) hash[8];

	pdata[0] = version;
	zr5hash(hash, pdata);

	// fill PoK
	pdata[0] = version | (hash[0] & POK_DATA_MASK);
	zr5hash(hash, pdata);

	memcpy(output, hash, 32);
}

__global__
void zr5_copy_round_data_gpu(uint32_t threads, uint32_t *d_hash, uint32_t* d_blake, uint32_t* d_groes, uint32_t* d_jh512, uint32_t* d_skein, int rnd)
{
	// copy 64 bytes hash in the right algo buffer
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint64_t offset = thread * 64 / 4;
		uint32_t *phash = &d_hash[offset];
		// algos hash order
		uint32_t norder = phash[0] % ARRAY_SIZE(permut);
		uint32_t algo = d_permut[norder][rnd];
		uint32_t* buffers[4] = { d_blake, d_groes, d_jh512, d_skein };

		if (rnd > 0) {
			int algosrc = d_permut[norder][rnd - 1];
			phash = buffers[algosrc] + offset;
		}

		// uint4 = 4x4 uint32_t = 16 bytes
		uint4 *psrc = (uint4*) phash;
		uint4 *pdst = (uint4*) (buffers[algo] + offset);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__host__
void zr5_move_data_to_hash(int thr_id, uint32_t threads, int rnd)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	zr5_copy_round_data_gpu <<<grid, block>>> (threads, d_hash[thr_id], d_blake[thr_id], d_groes[thr_id], d_jh512[thr_id], d_skein[thr_id], rnd);
}

__global__
void zr5_final_round_data_gpu(uint32_t threads, uint32_t* d_blake, uint32_t* d_groes, uint32_t* d_jh512, uint32_t* d_skein, uint32_t *d_hash, uint16_t *d_pokh)
{
	// after the 4 algos rounds, copy back hash to d_hash
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint64_t offset = thread * 16; // 64 / 4;
		uint32_t *phash = &d_hash[offset];
		uint16_t norder = phash[0] % ARRAY_SIZE(permut);
		uint16_t algosrc = d_permut[norder][3];

		uint32_t* buffers[4] = { d_blake, d_groes, d_jh512, d_skein };

		// copy only hash[0] + hash[6..7]
		uint2 *psrc = (uint2*) (buffers[algosrc] + offset);
		uint2 *pdst = (uint2*) phash;

		pdst[0].x = psrc[0].x;
		pdst[3] = psrc[3];

		//phash[7] = *(buffers[algosrc] + offset + 7);
	}
}

__host__
void zr5_final_round(int thr_id, uint32_t threads)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	zr5_final_round_data_gpu <<<grid, block>>> (threads, d_blake[thr_id], d_groes[thr_id], d_jh512[thr_id], d_skein[thr_id], d_hash[thr_id], d_pokh[thr_id]);
}

extern void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);

extern void zr5_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void zr5_keccak512_cpu_hash_pok(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* pdata, uint32_t *d_hash, uint16_t *d_poks);

extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_jh512_cpu_init(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_zr5(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) tmpdata[20];
	const uint32_t version = pdata[0] & (~POK_DATA_MASK);
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput =  device_intensity(thr_id, __func__, 1U << 18);
	throughput = min(throughput, (1U << 20)-1024);
	throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	memcpy(tmpdata, pdata, 80);

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		// hash buffer = keccak hash 64 required
		cudaMalloc(&d_hash[thr_id], 64 * throughput);
		cudaMalloc(&d_pokh[thr_id], 2 * throughput);

		cudaMemcpyToSymbol(d_permut, permut, 24*4, 0, cudaMemcpyHostToDevice);
		cudaMallocHost(&h_poks[thr_id], 2 * throughput);

		// data buffers for the 4 rounds
		cudaMalloc(&d_blake[thr_id], 64 * throughput);
		cudaMalloc(&d_groes[thr_id], 64 * throughput);
		cudaMalloc(&d_jh512[thr_id], 64 * throughput);
		cudaMalloc(&d_skein[thr_id], 64 * throughput);

		jackpot_keccak512_cpu_init(thr_id, throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		init[thr_id] = true;
	}

	tmpdata[0] = version;
	jackpot_keccak512_cpu_setBlock((void*)tmpdata, 80);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Keccak512 Hash with CUDA
		zr5_keccak512_cpu_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);

		for (int rnd=0; rnd<4; rnd++) {
			zr5_move_data_to_hash(thr_id, throughput, rnd);
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_blake[thr_id], order++);
			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_groes[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_jh512[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_skein[thr_id], order++);
		}

		// This generates all pok prefixes
		zr5_final_round(thr_id, throughput);

		// Keccak512 pok
		zr5_keccak512_cpu_hash_pok(thr_id, throughput, pdata[19], pdata, d_hash[thr_id], d_pokh[thr_id]);

		for (int rnd=0; rnd<4; rnd++) {
			zr5_move_data_to_hash(thr_id, throughput, rnd);
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_blake[thr_id], order++);
			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_groes[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_jh512[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_skein[thr_id], order++);
		}
		zr5_final_round(thr_id, throughput);

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhash64[8];
			uint32_t oldp0 = pdata[0];
			uint32_t oldp19 = pdata[19];
			uint32_t offset = foundNonce - pdata[19];
			uint32_t pok = 0;

			*hashes_done = pdata[19] - first_nonce + throughput;

			cudaMemcpy(h_poks[thr_id], d_pokh[thr_id], 2 * throughput, cudaMemcpyDeviceToHost);
			pok = version | (0x10000UL * h_poks[thr_id][offset]);
			pdata[0] = pok; pdata[19] = foundNonce;
			zr5hash(vhash64, pdata);
			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (secNonce != 0) {
					offset = secNonce - oldp19;
					pok = version | (0x10000UL * h_poks[thr_id][offset]);
					memcpy(tmpdata, pdata, 80);
					tmpdata[0] = pok; tmpdata[19] = secNonce;
					zr5hash(vhash64, tmpdata);
					if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
						pdata[21] = secNonce;
						pdata[22] = pok;
						res++;
					}
				}
				return res;
			} else {
				applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce);
				pdata[19]++;
				pdata[0] = oldp0;
			}
		} else
			pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
