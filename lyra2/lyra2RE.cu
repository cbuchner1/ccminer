
extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/Lyra2.h"

#include "miner.h"
}

#include <stdint.h>

// aus cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint64_t *d_hash[8];



extern void quark_check_cpu_init(int thr_id, int threads);
extern void quark_check_cpu_setTarget(const void *ptarget);
extern uint32_t quark_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);
extern uint32_t quark_check_cpu_hash_64_2(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint64_t *d_inputHash, int order);


extern void blake256_cpu_init(int thr_id, int threads);
extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);
extern void keccak256_cpu_hash_32(int thr_id, int threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void keccak256_cpu_init(int thr_id, int threads);
extern void skein256_cpu_hash_32(int thr_id, int threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void skein256_cpu_init(int thr_id, int threads);

extern void lyra2_cpu_hash_32(int thr_id, int threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void lyra2_cpu_init(int thr_id, int threads);

extern void groestl256_setTarget(const void *ptarget);
extern uint32_t groestl256_cpu_hash_32(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern void groestl256_cpu_init(int thr_id, int threads);
extern uint32_t groestl256_cpu64_hash_32(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern void groestl256_cpu64_init(int thr_id, int threads);


// X11 Hashfunktion
inline void lyra_hash(void *state, const void *input)
{
    // blake1-bmw2-grs3-skein4-jh5-keccak6-luffa7-cubehash8-shavite9-simd10-echo11
	sph_blake256_context     ctx_blake;
	sph_groestl256_context   ctx_groestl;
	sph_keccak256_context    ctx_keccak;
	sph_skein256_context     ctx_skein;

	uint32_t hashA[8], hashB[8], hash[8];
	uint32_t * data = (uint32_t*)input;
//	for (int i = 0; i<10; i++)	{ printf("cpu data %d %08x %08x\n", i, data[2*i],data[2*i+1]); }
	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);
	
	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hash);
//for (int i = 0; i<4; i++)	{ printf("cpu groestl %d %08x %08x\n", i, hash[2 * i], hash[2 * i + 1]); }
    memcpy(state, hash, 32);
}

extern float tp_coef[8];
extern bool opt_benchmark;

extern "C" int scanhash_lyra(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];
	if (tp_coef[thr_id]<0) { tp_coef[thr_id] = 4.; }
	const int throughput = (int) (256*256*tp_coef[thr_id]);

	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]); 
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint32_t) * throughput);
		blake256_cpu_init(thr_id, throughput);
		keccak256_cpu_init(thr_id,throughput);
		skein256_cpu_init(thr_id, throughput);
		lyra2_cpu_init(thr_id, throughput);
		groestl256_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]); 
	blake256_cpu_setBlock_80(pdata);
	groestl256_setTarget(ptarget); 

	do {
		int order = 0;

		// erstes Blake512 Hash mit CUDA
		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		lyra2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);


		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		// Scan nach Gewinner Hashes auf der GPU
 uint32_t	foundNonce = groestl256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
//foundNonce = pdata[19]+10;
		if  (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			//pdata[19]=foundNonce;
//			lyra_hash(vhash64, endiandata);

//			if ( ((uint64_t*)vhash64)[3] <= ((uint64_t*)ptarget)[3]) { // && fulltest(vhash64, ptarget)) {
//				printf("target %08x %08x %08x %08x\n", ptarget[0], ptarget[1], ptarget[2], ptarget[3]);
//				printf("target %08x %08x %08x %08x\n", ptarget[4], ptarget[5], ptarget[6], ptarget[7]);

				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
//			} else {
//				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
//			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
