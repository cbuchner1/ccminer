/*
 * test routine for new algorithm
 * 
 */

extern "C"
{
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_whirlpool.h"

#include "miner.h"
}

// aus cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint32_t *d_hash[8];

extern void whirlpool512_cpu_init(int thr_id, int threads, int flag);
extern void whirlpool512_setBlock_80(void *pdata);
extern void whirlpool512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);
extern void whirlpool512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_check_cpu_init(int thr_id, int threads);
extern void quark_check_cpu_setTarget(const void *ptarget);

extern uint32_t quark_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);
extern uint32_t test_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);

extern void quark_compactTest_cpu_init(int thr_id, int threads);
extern void quark_compactTest_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *inpHashes, 
											uint32_t *d_noncesTrue, size_t *nrmTrue, uint32_t *d_noncesFalse, size_t *nrmFalse,
											int order);

// fresh Hashfunktion
inline void test_hash(void *state, const void *input)
{
    // shavite-simd-shavite-simd-echo

    
    sph_whirlpool_context ctx_whirlpool;
    
	
    uint32_t hash[16];

	// shavite 1
    sph_whirlpool1_init(&ctx_whirlpool);
    sph_whirlpool1 (&ctx_whirlpool, input, 80);
    sph_whirlpool1_close(&ctx_whirlpool, (void*) hash);
	

	sph_whirlpool1_init(&ctx_whirlpool);
    sph_whirlpool1 (&ctx_whirlpool, (const void*) hash, 64);
    sph_whirlpool1_close(&ctx_whirlpool, (void*) hash); 
	
	sph_whirlpool1_init(&ctx_whirlpool);
    sph_whirlpool1 (&ctx_whirlpool, (const void*) hash, 64);
    sph_whirlpool1_close(&ctx_whirlpool, (void*) hash); 

	sph_whirlpool1_init(&ctx_whirlpool);
    sph_whirlpool1 (&ctx_whirlpool, (const void*) hash, 64);
    sph_whirlpool1_close(&ctx_whirlpool, (void*) hash); 

    memcpy(state, hash, 32);
}


extern bool opt_benchmark;

extern "C" int scanhash_test(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];

	const int throughput = 256*256*8;
	
	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		whirlpool512_cpu_init(thr_id, throughput,1);
		
		quark_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	//unsigned char echobefore[64], echoafter[64];

    uint32_t endiandata[20];
	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);	}
	whirlpool512_setBlock_80((void*)endiandata);
	quark_check_cpu_setTarget(ptarget);
	do {
		int order = 0;
		 
		whirlpool512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		   
		
		whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		whirlpool512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		
		
		uint32_t foundNonce = quark_check_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if  (foundNonce != 0xffffffff)
		{
			
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);

			test_hash(vhash64, endiandata);

			if( (vhash64[7]<=Htarg) && fulltest(vhash64, ptarget) ) {
                
                pdata[19] = foundNonce;
                *hashes_done = foundNonce - first_nonce + 1;
                return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU! vhash64 %08x and htarg %08x", thr_id, foundNonce,vhash64[7],Htarg);
			}
		}
		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
