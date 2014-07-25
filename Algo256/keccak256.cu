/*
 * test routine for new algorithm
 * 
 */

extern "C"
{
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_keccak.h"

#include "miner.h"
}

// aus cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint32_t *d_hash[8];

extern void keccak256_cpu_init(int thr_id, int threads);
extern void keccak256_setBlock_80(void *pdata,const void *ptarget);
extern uint32_t keccak256_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);

// fresh Hashfunktion
inline void cpu_hash(void *state, const void *input)
{

    sph_keccak_context ctx_keccak;
    
	
    uint32_t hash[16];

    sph_keccak256_init(&ctx_keccak);
    sph_keccak256 (&ctx_keccak, input, 80);
    sph_keccak256_close(&ctx_keccak, (void*) hash);			
    memcpy(state, hash, 32);
}


extern bool opt_benchmark;

extern "C" int scanhash_keccak256(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];

	const int throughput = 256*256*8*8;
	
	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		keccak256_cpu_init(thr_id, throughput);
		
		init[thr_id] = true;
	}

	

    uint32_t endiandata[20];
	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);	
	}
	keccak256_setBlock_80((void*)endiandata, ptarget);
	do {
		int order = 0;
		
		uint32_t foundNonce = keccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if  (foundNonce != 0xffffffff)
		{
			
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);

			cpu_hash(vhash64, endiandata);

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
