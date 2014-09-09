/*
 * whirlpool routine (djm)
 */
extern "C"
{
#include "sph/sph_whirlpool.h"
#include "miner.h"
}

// from cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint32_t *d_hash[8];

extern void x15_whirlpool_cpu_init(int thr_id, int threads, int mode);
extern void x15_whirlpool_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void whirlpool512_setBlock_80(void *pdata, const void *ptarget);
extern void whirlpool512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);
extern uint32_t whirlpool512_cpu_finalhash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);


// CPU Hash function
extern "C" void wcoinhash(void *state, const void *input)
{
	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	// shavite 1
	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, input, 80);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hashB);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hashB, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	memcpy(state, hash, 32);
}

extern "C" int scanhash_whc(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	const int throughput = 256*256*8;
	static bool init[8] = {0,0,0,0,0,0,0,0};
	uint32_t endiandata[20];
	uint32_t Htarg = ptarget[7];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = Htarg = 0x0000ff;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 1 /* old whirlpool */);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}

	whirlpool512_setBlock_80((void*)endiandata, ptarget);

	do {
		uint32_t foundNonce;
		int order = 0;

		whirlpool512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		foundNonce = whirlpool512_cpu_finalhash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);

			wcoinhash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
			}
			else if (vhash64[7] > Htarg) {
				applog(LOG_INFO, "GPU #%d: result for %08x is not in range: %x > %x", thr_id, foundNonce, vhash64[7], Htarg);
			}
			else {
				applog(LOG_INFO, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}
		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
