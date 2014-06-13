#include "uint256.h"
#include "sph/sph_groestl.h"

#include "cpuminer-config.h"
#include "miner.h"

#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>

extern bool opt_benchmark;

void myriadgroestl_cpu_init(int thr_id, int threads);
void myriadgroestl_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void myriadgroestl_cpu_hash(int thr_id, int threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce);

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

static void myriadhash(void *state, const void *input)
{
    sph_groestl512_context     ctx_groestl;

    uint32_t hashA[16], hashB[16];

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, input, 80);
    sph_groestl512_close(&ctx_groestl, hashA);

    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256,(unsigned char *)hashA, 64);
    SHA256_Final((unsigned char *)hashB, &sha256);
    memcpy(state, hashB, 32);
}

extern bool opt_benchmark;

extern "C" int scanhash_myriad(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{	
    if (opt_benchmark)
        ((uint32_t*)ptarget)[7] = 0x000000ff;

	uint32_t start_nonce = pdata[19]++;
	const uint32_t throughPut = 128 * 1024;

	uint32_t *outputHash = (uint32_t*)malloc(throughPut * 16 * sizeof(uint32_t));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];

	// init
	static bool init[8] = { false, false, false, false, false, false, false, false };
	if(!init[thr_id])
	{
#if BIG_DEBUG
#else
		myriadgroestl_cpu_init(thr_id, throughPut);
#endif
		init[thr_id] = true;
	}
	
	uint32_t endiandata[32];
	for (int kk=0; kk < 32; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
	myriadgroestl_cpu_setBlock(thr_id, endiandata, (void*)ptarget);
	
	do {
		// GPU
		uint32_t foundNounce = 0xFFFFFFFF;

		myriadgroestl_cpu_hash(thr_id, throughPut, pdata[19], outputHash, &foundNounce);

		if(foundNounce < 0xffffffff)
		{
			uint32_t tmpHash[8];
			endiandata[19] = SWAP32(foundNounce);
			myriadhash(tmpHash, endiandata);
			if (tmpHash[7] <= Htarg && 
					fulltest(tmpHash, ptarget)) {
						pdata[19] = foundNounce;
						*hashes_done = foundNounce - start_nonce;
						free(outputHash);
				return true;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNounce);
			}

			foundNounce = 0xffffffff;
		}

		if (pdata[19] + throughPut < pdata[19])
			pdata[19] = max_nonce;
		else pdata[19] += throughPut;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = pdata[19] - start_nonce;
	free(outputHash);
	return 0;
}

