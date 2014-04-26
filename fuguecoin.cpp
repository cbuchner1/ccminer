#include "uint256.h"
#include "sph/sph_fugue.h"

#include "cpuminer-config.h"
#include "miner.h"

#include <string.h>
#include <stdint.h>
#include <cuda_fugue256.h>

extern "C" void my_fugue256_init(void *cc);
extern "C" void my_fugue256(void *cc, const void *data, size_t len);
extern "C" void my_fugue256_close(void *cc, void *dst);
extern "C" void my_fugue256_addbits_and_close(void *cc, unsigned ub, unsigned n, void *dst);

// vorbereitete Kontexte nach den ersten 80 Bytes
sph_fugue256_context  ctx_fugue_const[8];

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

extern "C" int scanhash_fugue256(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{	
	uint32_t start_nonce = pdata[19]++;
	const uint32_t Htarg = ptarget[7];
	const uint32_t throughPut = 4096 * 128;

	// init
	static bool init[8] = { false, false, false, false, false, false, false, false };
	if(!init[thr_id])
	{
		fugue256_cpu_init(thr_id, throughPut);
		init[thr_id] = true;
	}
	
	// Endian Drehung ist notwendig
	uint32_t endiandata[20];
	for (int kk=0; kk < 20; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
	fugue256_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		// GPU
		uint32_t foundNounce = 0xFFFFFFFF;
		fugue256_cpu_hash(thr_id, throughPut, pdata[19], NULL, &foundNounce);

		if(foundNounce < 0xffffffff)
		{
			uint32_t hash[8];
			endiandata[19] = SWAP32(foundNounce);
			sph_fugue256_context ctx_fugue;
			sph_fugue256_init(&ctx_fugue);
			sph_fugue256 (&ctx_fugue, endiandata, 80);
			sph_fugue256_close(&ctx_fugue, &hash);

			if (hash[7] <= Htarg && fulltest(hash, ptarget))
			{
				pdata[19] = foundNounce;
				*hashes_done = foundNounce - start_nonce;
				return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNounce);
			}
		}

		if (pdata[19] + throughPut < pdata[19])
			pdata[19] = max_nonce;
		else pdata[19] += throughPut;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = pdata[19] - start_nonce;
	return 0;
}

void fugue256_hash(unsigned char* output, const unsigned char* input, int len)
{
	sph_fugue256_context ctx;
	sph_fugue256_init(&ctx);
    sph_fugue256(&ctx, input, len);    
    sph_fugue256_close(&ctx, (void *)output);
}
