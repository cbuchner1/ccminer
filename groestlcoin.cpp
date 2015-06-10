#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#include "sph/sph_groestl.h"
#include "cuda_groestlcoin.h"

#include "miner.h"

// CPU hash
void groestlhash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_groestl512_context ctx_groestl;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, input, 80);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hash, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_groestlcoin(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t start_nonce = pdata[19];
	uint32_t throughput = device_intensity(thr_id, __func__, 1 << 19); // 256*256*8
	throughput = min(throughput, max_nonce - start_nonce);

	uint32_t *outputHash = (uint32_t*)malloc(throughput * 64);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x000000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		groestlcoin_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	groestlcoin_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		uint32_t foundNounce = UINT32_MAX;

		*hashes_done = pdata[19] - start_nonce + throughput;

		// GPU hash
		groestlcoin_cpu_hash(thr_id, throughput, pdata[19], outputHash, &foundNounce);

		if (foundNounce < UINT32_MAX)
		{
			uint32_t _ALIGN(64) tmpHash[8];
			endiandata[19] = swab32(foundNounce);
			groestlhash(tmpHash, endiandata);

			if (tmpHash[7] <= ptarget[7] && fulltest(tmpHash, ptarget)) {
				pdata[19] = foundNounce;
				free(outputHash);
				return true;
			} else {
				applog(LOG_WARNING, "GPU #%d: result for nonce %08x does not validate on CPU!",
					device_map[thr_id], foundNounce);
			}
		}

		if ((uint64_t) pdata[19] + throughput > max_nonce) {
			*hashes_done = pdata[19] - start_nonce + 1;
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	free(outputHash);
	return 0;
}

