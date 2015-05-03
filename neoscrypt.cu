
extern "C" {
#include "neoscrypt/neoscrypt.h"
}

#include "cuda_helper.h"
#include "miner.h"

static uint32_t *d_hash[MAX_GPUS] ;
extern void neoscrypt_setBlockTarget(uint32_t * data, const void *ptarget);
extern void neoscrypt_cpu_init(int thr_id, uint32_t threads, uint32_t* hash);
extern uint32_t neoscrypt_cpu_hash_k4(int stratum, int thr_id, uint32_t threads, uint32_t startNounce, int order);

#define SHIFT 130

int scanhash_neoscrypt(int thr_id, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	int intensity = is_windows() ? 18 : 19;
	uint32_t throughput = device_intensity(thr_id, __func__, 1U << intensity);
	throughput = throughput / 32; /* set for max intensity ~= 20 */
	throughput = min(throughput, max_nonce - first_nonce + 1);

	static bool init[MAX_GPUS] = { 0 };
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 32 * SHIFT * sizeof(uint64_t) * throughput));
		neoscrypt_cpu_init(thr_id, throughput, d_hash[thr_id]);

		applog(LOG_INFO, "Using %d cuda threads", throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	if (have_stratum) {
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	} else {
		for (int k = 0; k < 20; k++)
			endiandata[k] = pdata[k];
	}

	neoscrypt_setBlockTarget(endiandata,ptarget);

	do {
		uint32_t foundNonce = neoscrypt_cpu_hash_k4((int)have_stratum, thr_id, throughput, pdata[19], 0);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash64[8];

			*hashes_done = pdata[19] - first_nonce + 1;

			if (have_stratum) {
				be32enc(&endiandata[19], foundNonce);
			} else {
				endiandata[19] = foundNonce;
			}
			neoscrypt((uchar*)vhash64, (uchar*) endiandata, 0x80000620U);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				pdata[19] = foundNonce;
				return 1;
			} else {
				applog(LOG_WARNING, "GPU #%d: result for nonce %08x does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (max_nonce > ((uint64_t)(pdata[19]) + throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
