#include <cuda_runtime.h>
#include "miner.h"
#include "neoscrypt/neoscrypt.h"

extern void neoscrypt_setBlockTarget(uint32_t * data, const void *ptarget);
extern void neoscrypt_cpu_init(int thr_id, uint32_t threads);
extern void neoscrypt_cpu_free(int thr_id);
extern uint32_t neoscrypt_cpu_hash_k4(int thr_id, uint32_t threads, uint32_t startNounce, int have_stratum, int order);

static bool init[MAX_GPUS] = { 0 };

int scanhash_neoscrypt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = is_windows() ? 18 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	throughput = throughput / 32; /* set for max intensity ~= 20 */
	api_set_throughput(thr_id, throughput);

	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce + 1);

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaDeviceSynchronize();
		cudaSetDevice(dev_id);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaGetLastError(); // reset errors if device is not "reset"

		if (device_sm[dev_id] <= 300) {
			applog(LOG_ERR, "Sorry neoscrypt is not supported on SM 3.0 devices");
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}

		applog(LOG_INFO, "GPU #%d: Using %d cuda threads", dev_id, throughput);
		neoscrypt_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	if (have_stratum) {
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], pdata[k]);
	} else {
		for (int k = 0; k < 20; k++)
			endiandata[k] = pdata[k];
	}

	neoscrypt_setBlockTarget(endiandata,ptarget);

	do {
		uint32_t foundNonce = neoscrypt_cpu_hash_k4(thr_id, throughput, pdata[19], have_stratum, 0);
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
				work_set_target_ratio(work, vhash64);
				pdata[19] = foundNonce;
				return 1;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
void free_neoscrypt(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	neoscrypt_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
