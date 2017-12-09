/**
 * Penta Blake
 */

#include <stdint.h>
#include <memory.h>
#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

/* hash by cpu with blake 256 */
extern "C" void pentablakehash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[64];

	sph_blake512_context ctx;

	sph_blake512_init(&ctx);
	sph_blake512(&ctx, input, 80);
	sph_blake512_close(&ctx, hash);

	sph_blake512(&ctx, hash, 64);
	sph_blake512_close(&ctx, hash);

	sph_blake512(&ctx, hash, 64);
	sph_blake512_close(&ctx, hash);

	sph_blake512(&ctx, hash, 64);
	sph_blake512_close(&ctx, hash);

	sph_blake512(&ctx, hash, 64);
	sph_blake512_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_free(int thr_id);
extern void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_pentablake(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int rc = 0;
	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 19);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000F;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));

		quark_blake512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);
		CUDA_LOG_ERROR();

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// GPU HASH
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + throughput;

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhash[8];

			be32enc(&endiandata[19], foundNonce);
			pentablakehash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				rc = 1;
				work_set_target_ratio(work, vhash);
				pdata[19] = foundNonce;
				return rc;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	return rc;
}

// cleanup
void free_pentablake(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();

	init[thr_id] = false;
}
