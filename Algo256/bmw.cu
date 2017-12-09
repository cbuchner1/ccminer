/**
 * bmw-256 MDT
 * tpruvot - 2015
 */
extern "C" {
#include "sph/sph_bmw.h"
}

#include <miner.h>
#include <cuda_helper.h>

static uint32_t *d_hash[MAX_GPUS];

extern void bmw256_midstate_init(int thr_id, uint32_t threads);
extern void bmw256_midstate_free(int thr_id);
extern void bmw256_setBlock_80(int thr_id, void *pdata);
extern void bmw256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, int swap);

extern uint32_t cuda_check_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash);

// CPU Hash
extern "C" void bmw_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_bmw256_context ctx;

	sph_bmw256_init(&ctx);
	sph_bmw256(&ctx, input, 80);
	sph_bmw256_close(&ctx, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

static __inline uint32_t swab32_if(uint32_t val, bool iftrue) {
	return iftrue ? swab32(val) : val;
}

extern "C" int scanhash_bmw(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	bool swapnonce = true;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 21);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0005;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);

		cuda_check_cpu_init(thr_id, throughput);
		bmw256_midstate_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}

	cudaGetLastError();
	bmw256_setBlock_80(thr_id, (void*)endiandata);

	cuda_check_cpu_setTarget(ptarget);

	do {
		bmw256_cpu_hash_80(thr_id, (int) throughput, pdata[19], d_hash[thr_id], (int) swapnonce);
		uint32_t foundNonce = cuda_check_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash64[8];
			endiandata[19] = swab32_if(foundNonce, swapnonce);
			bmw_hash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				*hashes_done = foundNonce - first_nonce + 1;
				pdata[19] = swab32_if(foundNonce,!swapnonce);
				work_set_target_ratio(work, vhash64);
				return 1;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_bmw(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	bmw256_midstate_free(thr_id);
	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
