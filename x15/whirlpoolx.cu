/*
 * whirlpool routine (djm)
 * whirlpoolx routine (provos alexis, tpruvot)
 */
extern "C" {
#include "sph/sph_whirlpool.h"
}

#include "miner.h"
#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS] = { 0 };

extern void whirlpoolx_cpu_init(int thr_id, uint32_t threads);
extern void whirlpoolx_cpu_free(int thr_id);
extern void whirlpoolx_setBlock_80(void *pdata, const void *ptarget);
extern uint32_t whirlpoolx_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce);
extern void whirlpoolx_precompute(int thr_id);

// CPU Hash function
extern "C" void whirlxHash(void *state, const void *input)
{
	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[64];
	unsigned char hash_xored[32];

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, input, 80);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	// compress the 48 first bytes of the hash to 32
	for (int i = 0; i < 32; i++) {
		hash_xored[i] = hash[i] ^ hash[i + 16];
	}
	memcpy(state, hash_xored, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_whirlx(int thr_id,  struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	int intensity = is_windows() ? 20 : 22;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		whirlpoolx_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	whirlpoolx_setBlock_80((void*)endiandata, ptarget);
	whirlpoolx_precompute(thr_id);

	do {
		uint32_t foundNonce = whirlpoolx_cpu_hash(thr_id, throughput, pdata[19]);
		if (foundNonce != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			whirlxHash(vhash64, endiandata);

			*hashes_done = pdata[19] - first_nonce + throughput;

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				work_set_target_ratio(work, vhash64);
				pdata[19] = foundNonce;
				return 1;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		pdata[19] += throughput;

		if (((uint64_t)pdata[19]+throughput) >= max_nonce) {
			break;
		}

	} while (!work_restart[thr_id].restart);

	*(hashes_done) = pdata[19] - first_nonce + 1;

	return 0;
}

// cleanup
extern "C" void free_whirlx(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	whirlpoolx_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
