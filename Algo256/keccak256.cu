/*
 * Keccak 256
 *
 */

extern "C"
{
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_keccak.h"

#include "miner.h"
}

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void keccak256_cpu_init(int thr_id, uint32_t threads);
extern void keccak256_setBlock_80(void *pdata,const void *ptarget);
extern uint32_t keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

// CPU Hash
extern "C" void keccak256_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_keccak_context ctx_keccak;

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256 (&ctx_keccak, input, 80);
	sph_keccak256_close(&ctx_keccak, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_keccak256(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = device_intensity(thr_id, __func__, 1U << 21); // 256*256*8*4
	throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0005;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], throughput * 64));
		keccak256_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	keccak256_setBlock_80((void*)endiandata, ptarget);
	do {
		int order = 0;

		*hashes_done = pdata[19] - first_nonce + throughput;

		uint32_t foundNonce = keccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			keccak256_hash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				pdata[19] = foundNonce;
				return 1;
			}
			else {
				applog(LOG_WARNING, "GPU #%d: result for nounce %08x does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		if ((uint64_t) pdata[19] + throughput > max_nonce) {
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	return 0;
}
