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

static uint32_t *d_hash[8];

extern void keccak256_cpu_init(int thr_id, int threads);
extern void keccak256_setBlock_80(void *pdata,const void *ptarget);
extern uint32_t keccak256_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);

// CPU Hash
extern "C" void keccak256_hash(void *state, const void *input)
{
	sph_keccak_context ctx_keccak;

	uint32_t hash[16];

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256 (&ctx_keccak, input, 80);
	sph_keccak256_close(&ctx_keccak, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[8] = { 0 };

extern "C" int scanhash_keccak256(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = opt_work_size ? opt_work_size : (1 << 21); // 256*256*8*4
	throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0005;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));
		keccak256_cpu_init(thr_id, (int) throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}

	keccak256_setBlock_80((void*)endiandata, ptarget);
	do {
		int order = 0;

		uint32_t foundNonce = keccak256_cpu_hash_80(thr_id, (int) throughput, pdata[19], d_hash[thr_id], order++);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			keccak256_hash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				*hashes_done = foundNonce - first_nonce + 1;
				pdata[19] = foundNonce;
				return 1;
			}
			else {
				applog(LOG_DEBUG, "GPU #%d: result for nounce %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}

		if ((uint64_t) pdata[19] + throughput > max_nonce) {
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
