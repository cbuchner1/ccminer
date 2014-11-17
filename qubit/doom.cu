/*
 * qubit algorithm
 *
 */
extern "C" {
#include "sph/sph_luffa.h"
}

#include "miner.h"

#include "cuda_helper.h"

static uint32_t *d_hash[8];

extern void qubit_luffa512_cpu_init(int thr_id, int threads);
extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);
extern void qubit_luffa512_cpufinal_setBlock_80(void *pdata, const void *ptarget);
extern uint32_t qubit_luffa512_cpu_finalhash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void doomhash(void *state, const void *input)
{
	// luffa512
	sph_luffa512_context ctx_luffa;

	uint8_t hash[64];

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	memcpy(state, hash, 32);
}


extern "C" int scanhash_doom(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	static bool init[8] = {0,0,0,0,0,0,0,0};
	uint32_t endiandata[20];
	int throughput = opt_work_size ? opt_work_size : (1 << 22); // 256*256*8*8
	throughput = min(throughput, (int)(max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000f;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));

		qubit_luffa512_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	qubit_luffa512_cpufinal_setBlock_80((void*)endiandata,ptarget);

	do {
		const uint32_t Htarg = ptarget[7];
		int order = 0;

		uint32_t foundNonce = qubit_luffa512_cpu_finalhash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			doomhash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget) )
			{
				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
			}
		}

		pdata[19] += throughput;

		if ((uint64_t) pdata[19] + throughput > max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
