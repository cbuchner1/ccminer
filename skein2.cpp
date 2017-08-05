/**
 * SKEIN512 80 + SKEIN512 64 (Woodcoin)
 * by tpruvot@github - 2015
 */
#include <string.h>

#include "sph/sph_skein.h"

#include "miner.h"
#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);

extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void skein2hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_skein512_context ctx_skein;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	memcpy(output, (void*) hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_skein2(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput = device_intensity(thr_id, __func__, 1 << 19); // 256*256*8
	throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		cudaMalloc(&d_hash[thr_id], throughput * 64U);

		quark_skein512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	skein512_cpu_setBlock_80((void*)endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 0);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + throughput;

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash64[8];

			endiandata[19] = foundNonce;
			skein2hash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (secNonce != 0) {
					if (!opt_quiet)
						applog(LOG_BLUE, "GPU #%d: found second nonce %08x !", device_map[thr_id], swab32(secNonce));
					pdata[21] = swab32(secNonce);
					res++;
				}
				pdata[19] = swab32(foundNonce);
				return res;
			} else {
				applog(LOG_WARNING, "GPU #%d: result for nonce %08x does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		if ((uint64_t) throughput + pdata[19] > max_nonce) {
			*hashes_done = pdata[19] - first_nonce;
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	return 0;
}
