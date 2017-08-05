/*
 * deepcoin algorithm
 *
 */
extern "C" {
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void qubit_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern "C" void deephash(void *state, const void *input)
{
	uint8_t _ALIGN(64) hash[64];

	// luffa-80 cubehash-64 echo-64
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_echo512_context ctx_echo;

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_deep(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput =  device_intensity(thr_id, __func__, 1U << 19); // 256*256*8
	throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000f;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], throughput * 64));

		qubit_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	qubit_luffa512_cpu_setBlock_80((void*)endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			deephash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (secNonce != 0) {
					pdata[21] = secNonce;
					res++;
				}
				pdata[19] = foundNonce;
				return res;
			}
			else {
				applog(LOG_WARNING, "GPU #%d: result for nonce %08x does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
