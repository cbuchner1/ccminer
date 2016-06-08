extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"

#include "cuda_helper.h"
#include "quark/cuda_quark.h"

static uint32_t *d_hash[MAX_GPUS];

// Original nist5hash Funktion aus einem miner Quelltext
extern "C" void nist5hash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    
    uint8_t hash[64];

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, (void*) hash);
    
    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_nist5(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int res = 0;

	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << 20); // 256*256*16
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00FF;

	if (!init[thr_id])
	{
		cudaDeviceSynchronize();
		cudaSetDevice(device_map[thr_id]);

		// Constants copy/init (no device alloc in these algos)
		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);

		// char[64] work space for hashes results
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput));

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

#ifdef USE_STREAMS
	cudaStream_t stream[5];
	for (int i = 0; i < 5; i++)
		cudaStreamCreate(&stream[i]);
#endif

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + throughput;

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			nist5hash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash64);
				if (secNonce != 0) {
					be32enc(&endiandata[19], secNonce);
					nist5hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio)
						work_set_target_ratio(work, vhash64);
					pdata[21] = secNonce;
					res++;
				}
				pdata[19] = foundNonce;
				goto out;
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

out:
//	*hashes_done = pdata[19] - first_nonce;
#ifdef USE_STREAMS
	for (int i = 0; i < 5; i++)
		cudaStreamDestroy(stream[i]);
#endif

	return res;
}

// ressources cleanup
extern "C" void free_nist5(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
