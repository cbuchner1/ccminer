/**
 * Fresh algorithm
 */
extern "C" {
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}
#include "miner.h"
#include "cuda_helper.h"

// to test gpu hash on a null buffer
#define NULLTEST 0

static uint32_t *d_hash[MAX_GPUS];

extern void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
extern void x11_shavite512_setBlock_80(void *pdata);
extern void x11_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern int  x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x11_simd512_cpu_free(int thr_id);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);


// CPU Hash
extern "C" void fresh_hash(void *state, const void *input)
{
	// shavite-simd-shavite-simd-echo

	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashA hash
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, input, 80);
	sph_shavite512_close(&ctx_shavite, hashA);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hashA, 64);
	sph_simd512_close(&ctx_simd, hashB);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, hashB, 64);
	sph_shavite512_close(&ctx_shavite, hashA);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, hashA, 64);
	sph_simd512_close(&ctx_simd, hashB);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hashB, 64);
	sph_echo512_close(&ctx_echo, hashA);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_fresh(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << 19);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		CUDA_LOG_ERROR();

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput + 4), -1);

		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	
	x11_shavite512_setBlock_80((void*)endiandata);
	cuda_check_cpu_setTarget(ptarget);
	do {
		uint32_t foundNonce;
		int order = 0;

		// GPU Hash
		x11_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

#if NULLTEST
		uint32_t buf[8]; memset(buf, 0, sizeof buf);
		CUDA_SAFE_CALL(cudaMemcpy(buf, d_hash[thr_id], sizeof buf, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		print_hash((unsigned char*)buf); printf("\n");
#endif
		*hashes_done = pdata[19] - first_nonce + throughput;

		foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			fresh_hash(vhash64, endiandata);

			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash64);
				if (secNonce != 0) {
					be32enc(&endiandata[19], secNonce);
					fresh_hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio)
						work_set_target_ratio(work, vhash64);
					pdata[21] = secNonce;
					res++;
				}
				pdata[19] = foundNonce;
				return res;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}

// cleanup
extern "C" void free_fresh(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaSetDevice(device_map[thr_id]);

	cudaFree(d_hash[thr_id]);
	x11_simd512_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
