extern "C" {
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern "C" void skein2hash(void *output, const void *input)
{
	sph_skein512_context ctx_skein;

	uint32_t hash[16];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);
	//applog_hash((uchar*)hash);
	//applog_hash((uchar*)&hash[8]);
	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	memcpy(output, hash, 32);
}

#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 16*sizeof(uint32_t)); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 16*sizeof(uint32_t), cudaMemcpyDeviceToHost); \
		printf("SK2 %s %08x %08x %08x %08x...\n", algo, \
			swab32(debugbuf[0]), swab32(debugbuf[1]), swab32(debugbuf[2]), swab32(debugbuf[3])); \
		cudaFree(debugbuf); \
		} \
}
#else
#define TRACE(algo) {}
#endif

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_skein2(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput =  device_intensity(thr_id, __func__, 1 << 19); // 256*256*8
	throughput = min(throughput,  (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0FFF;

	if (!init[thr_id])
	{
		cudaDeviceReset();
		cudaSetDevice(device_map[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 64 * throughput));

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	skein512_cpu_setBlock_80((void*)endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Hash with CUDA
		skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		TRACE("80:");
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("64:");

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhash64[8];

			endiandata[19] = foundNonce;
			skein2hash(vhash64, endiandata);

			#define Htarg ptarget[7]
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (secNonce != 0) {
					if (!opt_quiet)
						applog(LOG_BLUE, "GPU #%d: found second nonce %08x !", device_map[thr_id], swab32(secNonce));
					pdata[21] = swab32(secNonce);
					res++;
				}
				pdata[19] = swab32(foundNonce);
				return res;
			}
			else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
