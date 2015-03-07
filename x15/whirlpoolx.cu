/*
 * whirlpool routine (djm)
 * whirlpoolx routine (provos alexis)
 */
extern "C"
{
#include "sph/sph_whirlpool.h"
#include "miner.h"
}

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void whirlpoolx_cpu_init(int thr_id, int threads);
extern void whirlpoolx_setBlock_80(void *pdata, const void *ptarget);
extern uint32_t cpu_whirlpoolx(int thr_id, uint32_t threads, uint32_t startNounce);
extern void whirlpoolx_precompute();

// CPU Hash function
extern "C" void whirlxHash(void *state, const void *input)
{

	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[64];
	unsigned char hash_xored[32];

	memset(hash, 0, sizeof hash);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, input, 80);
	sph_whirlpool_close(&ctx_whirlpool, hash);


	for (uint32_t i = 0; i < 32; i++){
	        hash_xored[i] = hash[i] ^ hash[i + 16];
	}
	memcpy(state, hash_xored, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_whirlpoolx(int thr_id, uint32_t *pdata,const uint32_t *ptarget, uint32_t max_nonce,unsigned long *hashes_done){
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	uint32_t throughput = pow(2,25);
	throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		whirlpoolx_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
	}

	whirlpoolx_setBlock_80((void*)endiandata, ptarget);
	whirlpoolx_precompute();
	uint64_t n=pdata[19];
	uint32_t foundNonce;
	do {
		if(n+throughput>=max_nonce){
//			applog(LOG_INFO, "GPU #%d: Preventing glitch.", thr_id);
			throughput=max_nonce-n;
		}
		foundNonce = cpu_whirlpoolx(thr_id, throughput, n);
		if (foundNonce != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			whirlxHash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				int res = 1;
//				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				*hashes_done = n - first_nonce + throughput;
/*				if (secNonce != 0) {
					pdata[21] = secNonce;
					res++;
				}*/
				pdata[19] = foundNonce;
				return res;
			}
			else if (vhash64[7] > Htarg) {
				applog(LOG_INFO, "GPU #%d: result for %08x is not in range: %x > %x", thr_id, foundNonce, vhash64[7], Htarg);
			}
			else {
				applog(LOG_INFO, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}
		n += throughput;

	} while (n < max_nonce && !work_restart[thr_id].restart);
	*hashes_done = n - first_nonce;
	return 0;
}
