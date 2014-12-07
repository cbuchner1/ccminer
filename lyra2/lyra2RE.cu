extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"

static _ALIGN(64) uint64_t *d_hash[8];

extern void blake256_cpu_init(int thr_id, int threads);
extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);
extern void keccak256_cpu_hash_32(int thr_id, int threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void keccak256_cpu_init(int thr_id, int threads);
extern void skein256_cpu_hash_32(int thr_id, int threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void skein256_cpu_init(int thr_id, int threads);

extern void lyra2_cpu_hash_32(int thr_id, int threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void lyra2_cpu_init(int thr_id, int threads);

extern void groestl256_setTarget(const void *ptarget);
extern uint32_t groestl256_cpu_hash_32(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern void groestl256_cpu_init(int thr_id, int threads);

extern "C" void lyra2_hash(void *state, const void *input)
{
	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_skein256_context     ctx_skein;
	sph_groestl256_context   ctx_groestl;

	uint32_t hashA[8], hashB[8];

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}

static bool init[8] = { 0 };

extern "C" int scanhash_lyra2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 19 : 18;
	int throughput = opt_work_size ? opt_work_size : (1 << intensity); // 18=256*256*4;
	throughput = min(throughput, (int)(max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		blake256_cpu_init(thr_id, throughput);
		keccak256_cpu_init(thr_id,throughput);
		skein256_cpu_init(thr_id, throughput);
		groestl256_cpu_init(thr_id, throughput);
		lyra2_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	blake256_cpu_setBlock_80(pdata);
	groestl256_setTarget(ptarget);

	do {
		int order = 0;
		uint32_t foundNonce;

		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		foundNonce = groestl256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if (foundNonce != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			lyra2_hash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = foundNonce;
				return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
