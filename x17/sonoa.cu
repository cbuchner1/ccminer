/**
 * x97 SONO
 **/

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#define NBN 2

static uint32_t *d_hash[MAX_GPUS];

extern void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int flag);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x17_sha512_cpu_init(int thr_id, uint32_t threads);
extern void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x17_haval256_cpu_init(int thr_id, uint32_t threads);
extern void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const int outlen);

// CPU Hash Validation
extern "C" void sonoa_hash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[64];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;
	sph_haval256_5_context ctx_haval;


	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*)hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512_init(&ctx_hamsi);
	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512, (const void*)hash, 64);
	sph_sha512_close(&ctx_sha512, (void*)hash);

	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);


	sph_bmw512(&ctx_bmw, (const void*)hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*)hash);

	sph_groestl512(&ctx_groestl, (const void*)hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*)hash);

	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_jh512(&ctx_jh, (const void*)hash, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	sph_keccak512(&ctx_keccak, (const void*)hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*)hash);

	sph_luffa512(&ctx_luffa, (const void*)hash, 64);
	sph_luffa512_close(&ctx_luffa, (void*)hash);

	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_shavite512(&ctx_shavite, (const void*)hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*)hash);

	sph_simd512(&ctx_simd, (const void*)hash, 64);
	sph_simd512_close(&ctx_simd, (void*)hash);

	sph_echo512(&ctx_echo, (const void*)hash, 64);
	sph_echo512_close(&ctx_echo, (void*)hash);

	sph_hamsi512(&ctx_hamsi, (const void*)hash, 64);
	sph_hamsi512_close(&ctx_hamsi, (void*)hash);

	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_shabal512(&ctx_shabal, (const void*)hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*)hash);

	sph_whirlpool(&ctx_whirlpool, (const void*)hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, (void*)hash);

	sph_sha512(&ctx_sha512, (const void*)hash, 64);
	sph_sha512_close(&ctx_sha512, (void*)hash);

	sph_haval256_5_init(&ctx_haval);
	sph_haval256_5(&ctx_haval, (const void*)hash, 64);
	sph_haval256_5_close(&ctx_haval, (void*)hash);

	memcpy(output, hash, 32);
}

#define x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash) \
  x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash, order++); \
  if (use_compat_kernels[thr_id]) x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash, order++); \
  else x16_echo512_cpu_hash_64(thr_id, throughput, d_hash)


static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_sonoa(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	uint32_t default_throughput = 1 << 18;
	if (device_sm[dev_id] <= 500) default_throughput = 1 << 18;
	else if (device_sm[dev_id] <= 520) default_throughput = 1 << 18;
	else if (device_sm[dev_id]  > 520) default_throughput = (1 << 19) + (1 << 18);

	uint32_t throughput = cuda_default_throughput(thr_id, default_throughput);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	throughput &= 0xFFFFFF00;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO,thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id])
			x11_echo512_cpu_init(thr_id, throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffaCubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x17_sha512_cpu_init(thr_id, throughput);
		x17_haval256_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	int warn = 0;
	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);

		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x16_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffaCubehash512_cpu_hash_64(thr_id, throughput, d_hash[thr_id], order++);
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_simd_echo512_cpu_hash_64(thr_id, throughput, d_hash[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		x17_haval256_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], 256); order++;

		*hashes_done = pdata[19] - first_nonce + throughput;

                work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
                if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			sonoa_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					sonoa_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!warn) {
					warn++;
					pdata[19] = work->nonces[0] + 1;
					continue;
				} else {
					if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
					warn = 0;
				}
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_sonoa(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
