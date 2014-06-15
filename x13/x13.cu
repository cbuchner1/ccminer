/*
 * X13 algorithm built on cbuchner1's original X11
 * 
 */

extern "C"
{
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

#include "miner.h"
}

// aus cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen
static uint32_t *d_hash[8];

extern void quark_blake512_cpu_init(int thr_id, int threads);
extern void quark_blake512_cpu_setBlock_80(void *pdata);
extern void quark_blake512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);
extern void quark_blake512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_bmw512_cpu_init(int thr_id, int threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_groestl512_cpu_init(int thr_id, int threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
//extern void quark_doublegroestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, int threads);
extern void quark_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_keccak512_cpu_init(int thr_id, int threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_jh512_cpu_init(int thr_id, int threads);
extern void quark_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, int threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_cubehash512_cpu_init(int thr_id, int threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_shavite512_cpu_init(int thr_id, int threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_simd512_cpu_init(int thr_id, int threads);
extern void x11_simd512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_echo512_cpu_init(int thr_id, int threads);
extern void x11_echo512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_hamsi512_cpu_init(int thr_id, int threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, int threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_check_cpu_init(int thr_id, int threads);
extern void quark_check_cpu_setTarget(const void *ptarget);
extern uint32_t quark_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);

extern void quark_compactTest_cpu_init(int thr_id, int threads);
extern void quark_compactTest_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *inpHashes, 
											uint32_t *d_noncesTrue, size_t *nrmTrue, uint32_t *d_noncesFalse, size_t *nrmFalse,
											int order);

// X13 Hashfunktion
inline void x13hash(void *state, const void *input)
{
    // blake1-bmw2-grs3-skein4-jh5-keccak6-luffa7-cubehash8-shavite9-simd10-echo11-hamsi12-fugue13

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

    uint32_t hash[16];

    sph_blake512_init(&ctx_blake);
    // ZBLAKE;
    sph_blake512 (&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, (void*) hash);

    sph_bmw512_init(&ctx_bmw);
    // ZBMW;
    sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
    sph_bmw512_close(&ctx_bmw, (void*) hash);

    sph_groestl512_init(&ctx_groestl);
    // ZGROESTL;
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_skein512_init(&ctx_skein);
    // ZSKEIN;
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    sph_jh512_init(&ctx_jh);
    // ZJH;
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    sph_keccak512_init(&ctx_keccak);
    // ZKECCAK;
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_luffa512_init(&ctx_luffa);
    // ZLUFFA;
    sph_luffa512 (&ctx_luffa, (const void*) hash, 64);
    sph_luffa512_close (&ctx_luffa, (void*) hash);

    sph_cubehash512_init(&ctx_cubehash);
    // ZCUBEHASH;
    sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
    sph_cubehash512_close(&ctx_cubehash, (void*) hash);

    sph_shavite512_init(&ctx_shavite);
    // ZSHAVITE;
    sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
    sph_shavite512_close(&ctx_shavite, (void*) hash);

    sph_simd512_init(&ctx_simd);
    // ZSIMD
    sph_simd512 (&ctx_simd, (const void*) hash, 64);
    sph_simd512_close(&ctx_simd, (void*) hash);

    sph_echo512_init(&ctx_echo);
    // ZECHO
    sph_echo512 (&ctx_echo, (const void*) hash, 64);
    sph_echo512_close(&ctx_echo, (void*) hash); 

    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512 (&ctx_hamsi, (const void*) hash, 64);
    sph_hamsi512_close(&ctx_hamsi, (void*) hash); 

    sph_fugue512_init(&ctx_fugue);
    sph_fugue512 (&ctx_fugue, (const void*) hash, 64);
    sph_fugue512_close(&ctx_fugue, (void*) hash); 

    memcpy(state, hash, 32);
}


extern bool opt_benchmark;

extern "C" int scanhash_x13(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];

	const int throughput = 256*256*8;

	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		quark_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	//unsigned char echobefore[64], echoafter[64];

    uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	quark_blake512_cpu_setBlock_80((void*)endiandata);
	quark_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

        // erstes Blake512 Hash mit CUDA
		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		// das ist der unbedingte Branch für BMW512
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für Groestl512
		quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für Skein512
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für JH512
		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für Keccak512
		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für Luffa512
		x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für Cubehash512
		x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für Shavite512
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für SIMD512
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// das ist der unbedingte Branch für ECHO512
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

        x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// Scan nach Gewinner Hashes auf der GPU
		uint32_t foundNonce = quark_check_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if  (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			x13hash(vhash64, endiandata);

			if( (vhash64[7]<=Htarg) && fulltest(vhash64, ptarget) ) {
                
                pdata[19] = foundNonce;
                *hashes_done = foundNonce - first_nonce + 1;
                return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
