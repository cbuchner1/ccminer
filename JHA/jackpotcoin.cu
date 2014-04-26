
extern "C"
{
#include "sph/sph_keccak.h"
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_skein.h"
}

#include "miner.h"
#include <stdint.h>

// aus cpu-miner.c
extern int device_map[8];
extern bool opt_benchmark;

// Speicher für Input/Output der verketteten Hashfunktionen
static uint32_t *d_hash[8];

extern void jackpot_keccak512_cpu_init(int thr_id, int threads);
extern void jackpot_keccak512_cpu_setBlock_88(void *pdata);
extern void jackpot_keccak512_cpu_hash_88(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void quark_check_cpu_init(int thr_id, int threads);
extern void quark_check_cpu_setTarget(const void *ptarget);
extern uint32_t quark_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);

// Original jackpothash Funktion aus einem miner Quelltext
inline unsigned int jackpothash(void *state, const void *input)
{
    sph_blake512_context     ctx_blake;
    sph_groestl512_context   ctx_groestl;
    sph_jh512_context        ctx_jh;
    sph_keccak512_context    ctx_keccak;
    sph_skein512_context     ctx_skein;

    uint32_t hash[16];

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, input, 88);
    sph_keccak512_close(&ctx_keccak, hash);

    unsigned int round_mask = (
       (unsigned int)(((unsigned char *)input)[84]) <<  0 |
       (unsigned int)(((unsigned char *)input)[85]) <<  8 |
       (unsigned int)(((unsigned char *)input)[86]) << 16 |
       (unsigned int)(((unsigned char *)input)[87]) << 24 );
    unsigned int round_max  = hash[0] & round_mask;
    unsigned int round;
    for (round = 0; round < round_max; round++) {
        switch (hash[0] & 3) {
          case 0:
               sph_blake512_init(&ctx_blake);
               sph_blake512 (&ctx_blake, hash, 64);
               sph_blake512_close(&ctx_blake, hash);
               break;
          case 1:
               sph_groestl512_init(&ctx_groestl);
               sph_groestl512 (&ctx_groestl, hash, 64);
               sph_groestl512_close(&ctx_groestl, hash);
               break;
          case 2:
               sph_jh512_init(&ctx_jh);
               sph_jh512 (&ctx_jh, hash, 64);
               sph_jh512_close(&ctx_jh, hash);
               break;
          case 3:
               sph_skein512_init(&ctx_skein);
               sph_skein512 (&ctx_skein, hash, 64);
               sph_skein512_close(&ctx_skein, hash);
               break;
        }
    }
    memcpy(state, hash, 32);

    return round_max;
}


static int bit_population(uint32_t n){
  int c =0;
  while(n){
    c += n&1;
    n = n>>1;
  }
  return c;
}

extern "C" int scanhash_jackpot(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	// TODO: entfernen für eine Release! Ist nur zum Testen!
	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x00000f;
		((uint32_t*)pdata)[21] = 0x07000000;  // round_mask von 7 vorgeben
    }

	const uint32_t Htarg = ptarget[7];

	const int throughput = 256*4096; // 100;

	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);
		jackpot_keccak512_cpu_init(thr_id, throughput);
		quark_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[22];
	for (int k=0; k < 22; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	unsigned int round_mask = (
		(unsigned int)(((unsigned char *)endiandata)[84]) <<  0 |
		(unsigned int)(((unsigned char *)endiandata)[85]) <<  8 |
		(unsigned int)(((unsigned char *)endiandata)[86]) << 16 |
		(unsigned int)(((unsigned char *)endiandata)[87]) << 24 );

	// Zählen wie viele Bits in round_mask gesetzt sind
	int bitcount = bit_population(round_mask);

	jackpot_keccak512_cpu_setBlock_88((void*)endiandata);
	quark_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// erstes Blake512 Hash mit CUDA
		jackpot_keccak512_cpu_hash_88(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		// TODO: hier fehlen jetzt natürlich noch die anderen Hashrunden.
		// bei round_mask=7 haben wir eine 1:8 Chance, dass das Hash dennoch
		// die Kriterien erfüllt wenn hash[0] & round_mask  zufällig 0 ist.

		// Scan nach Gewinner Hashes auf der GPU
		uint32_t foundNonce = quark_check_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if  (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);

			// diese jackpothash Funktion gibt die Zahl der zusätzlichen Runden zurück
			unsigned int rounds = jackpothash(vhash64, endiandata);

			// wir akzeptieren nur solche Hashes wo ausschliesslich Keccak verwendet wurde
			if (rounds == 0) {
				if ((vhash64[7]<=Htarg) && fulltest(vhash64, ptarget)) {

					pdata[19] = foundNonce;
					*hashes_done = (foundNonce - first_nonce + 1) / (1 << bitcount);
					return 1;
				} else {
					applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU (%d rounds)!", thr_id, foundNonce, rounds);
				}
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = (pdata[19] - first_nonce + 1) / (1 << bitcount);
	return 0;
}
