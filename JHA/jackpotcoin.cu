extern "C"
{
#include "sph/sph_keccak.h"
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"

static uint32_t *d_hash[8];

extern void jackpot_keccak512_cpu_init(int thr_id, int threads);
extern void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);
extern void jackpot_keccak512_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void quark_blake512_cpu_init(int thr_id, int threads);
extern void quark_blake512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_groestl512_cpu_init(int thr_id, int threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_jh512_cpu_init(int thr_id, int threads);
extern void quark_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, int threads);
extern void quark_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void jackpot_compactTest_cpu_init(int thr_id, int threads);
extern void jackpot_compactTest_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable, 
											uint32_t *d_nonces1, size_t *nrm1,
											uint32_t *d_nonces2, size_t *nrm2,
											int order);

// Speicher zur Generierung der Noncevektoren für die bedingten Hashes
static uint32_t *d_jackpotNonces[8];
static uint32_t *d_branch1Nonces[8];
static uint32_t *d_branch2Nonces[8];
static uint32_t *d_branch3Nonces[8];

// Original jackpothash Funktion aus einem miner Quelltext
extern "C" unsigned int jackpothash(void *state, const void *input)
{
    sph_blake512_context     ctx_blake;
    sph_groestl512_context   ctx_groestl;
    sph_jh512_context        ctx_jh;
    sph_keccak512_context    ctx_keccak;
    sph_skein512_context     ctx_skein;

    uint32_t hash[16];

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, input, 80);
    sph_keccak512_close(&ctx_keccak, hash);

    unsigned int round;
    for (round = 0; round < 3; round++) {
        if (hash[0] & 0x01) {
           sph_groestl512_init(&ctx_groestl);
           sph_groestl512 (&ctx_groestl, (&hash), 64);
           sph_groestl512_close(&ctx_groestl, (&hash));
        }
        else {
           sph_skein512_init(&ctx_skein);
           sph_skein512 (&ctx_skein, (&hash), 64);
           sph_skein512_close(&ctx_skein, (&hash));
        }
        if (hash[0] & 0x01) {
           sph_blake512_init(&ctx_blake);
           sph_blake512 (&ctx_blake, (&hash), 64);
           sph_blake512_close(&ctx_blake, (&hash));
        }
        else {
           sph_jh512_init(&ctx_jh);
           sph_jh512 (&ctx_jh, (&hash), 64);
           sph_jh512_close(&ctx_jh, (&hash));
        }
    }
    memcpy(state, hash, 32);

    return round;
}

extern "C" int scanhash_jackpot(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x000f;

	int throughput = opt_work_size ? opt_work_size : (1 << 20); // 256*4096
	throughput = min(throughput, (int)(max_nonce - first_nonce));

	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));

		jackpot_keccak512_cpu_init(thr_id, throughput);
		jackpot_compactTest_cpu_init(thr_id, throughput);
		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		cudaMalloc(&d_branch1Nonces[thr_id], sizeof(uint32_t)*throughput*2);
		cudaMalloc(&d_branch2Nonces[thr_id], sizeof(uint32_t)*throughput*2);
		cudaMalloc(&d_branch3Nonces[thr_id], sizeof(uint32_t)*throughput*2);

		CUDA_SAFE_CALL(cudaMalloc(&d_jackpotNonces[thr_id], sizeof(uint32_t)*throughput*2));

		init[thr_id] = true;
	}

	uint32_t endiandata[22];
	for (int k=0; k < 22; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	jackpot_keccak512_cpu_setBlock((void*)endiandata, 80);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// erstes Keccak512 Hash mit CUDA
		jackpot_keccak512_cpu_hash(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		size_t nrm1, nrm2, nrm3;

		// Runde 1 (ohne Gröstl)

		jackpot_compactTest_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], NULL,
				d_branch1Nonces[thr_id], &nrm1,
				d_branch3Nonces[thr_id], &nrm3,
				order++);

		// verfolge den skein-pfad weiter
		quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

		// noch schnell Blake & JH
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// Runde 3 (komplett)

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_groestl512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// Runde 3 (komplett)

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_groestl512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// Scan nach Gewinner Hashes auf der GPU
		uint32_t foundNonce = cuda_check_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
		if  (foundNonce != 0xffffffff)
		{
			unsigned int rounds;
			uint32_t vhash64[8];
			uint32_t Htarg = ptarget[7];
			be32enc(&endiandata[19], foundNonce);

			// diese jackpothash Funktion gibt die Zahl der Runden zurück
			rounds = jackpothash(vhash64, endiandata);

			if ((vhash64[7]<=Htarg) && fulltest(vhash64, ptarget)) {

				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				//applog(LOG_INFO, "GPU #%d: result for nonce $%08X does validate on CPU (%d rounds)!", thr_id, foundNonce, rounds);
				return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU (%d rounds)!", thr_id, foundNonce, rounds);
			}
		}

		if ((uint64_t) pdata[19] + throughput > (uint64_t) max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
