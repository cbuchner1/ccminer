#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>

#include "uint256.h"
#include "sph/sph_groestl.h"
#include "cuda_groestlcoin.h"

#include "miner.h"

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

// CPU-groestl
extern "C" void groestlhash(void *state, const void *input)
{
    sph_groestl512_context ctx_groestl;

    //these uint512 in the c++ source of the client are backed by an array of uint32
    uint32_t hashA[16], hashB[16];

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, input, 80); //6
    sph_groestl512_close(&ctx_groestl, hashA); //7

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, hashA, 64); //6
    sph_groestl512_close(&ctx_groestl, hashB); //7

    memcpy(state, hashB, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_groestlcoin(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
    uint32_t max_nonce, unsigned long *hashes_done)
{
    uint32_t start_nonce = pdata[19]++;
    uint32_t throughput = device_intensity(thr_id, __func__, 1 << 19); // 256*256*8
    throughput = min(throughput, max_nonce - start_nonce);

    uint32_t *outputHash = (uint32_t*)malloc(throughput * 16 * sizeof(uint32_t));

    if (opt_benchmark)
        ((uint32_t*)ptarget)[7] = 0x000000ff;

    // init
    if(!init[thr_id])
    {
        groestlcoin_cpu_init(thr_id, throughput);
        init[thr_id] = true;
    }

    // Endian Drehung ist notwendig
    uint32_t endiandata[32];
    for (int kk=0; kk < 32; kk++)
        be32enc(&endiandata[kk], pdata[kk]);

    // Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
    groestlcoin_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

    do {
        // GPU
        uint32_t foundNounce = 0xFFFFFFFF;
        const uint32_t Htarg = ptarget[7];

        groestlcoin_cpu_hash(thr_id, throughput, pdata[19], outputHash, &foundNounce);

        if(foundNounce < 0xffffffff)
        {
            uint32_t tmpHash[8];
            endiandata[19] = SWAP32(foundNounce);
            groestlhash(tmpHash, endiandata);

            if (tmpHash[7] <= Htarg && fulltest(tmpHash, ptarget)) {
                pdata[19] = foundNounce;
                *hashes_done = foundNounce - start_nonce + 1;
                free(outputHash);
                return true;
            } else {
                applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNounce);
            }

            foundNounce = 0xffffffff;
        }

        if (pdata[19] + throughput < pdata[19])
            pdata[19] = max_nonce;
        else pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

    *hashes_done = pdata[19] - start_nonce + 1;
    free(outputHash);
    return 0;
}

