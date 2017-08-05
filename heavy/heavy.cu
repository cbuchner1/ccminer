#include <stdio.h>
#include <openssl/sha.h>
#include <map>
// include thrust
#include <thrust/remove.h>
#include <thrust/device_vector.h>

#include "miner.h"

extern "C" {
#include "sph/sph_keccak.h"
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
}
#include "hefty1.h"
#include "heavy/heavy.h"
#include "cuda_helper.h"

extern uint32_t *d_hash2output[MAX_GPUS];
extern uint32_t *d_hash3output[MAX_GPUS];
extern uint32_t *d_hash4output[MAX_GPUS];
extern uint32_t *d_hash5output[MAX_GPUS];

#define HEAVYCOIN_BLKHDR_SZ 84
#define MNR_BLKHDR_SZ       80

// nonce-array für die threads
uint32_t *heavy_nonceVector[MAX_GPUS];

extern uint32_t *heavy_heftyHashes[MAX_GPUS];

/* Combines top 64-bits from each hash into a single hash */
static void combine_hashes(uint32_t *out, const uint32_t *hash1, const uint32_t *hash2, const uint32_t *hash3, const uint32_t *hash4)
{
    const uint32_t *hash[4] = { hash1, hash2, hash3, hash4 };
    int bits;
    unsigned int i;
    uint32_t mask;
    unsigned int k;

    /* Transpose first 64 bits of each hash into out */
    memset(out, 0, 32);
    bits = 0;
    for (i = 7; i >= 6; i--) {
        for (mask = 0x80000000; mask; mask >>= 1) {
            for (k = 0; k < 4; k++) {
                out[(255 - bits)/32] <<= 1;
                if ((hash[k][i] & mask) != 0)
                    out[(255 - bits)/32] |= 1;
                bits++;
            }
        }
    }
}

#ifdef _MSC_VER
#include <intrin.h>
static uint32_t __inline bitsset( uint32_t x )
{
    DWORD r = 0;
    _BitScanReverse(&r, x);
    return r;
}
#else
static uint32_t bitsset( uint32_t x )
{
    return 31-__builtin_clz(x);
}
#endif

// Finde das high bit in einem Multiword-Integer.
static int findhighbit(const uint32_t *ptarget, int words)
{
    int i;
    int highbit = 0;
    for (i=words-1; i >= 0; --i)
    {
        if (ptarget[i] != 0) {
            highbit = i*32 + bitsset(ptarget[i])+1;
            break;
        }
    }
    return highbit;
}

// Generiere ein Multiword-Integer das die Zahl
// (2 << highbit) - 1 repräsentiert.
static void genmask(uint32_t *ptarget, int words, int highbit)
{
    int i;
    for (i=words-1; i >= 0; --i)
    {
        if ((i+1)*32 <= highbit)
            ptarget[i] = UINT32_MAX;
        else if (i*32 > highbit)
            ptarget[i] = 0x00000000;
        else
            ptarget[i] = (1 << (highbit-i*32)) - 1;
    }
}

struct check_nonce_for_remove
{
    check_nonce_for_remove(uint64_t target, uint32_t *hashes, uint32_t hashlen, uint32_t startNonce) :
        m_target(target),
        m_hashes(hashes),
        m_hashlen(hashlen),
        m_startNonce(startNonce) { }

    uint64_t  m_target;
    uint32_t *m_hashes;
    uint32_t  m_hashlen;
    uint32_t  m_startNonce;

    __device__
    bool operator()(const uint32_t x)
    {
        // Position im Hash Buffer
        uint32_t hashIndex = x - m_startNonce;
        // Wert des Hashes (als uint64_t) auslesen.
        // Steht im 6. und 7. Wort des Hashes (jeder dieser Hashes hat 512 Bits)
        uint64_t hashValue = *((uint64_t*)(&m_hashes[m_hashlen*hashIndex + 6]));
        bool res = (hashValue & m_target) != hashValue;
        //printf("ndx=%x val=%08x target=%lx\n", hashIndex, hashValue, m_target);
        // gegen das Target prüfen. Es dürfen nur Bits aus dem Target gesetzt sein.
        return res;
    }
};

static bool init[MAX_GPUS] = { 0 };

__host__
int scanhash_heavy(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done, uint32_t maxvote, int blocklen)
{
    const uint32_t first_nonce = pdata[19];
    // CUDA will process thousands of threads.
    uint32_t throughput = device_intensity(thr_id, __func__, (1U << 19) - 256);
    throughput = min(throughput, max_nonce - first_nonce);

    int rc = 0;
    uint32_t *hash = NULL;
    uint32_t *cpu_nonceVector = NULL;
    CUDA_SAFE_CALL(cudaMallocHost(&hash, throughput*8*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMallocHost(&cpu_nonceVector, throughput*sizeof(uint32_t)));

    int nrmCalls[6];
    memset(nrmCalls, 0, sizeof(int) * 6);

    if (opt_benchmark)
        ((uint32_t*)ptarget)[7] = 0x00ff;

    // für jeden Hash ein individuelles Target erstellen basierend
    // auf dem höchsten Bit, das in ptarget gesetzt ist.
    int highbit = findhighbit(ptarget, 8);
    uint32_t target2[2], target3[2], target4[2], target5[2];
    genmask(target2, 2, highbit/4+(((highbit%4)>3)?1:0) ); // SHA256
    genmask(target3, 2, highbit/4+(((highbit%4)>2)?1:0) ); // keccak512
    genmask(target4, 2, highbit/4+(((highbit%4)>1)?1:0) ); // groestl512
    genmask(target5, 2, highbit/4+(((highbit%4)>0)?1:0) ); // blake512

    if (!init[thr_id])
    {
        hefty_cpu_init(thr_id, throughput);
        sha256_cpu_init(thr_id, throughput);
        keccak512_cpu_init(thr_id, throughput);
        groestl512_cpu_init(thr_id, throughput);
        blake512_cpu_init(thr_id, throughput);
        combine_cpu_init(thr_id, throughput);

        CUDA_SAFE_CALL(cudaMalloc(&heavy_nonceVector[thr_id], sizeof(uint32_t) * throughput));

        init[thr_id] = true;
    }

    if (blocklen == HEAVYCOIN_BLKHDR_SZ)
    {
        uint16_t *ext = (uint16_t *)&pdata[20];

        if (opt_vote > maxvote) {
            applog(LOG_WARNING, "Your block reward vote (%hu) exceeds "
                    "the maxvote reported by the pool (%hu).",
                    opt_vote, maxvote);
        }

        if (opt_trust_pool && opt_vote > maxvote) {
            applog(LOG_WARNING, "Capping block reward vote to maxvote reported by pool.");
            ext[0] = maxvote;
        }
        else
            ext[0] = opt_vote;
    }

    // Setze die Blockdaten
    hefty_cpu_setBlock(thr_id, throughput, pdata, blocklen);
    sha256_cpu_setBlock(pdata, blocklen);
    keccak512_cpu_setBlock(pdata, blocklen);
    groestl512_cpu_setBlock(pdata, blocklen);
    blake512_cpu_setBlock(pdata, blocklen);

    do {

        ////// Compaction init
        thrust::device_ptr<uint32_t> devNoncePtr(heavy_nonceVector[thr_id]);
        thrust::device_ptr<uint32_t> devNoncePtrEnd((heavy_nonceVector[thr_id]) + throughput);
        uint32_t actualNumberOfValuesInNonceVectorGPU = throughput;
        uint64_t *t;

        hefty_cpu_hash(thr_id, throughput, pdata[19]);
        //cudaThreadSynchronize();
        sha256_cpu_hash(thr_id, throughput, pdata[19]);
        //cudaThreadSynchronize();

        // Hier ist die längste CPU Wartephase. Deshalb ein strategisches MyStreamSynchronize() hier.
        MyStreamSynchronize(NULL, 1, thr_id);

        ////// Compaction
        t = (uint64_t*) target2;
        devNoncePtrEnd = thrust::remove_if(devNoncePtr, devNoncePtrEnd, check_nonce_for_remove(*t, d_hash2output[thr_id], 8, pdata[19]));
        actualNumberOfValuesInNonceVectorGPU = (uint32_t)(devNoncePtrEnd - devNoncePtr);
        if(actualNumberOfValuesInNonceVectorGPU == 0)
            goto emptyNonceVector;

        keccak512_cpu_hash(thr_id, actualNumberOfValuesInNonceVectorGPU, pdata[19]);
        //cudaThreadSynchronize();

        ////// Compaction
        t = (uint64_t*) target3;
        devNoncePtrEnd = thrust::remove_if(devNoncePtr, devNoncePtrEnd, check_nonce_for_remove(*t, d_hash3output[thr_id], 16, pdata[19]));
        actualNumberOfValuesInNonceVectorGPU = (uint32_t)(devNoncePtrEnd - devNoncePtr);
        if(actualNumberOfValuesInNonceVectorGPU == 0)
            goto emptyNonceVector;

        blake512_cpu_hash(thr_id, actualNumberOfValuesInNonceVectorGPU, pdata[19]);
        //cudaThreadSynchronize();

        ////// Compaction
        t = (uint64_t*) target5;
        devNoncePtrEnd = thrust::remove_if(devNoncePtr, devNoncePtrEnd, check_nonce_for_remove(*t, d_hash5output[thr_id], 16, pdata[19]));
        actualNumberOfValuesInNonceVectorGPU = (uint32_t)(devNoncePtrEnd - devNoncePtr);
        if(actualNumberOfValuesInNonceVectorGPU == 0)
            goto emptyNonceVector;

        groestl512_cpu_hash(thr_id, actualNumberOfValuesInNonceVectorGPU, pdata[19]);
        //cudaThreadSynchronize();

        ////// Compaction
        t = (uint64_t*) target4;
        devNoncePtrEnd = thrust::remove_if(devNoncePtr, devNoncePtrEnd, check_nonce_for_remove(*t, d_hash4output[thr_id], 16, pdata[19]));
        actualNumberOfValuesInNonceVectorGPU = (uint32_t)(devNoncePtrEnd - devNoncePtr);
        if(actualNumberOfValuesInNonceVectorGPU == 0)
            goto emptyNonceVector;

        // combine
        combine_cpu_hash(thr_id, actualNumberOfValuesInNonceVectorGPU, pdata[19], hash);

        if (opt_tracegpu) {
            applog(LOG_BLUE, "heavy GPU hash:");
            applog_hash((uchar*)hash);
        }

        // Ergebnisse kopieren
        if(actualNumberOfValuesInNonceVectorGPU > 0)
        {
            size_t size = sizeof(uint32_t) * actualNumberOfValuesInNonceVectorGPU;
            CUDA_SAFE_CALL(cudaMemcpy(cpu_nonceVector, heavy_nonceVector[thr_id], size, cudaMemcpyDeviceToHost));
            cudaThreadSynchronize();

            for (uint32_t i=0; i < actualNumberOfValuesInNonceVectorGPU; i++)
            {
                uint32_t nonce = cpu_nonceVector[i];
                uint32_t *foundhash = &hash[8*i];
                if (foundhash[7] <= ptarget[7]) {
                    if (fulltest(foundhash, ptarget)) {
                        uint32_t verification[8];
                        pdata[19] += nonce - pdata[19];
                        heavycoin_hash((uchar*)verification, (uchar*)pdata, blocklen);
                        if (memcmp(verification, foundhash, 8*sizeof(uint32_t))) {
                            applog(LOG_ERR, "hash for nonce=$%08X does not validate on CPU!\n", nonce);
                        } else {
                            *hashes_done = pdata[19] - first_nonce;
                            rc = 1;
                            goto exit;
                        }
                    }
                }
            }
        }

emptyNonceVector:

        pdata[19] += throughput;

    } while (pdata[19] < max_nonce && !work_restart[thr_id].restart);
    *hashes_done = pdata[19] - first_nonce;

exit:
    cudaFreeHost(cpu_nonceVector);
    cudaFreeHost(hash);
    return rc;
}

__host__
void heavycoin_hash(uchar* output, const uchar* input, int len)
{
    unsigned char hash1[32];
    unsigned char hash2[32];
    uint32_t hash3[16];
    uint32_t hash4[16];
    uint32_t hash5[16];
    uint32_t *final;
    SHA256_CTX ctx;
    sph_keccak512_context keccakCtx;
    sph_groestl512_context groestlCtx;
    sph_blake512_context blakeCtx;

    HEFTY1(input, len, hash1);

    /* HEFTY1 is new, so take an extra security measure to eliminate
     * the possiblity of collisions:
     *
     *     Hash(x) = SHA256(x + HEFTY1(x))
     *
     * N.B. '+' is concatenation.
     */
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, len);
    SHA256_Update(&ctx, hash1, sizeof(hash1));
    SHA256_Final(hash2, &ctx);

    /* Additional security: Do not rely on a single cryptographic hash
     * function.  Instead, combine the outputs of 4 of the most secure
     * cryptographic hash functions-- SHA256, KECCAK512, GROESTL512
     * and BLAKE512.
     */

    sph_keccak512_init(&keccakCtx);
    sph_keccak512(&keccakCtx, input, len);
    sph_keccak512(&keccakCtx, hash1, sizeof(hash1));
    sph_keccak512_close(&keccakCtx, (void *)&hash3);

    sph_groestl512_init(&groestlCtx);
    sph_groestl512(&groestlCtx, input, len);
    sph_groestl512(&groestlCtx, hash1, sizeof(hash1));
    sph_groestl512_close(&groestlCtx, (void *)&hash4);

    sph_blake512_init(&blakeCtx);
    sph_blake512(&blakeCtx, input, len);
    sph_blake512(&blakeCtx, (unsigned char *)&hash1, sizeof(hash1));
    sph_blake512_close(&blakeCtx, (void *)&hash5);

    final = (uint32_t *)output;
    combine_hashes(final, (uint32_t *)hash2, hash3, hash4, hash5);
}
