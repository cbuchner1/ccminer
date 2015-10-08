// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include "cuda_helper.h"

#define TPB 256
#define THF 4

#if __CUDA_ARCH__ >= 300
#include "quark/groestl_functions_quad.h"
#include "quark/groestl_transf_quad.h"
#endif

#include "quark/cuda_quark_groestl512_sm20.cu"

__global__ __launch_bounds__(TPB, THF)
void quark_groestl512_gpu_hash_64_quad(uint32_t threads, uint32_t startNounce, uint32_t * __restrict g_hash, uint32_t * __restrict g_nonceVector)
{
#if __CUDA_ARCH__ >= 300
    // durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t message[8];
        uint32_t state[8];

        uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
        off_t hashPosition = nounce - startNounce;
        uint32_t *pHash = &g_hash[hashPosition << 4];

        const uint32_t thr = threadIdx.x % THF;

        #pragma unroll
        for(int k=0;k<4;k++) message[k] = pHash[thr + (k * THF)];

        #pragma unroll
        for(int k=4;k<8;k++) message[k] = 0;

        if (thr == 0) message[4] = 0x80U;
        if (thr == 3) message[7] = 0x01000000U;

        uint32_t msgBitsliced[8];
        to_bitslice_quad(message, msgBitsliced);

        groestl512_progressMessage_quad(state, msgBitsliced);

        // Nur der erste von jeweils 4 Threads bekommt das Ergebns-Hash
        uint32_t hash[16];
        from_bitslice_quad(state, hash);

        // uint4 = 4x4 uint32_t = 16 bytes
        if (thr == 0) {
            uint4 *phash = (uint4*) hash;
            uint4 *outpt = (uint4*) pHash;
            outpt[0] = phash[0];
            outpt[1] = phash[1];
            outpt[2] = phash[2];
            outpt[3] = phash[3];
        }
/*
        if (thr == 0) {
            #pragma unroll
            for(int k=0;k<16;k++) outpHash[k] = hash[k];
        }
*/
    }
#endif
}

__global__ void __launch_bounds__(TPB, THF)
 quark_doublegroestl512_gpu_hash_64_quad(uint32_t threads, uint32_t startNounce, uint32_t *g_hash, uint32_t *g_nonceVector)
{
#if __CUDA_ARCH__ >= 300
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x)>>2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t message[8];
        uint32_t state[8];

        uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);

        off_t hashPosition = nounce - startNounce;
        uint32_t * inpHash = &g_hash[hashPosition<<4];
        const uint16_t thr = threadIdx.x % THF;

        #pragma unroll
        for(int k=0;k<4;k++) message[k] = inpHash[(k * THF) + thr];

        #pragma unroll
        for(int k=4;k<8;k++) message[k] = 0;

        if (thr == 0) message[4] = 0x80;
        if (thr == 3) message[7] = 0x01000000;

        uint32_t msgBitsliced[8];
        to_bitslice_quad(message, msgBitsliced);

        for (int round=0; round<2; round++)
        {
            groestl512_progressMessage_quad(state, msgBitsliced);

            if (round < 1)
            {
                // Verkettung zweier Runden inclusive Padding.
                msgBitsliced[ 0] = __byte_perm(state[ 0], 0x00800100, 0x4341 + (((threadIdx.x%4)==3)<<13));
                msgBitsliced[ 1] = __byte_perm(state[ 1], 0x00800100, 0x4341);
                msgBitsliced[ 2] = __byte_perm(state[ 2], 0x00800100, 0x4341);
                msgBitsliced[ 3] = __byte_perm(state[ 3], 0x00800100, 0x4341);
                msgBitsliced[ 4] = __byte_perm(state[ 4], 0x00800100, 0x4341);
                msgBitsliced[ 5] = __byte_perm(state[ 5], 0x00800100, 0x4341);
                msgBitsliced[ 6] = __byte_perm(state[ 6], 0x00800100, 0x4341);
                msgBitsliced[ 7] = __byte_perm(state[ 7], 0x00800100, 0x4341 + (((threadIdx.x%4)==0)<<4));
            }
        }

        // Nur der erste von jeweils 4 Threads bekommt das Ergebns-Hash
        uint32_t *outpHash = inpHash;
        uint32_t hash[16];
        from_bitslice_quad(state, hash);

        if (thr == 0)
        {
            #pragma unroll
            for(int k=0;k<16;k++) outpHash[k] = hash[k];
        }
    }
#endif
}

__host__
void quark_groestl512_cpu_init(int thr_id, uint32_t threads)
{
    int dev_id = device_map[thr_id];
    cuda_get_arch(thr_id);
    if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
        quark_groestl512_sm20_init(thr_id, threads);
}

__host__
void quark_groestl512_cpu_free(int thr_id)
{
    int dev_id = device_map[thr_id];
    if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
        quark_groestl512_sm20_free(thr_id);
}

__host__
void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    int threadsperblock = TPB;

    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const int factor = THF;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    int dev_id = device_map[thr_id];

    if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300)
        quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
    else
        quark_groestl512_sm20_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__
void quark_doublegroestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int factor = THF;
    int threadsperblock = TPB;

    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    int dev_id = device_map[thr_id];

    if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300)
        quark_doublegroestl512_gpu_hash_64_quad<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
    else
        quark_doublegroestl512_sm20_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);

    MyStreamSynchronize(NULL, order, thr_id);
}
