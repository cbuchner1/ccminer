// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#endif

#define TPB 256
#define THF 4U

#if __CUDA_ARCH__ >= 300
#include "quark/groestl_functions_quad.h"
#include "quark/groestl_transf_quad.h"
#endif

#include "quark/cuda_quark_groestl512_sm20.cu"

__global__ __launch_bounds__(TPB, THF)
void quark_groestl512_gpu_hash_64_quad(const uint32_t threads, const uint32_t startNounce, uint32_t * g_hash, uint32_t * __restrict g_nonceVector)
{
#if __CUDA_ARCH__ >= 300
    // durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t message[8];
        uint32_t state[8];

        uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
        off_t hashPosition = nounce - startNounce;
        uint32_t *pHash = &g_hash[hashPosition << 4];

        const uint32_t thr = threadIdx.x & 0x3; // % THF

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
        uint32_t __align__(16) hash[16];
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
    uint32_t threadsperblock = TPB;

    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const uint32_t factor = THF;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    int dev_id = device_map[thr_id];

    if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300)
        quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
    else
        quark_groestl512_sm20_hash_64(thr_id, threads, startNounce, d_nonceVector, d_hash, order);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    // MyStreamSynchronize(NULL, order, thr_id);
}
