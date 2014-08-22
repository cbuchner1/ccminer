// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#define TPB 256
#define THF 4

// aus cpu-miner.c
extern int device_map[8];

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// diese Struktur wird in der Init Funktion angefordert
static cudaDeviceProp props[8];

// 64 Register Variante für Compute 3.0
#include "groestl_functions_quad.cu"
#include "bitslice_transformations_quad.cu"

__global__ __launch_bounds__(TPB, THF)
void quark_groestl512_gpu_hash_64_quad(int threads, uint32_t startNounce, uint32_t * __restrict g_hash, uint32_t * __restrict g_nonceVector)
{
    // durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t message[8];
        uint32_t state[8];

        uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
        int hashPosition = nounce - startNounce;
        uint32_t *inpHash = &g_hash[hashPosition << 4];

        const uint16_t thr = threadIdx.x % THF;

        #pragma unroll
        for(int k=0;k<4;k++) message[k] = inpHash[(k * THF) + thr];

        #pragma unroll
        for(int k=4;k<8;k++) message[k] = 0;

        if (thr == 0) message[4] = 0x80;
        if (thr == 3) message[7] = 0x01000000;

        uint32_t msgBitsliced[8];
        to_bitslice_quad(message, msgBitsliced);

        groestl512_progressMessage_quad(state, msgBitsliced);

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
}

__global__ void __launch_bounds__(TPB, THF)
 quark_doublegroestl512_gpu_hash_64_quad(int threads, uint32_t startNounce, uint32_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x)>>2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t message[8];
        uint32_t state[8];

        uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
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
}

// Setup-Funktionen
__host__ void quark_groestl512_cpu_init(int thr_id, int threads)
{
    cudaGetDeviceProperties(&props[thr_id], device_map[thr_id]);
}

__host__ void quark_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    int threadsperblock = TPB;

    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const int factor = THF;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_groestl512_gpu_hash_64_quad<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_nonceVector);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void quark_doublegroestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    int threadsperblock = TPB;

    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const int factor = THF;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_doublegroestl512_gpu_hash_64_quad<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_nonceVector);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, order, thr_id);
}
