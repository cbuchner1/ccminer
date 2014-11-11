// Auf Groestlcoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include <host_defines.h>

// globaler Speicher für alle HeftyHashes aller Threads
__constant__ uint32_t pTarget[8]; // Single GPU
extern uint32_t *d_resultNonce[8];

__constant__ uint32_t groestlcoin_gpu_msg[32];

// 64 Register Variante für Compute 3.0
#include "groestl_functions_quad.cu"
#include "bitslice_transformations_quad.cu"

#define SWAB32(x) cuda_swab32(x)

__global__ __launch_bounds__(256, 4)
void groestlcoin_gpu_hash_quad(int threads, uint32_t startNounce, uint32_t *resNounce)
{
    // durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    int thread = (blockDim.x * blockIdx.x + threadIdx.x) / 4;
    if (thread < threads)
    {
        // GROESTL
        uint32_t paddedInput[8];
#pragma unroll 8
        for(int k=0;k<8;k++) paddedInput[k] = groestlcoin_gpu_msg[4*k+threadIdx.x%4];

        uint32_t nounce = startNounce + thread;
        if ((threadIdx.x % 4) == 3)
            paddedInput[4] = SWAB32(nounce);  // 4*4+3 = 19

        uint32_t msgBitsliced[8];
        to_bitslice_quad(paddedInput, msgBitsliced);

        uint32_t state[8];
        for (int round=0; round<2; round++)
        {
            groestl512_progressMessage_quad(state, msgBitsliced);

            if (round < 1)
            {
                // Verkettung zweier Runden inclusive Padding.
                msgBitsliced[ 0] = __byte_perm(state[ 0], 0x00800100, 0x4341 + ((threadIdx.x%4)==3)*0x2000);
                msgBitsliced[ 1] = __byte_perm(state[ 1], 0x00800100, 0x4341);
                msgBitsliced[ 2] = __byte_perm(state[ 2], 0x00800100, 0x4341);
                msgBitsliced[ 3] = __byte_perm(state[ 3], 0x00800100, 0x4341);
                msgBitsliced[ 4] = __byte_perm(state[ 4], 0x00800100, 0x4341);
                msgBitsliced[ 5] = __byte_perm(state[ 5], 0x00800100, 0x4341);
                msgBitsliced[ 6] = __byte_perm(state[ 6], 0x00800100, 0x4341);
                msgBitsliced[ 7] = __byte_perm(state[ 7], 0x00800100, 0x4341 + ((threadIdx.x%4)==0)*0x0010);
            }
        }

        // Nur der erste von jeweils 4 Threads bekommt das Ergebns-Hash
        uint32_t out_state[16];
        from_bitslice_quad(state, out_state);
        
        if (threadIdx.x % 4 == 0)
        {
            int i, position = -1;
            bool rc = true;

    #pragma unroll 8
            for (i = 7; i >= 0; i--) {
                if (out_state[i] > pTarget[i]) {
                    if(position < i) {
                        position = i;
                        rc = false;
                    }
                 }
                 if (out_state[i] < pTarget[i]) {
                    if(position < i) {
                        position = i;
                        rc = true;
                    }
                 }
            }

            if(rc == true)
                if(resNounce[0] > nounce)
                    resNounce[0] = nounce;
        }
    }
}

// Setup-Funktionen
__host__ void groestlcoin_cpu_init(int thr_id, int threads)
{
    cudaSetDevice(device_map[thr_id]);

    // Speicher für Gewinner-Nonce belegen
    cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)); 
}

__host__ void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
    // Nachricht expandieren und setzen
    uint32_t msgBlock[32];

    memset(msgBlock, 0, sizeof(uint32_t) * 32);
    memcpy(&msgBlock[0], data, 80);

    // Erweitere die Nachricht auf den Nachrichtenblock (padding)
    // Unsere Nachricht hat 80 Byte
    msgBlock[20] = 0x80;
    msgBlock[31] = 0x01000000;

    // groestl512 braucht hierfür keinen CPU-Code (die einzige Runde wird
    // auf der GPU ausgeführt)

    // Blockheader setzen (korrekte Nonce und Hefty Hash fehlen da drin noch)
    cudaMemcpyToSymbol( groestlcoin_gpu_msg,
                        msgBlock,
                        128);

    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    cudaMemcpyToSymbol( pTarget,
                        pTargetIn,
                        sizeof(uint32_t) * 8 );
}

__host__ void groestlcoin_cpu_hash(int thr_id, int threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce)
{
    int threadsperblock = 256;

    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    int factor = 4;

        // berechne wie viele Thread Blocks wir brauchen
    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    groestlcoin_gpu_hash_quad<<<grid, block, shared_size>>>(threads, startNounce, d_resultNonce[thr_id]);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, 0, thr_id);

    cudaMemcpy(nounce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
