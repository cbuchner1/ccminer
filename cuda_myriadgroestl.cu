// Auf Myriadcoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// globaler Speicher für alle HeftyHashes aller Threads
__constant__ uint32_t pTarget[8]; // Single GPU
uint32_t *d_outputHashes[8];
extern uint32_t *d_resultNonce[8];

__constant__ uint32_t myriadgroestl_gpu_msg[32];

// muss expandiert werden
__constant__ uint32_t myr_sha256_gpu_constantTable[64];
__constant__ uint32_t myr_sha256_gpu_constantTable2[64];
__constant__ uint32_t myr_sha256_gpu_hashTable[8];

uint32_t myr_sha256_cpu_hashTable[] = { 
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
uint32_t myr_sha256_cpu_constantTable[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

uint32_t myr_sha256_cpu_w2Table[] = {
    0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
    0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
    0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
    0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
    0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
    0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
    0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484 };

// 64 Register Variante für Compute 3.0
#include "groestl_functions_quad.cu"
#include "bitslice_transformations_quad.cu"

#define SWAB32(x)        ( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
    // Kepler (Compute 3.5)
    #define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif
#define R(x, n)            ((x) >> (n))
#define Ch(x, y, z)        ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define S0(x)            (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)            (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)            (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)            (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))

__device__ void myriadgroestl_gpu_sha256(uint32_t *message)
{
    uint32_t W1[16];
    uint32_t W2[16];

    // Initialisiere die register a bis h mit der Hash-Tabelle
    uint32_t regs[8];
    uint32_t hash[8];

    // pre
#pragma unroll 8
    for (int k=0; k < 8; k++)
    {
        regs[k] = myr_sha256_gpu_hashTable[k];
        hash[k] = regs[k];
    }
    
#pragma unroll 16
    for(int k=0;k<16;k++)
        W1[k] = SWAB32(message[k]);

// Progress W1
#pragma unroll 16
    for(int j=0;j<16;j++)
    {
        uint32_t T1, T2;
        T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j] + W1[j];
        T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
        
        #pragma unroll 7
        for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
        regs[0] = T1 + T2;
        regs[4] += T1;
    }

// Progress W2...W3
////// PART 1
#pragma unroll 2
    for(int j=0;j<2;j++)
        W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];
#pragma unroll 5
    for(int j=2;j<7;j++)
        W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

#pragma unroll 8
    for(int j=7;j<15;j++)
        W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

    W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

    // Rundenfunktion
#pragma unroll 16
    for(int j=0;j<16;j++)
    {
        uint32_t T1, T2;
        T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j + 16] + W2[j];
        T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
        
        #pragma unroll 7
        for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
        regs[0] = T1 + T2;
        regs[4] += T1;
    }

////// PART 2
#pragma unroll 2
    for(int j=0;j<2;j++)
        W1[j] = s1(W2[14+j]) + W2[9+j] + s0(W2[1+j]) + W2[j];
#pragma unroll 5
    for(int j=2;j<7;j++)
        W1[j] = s1(W1[j-2]) + W2[9+j] + s0(W2[1+j]) + W2[j];

#pragma unroll 8
    for(int j=7;j<15;j++)
        W1[j] = s1(W1[j-2]) + W1[j-7] + s0(W2[1+j]) + W2[j];

    W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

    // Rundenfunktion
#pragma unroll 16
    for(int j=0;j<16;j++)
    {
        uint32_t T1, T2;
        T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j + 32] + W1[j];
        T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
        
        #pragma unroll 7
        for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
        regs[0] = T1 + T2;
        regs[4] += T1;
    }

////// PART 3
#pragma unroll 2
    for(int j=0;j<2;j++)
        W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];
#pragma unroll 5
    for(int j=2;j<7;j++)
        W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

#pragma unroll 8
    for(int j=7;j<15;j++)
        W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

    W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

    // Rundenfunktion
#pragma unroll 16
    for(int j=0;j<16;j++)
    {
        uint32_t T1, T2;
        T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j + 48] + W2[j];
        T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
        
        #pragma unroll 7
        for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
        regs[0] = T1 + T2;
        regs[4] += T1;
    }

#pragma unroll 8
    for(int k=0;k<8;k++)
        hash[k] += regs[k];

    /////
    ///// Zweite Runde (wegen Msg-Padding)
    /////
#pragma unroll 8
    for(int k=0;k<8;k++)
        regs[k] = hash[k];

// Progress W1
#pragma unroll 64
    for(int j=0;j<64;j++)
    {
        uint32_t T1, T2;
        T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable2[j];
        T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
        
        #pragma unroll 7
        for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
        regs[0] = T1 + T2;
        regs[4] += T1;
    }

#pragma unroll 8
    for(int k=0;k<8;k++)
        hash[k] += regs[k];

    //// FERTIG

#pragma unroll 8
    for(int k=0;k<8;k++)
        message[k] = SWAB32(hash[k]);
}

__global__ void __launch_bounds__(256, 4)
 myriadgroestl_gpu_hash_quad(int threads, uint32_t startNounce, uint32_t *hashBuffer)
{
    // durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    int thread = (blockDim.x * blockIdx.x + threadIdx.x) / 4;
    if (thread < threads)
    {
        // GROESTL
        uint32_t paddedInput[8];
#pragma unroll 8
        for(int k=0;k<8;k++) paddedInput[k] = myriadgroestl_gpu_msg[4*k+threadIdx.x%4];

        uint32_t nounce = startNounce + thread;
        if ((threadIdx.x % 4) == 3)
            paddedInput[4] = SWAB32(nounce);  // 4*4+3 = 19

        uint32_t msgBitsliced[8];
        to_bitslice_quad(paddedInput, msgBitsliced);

        uint32_t state[8];

        groestl512_progressMessage_quad(state, msgBitsliced);

        uint32_t out_state[16];
        from_bitslice_quad(state, out_state);

        if ((threadIdx.x & 0x03) == 0)
        {
            uint32_t *outpHash = &hashBuffer[16 * thread];
#pragma unroll 16
            for(int k=0;k<16;k++) outpHash[k] = out_state[k];
        }
    }
}

__global__ void
 myriadgroestl_gpu_hash_quad2(int threads, uint32_t startNounce, uint32_t *resNounce, uint32_t *hashBuffer)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = startNounce + thread;

        uint32_t out_state[16];
        uint32_t *inpHash = &hashBuffer[16 * thread];
#pragma unroll 16
        for (int i=0; i < 16; i++)
            out_state[i] = inpHash[i];

        myriadgroestl_gpu_sha256(out_state);
        
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

// Setup-Funktionen
__host__ void myriadgroestl_cpu_init(int thr_id, int threads)
{
    cudaSetDevice(device_map[thr_id]);
    
    cudaMemcpyToSymbol( myr_sha256_gpu_hashTable,
                        myr_sha256_cpu_hashTable,
                        sizeof(uint32_t) * 8 );

    cudaMemcpyToSymbol( myr_sha256_gpu_constantTable,
                        myr_sha256_cpu_constantTable,
                        sizeof(uint32_t) * 64 );

    // zweite CPU-Tabelle bauen und auf die GPU laden
    uint32_t temp[64];
    for(int i=0;i<64;i++)
        temp[i] = myr_sha256_cpu_w2Table[i] + myr_sha256_cpu_constantTable[i];

    cudaMemcpyToSymbol( myr_sha256_gpu_constantTable2,
                        temp,
                        sizeof(uint32_t) * 64 );

    // Speicher für Gewinner-Nonce belegen
    cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)); 

    // Speicher für temporäreHashes
    cudaMalloc(&d_outputHashes[thr_id], 16*sizeof(uint32_t)*threads); 
}

__host__ void myriadgroestl_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
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
    cudaMemcpyToSymbol( myriadgroestl_gpu_msg,
                        msgBlock,
                        128);

    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    cudaMemcpyToSymbol( pTarget,
                        pTargetIn,
                        sizeof(uint32_t) * 8 );
}

__host__ void myriadgroestl_cpu_hash(int thr_id, int threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce)
{
    int threadsperblock = 256;

    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const int factor=4;

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
    dim3 block(threadsperblock);

    myriadgroestl_gpu_hash_quad<<<grid, block, shared_size>>>(threads, startNounce, d_outputHashes[thr_id]);
    dim3 grid2((threads + threadsperblock-1)/threadsperblock);
    myriadgroestl_gpu_hash_quad2<<<grid2, block, shared_size>>>(threads, startNounce, d_resultNonce[thr_id], d_outputHashes[thr_id]);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, 0, thr_id);

    cudaMemcpy(nounce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
