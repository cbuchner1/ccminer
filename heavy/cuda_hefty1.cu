#include <stdio.h>
#include <memory.h>

#include "miner.h"

#include "cuda_helper.h"

#define USE_SHARED 1

// globaler Speicher für alle HeftyHashes aller Threads
uint32_t *heavy_heftyHashes[MAX_GPUS];

/* Hash-Tabellen */
__constant__ uint32_t hefty_gpu_constantTable[64];
#if USE_SHARED
#define heftyLookUp(x) (*((uint32_t*)heftytab + (x)))
#else
#define heftyLookUp(x) hefty_gpu_constantTable[x]
#endif

// muss expandiert werden
__constant__ uint32_t hefty_gpu_blockHeader[16]; // 2x512 Bit Message
__constant__ uint32_t hefty_gpu_register[8];
__constant__ uint32_t hefty_gpu_sponge[4];

uint32_t hefty_cpu_hashTable[] = {
    0x6a09e667UL,
    0xbb67ae85UL,
    0x3c6ef372UL,
    0xa54ff53aUL,
    0x510e527fUL,
    0x9b05688cUL,
    0x1f83d9abUL,
    0x5be0cd19UL
};

uint32_t hefty_cpu_constantTable[] = {
    0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL,
    0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
    0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL,
    0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
    0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
    0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
    0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL,
    0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
    0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL,
    0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
    0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL,
    0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
    0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL,
    0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
    0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
    0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
};

#if 0
#define S(x, n)          (((x) >> (n)) | ((x) << (32 - (n))))
#else
__host__ __device__
static uint32_t S(uint32_t x, int n)
{
    return (((x) >> (n)) | ((x) << (32 - (n))));
}
#endif

#define R(x, n)          ((x) >> (n))
#define Ch(x, y, z)      ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)     ((x & (y | z)) | (y & z))
#define S0(x)            (S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define S1(x)            (S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define s0(x)            (S(x, 7) ^ S(x, 18) ^ R(x, 3))
#define s1(x)            (S(x, 17) ^ S(x, 19) ^ R(x, 10))

#define SWAB32(x)        ( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

// uint8_t
#define smoosh4(x)       ( ((x)>>4) ^ ((x) & 0x0F) )

__host__ __forceinline__ __device__
uint8_t smoosh2(uint32_t x)
{
    uint16_t w = (x >> 16) ^ (x & 0xffff);
    uint8_t n = smoosh4( (uint8_t)( (w >> 8) ^ (w & 0xFF) ) );
    return 24 - (((n >> 2) ^ (n & 0x03)) << 3);
}
// 4 auf einmal
#define smoosh4Quad(x)   ( (((x)>>4) ^ (x)) & 0x0F0F0F0F )
#define getByte(x,y)     ( ((x) >> (y)) & 0xFF )

__host__ __forceinline__ __device__
void Mangle(uint32_t *inp)
{
    uint32_t r = smoosh4Quad(inp[0]);
    uint32_t inp0org;
    uint32_t tmp0Mask, tmp1Mask;
    uint32_t in1, in2, isAddition;
    int32_t tmp;
    uint8_t b;

    inp[1] = inp[1] ^ S(inp[0], getByte(r, 24));

    r += 0x01010101;
    tmp = smoosh2(inp[1]);
    b = getByte(r,tmp);
    inp0org = S(inp[0], b);
    tmp0Mask = (uint32_t) -((tmp >> 3) & 1); // Bit 3 an Position 0
    tmp1Mask = (uint32_t) -((tmp >> 4) & 1); // Bit 4 an Position 0

    in1 =    (inp[2] & ~inp0org) |
            (tmp1Mask & ~inp[2] & inp0org) |
            (~tmp0Mask & ~inp[2] & inp0org);
    in2 = inp[2] += ~inp0org;
    isAddition = ~tmp0Mask & tmp1Mask;
    inp[2] = isAddition ? in2 : in1;

    r += 0x01010101;
    tmp = smoosh2(inp[1] ^ inp[2]);
    b = getByte(r,tmp);
    inp0org = S(inp[0], b);
    tmp0Mask = (uint32_t) -((tmp >> 3) & 1); // Bit 3 an Position 0
    tmp1Mask = (uint32_t) -((tmp >> 4) & 1); // Bit 4 an Position 0

    in1 =    (inp[3] & ~inp0org) |
            (tmp1Mask & ~inp[3] & inp0org) |
            (~tmp0Mask & ~inp[3] & inp0org);
    in2 = inp[3] += ~inp0org;
    isAddition = ~tmp0Mask & tmp1Mask;
    inp[3] = isAddition ? in2 : in1;

    inp[0] ^= (inp[1] ^ inp[2]) + inp[3];
}

__host__ __forceinline__ __device__
void Absorb(uint32_t *inp, uint32_t x)
{
    inp[0] ^= x;
    Mangle(inp);
}

__host__ __forceinline__ __device__
uint32_t Squeeze(uint32_t *inp)
{
    uint32_t y = inp[0];
    Mangle(inp);
    return y;
}

__host__ __forceinline__ __device__
uint32_t Br(uint32_t *sponge, uint32_t x)
{
    uint32_t r = Squeeze(sponge);
    uint32_t t = ((r >> 8) & 0x1F);
    uint32_t y = 1 << t;

    uint32_t a = (((r>>1) & 0x01) << t) & y;
    uint32_t b = ((r & 0x01) << t) & y;
    uint32_t c = x & y;

    uint32_t retVal = (x & ~y) | (~b & c) | (a & ~c);
    return retVal;
}

__device__ __forceinline__
void hefty_gpu_round(uint32_t *regs, uint32_t W, uint32_t K, uint32_t *sponge)
{
    uint32_t tmpBr;

    uint32_t brG = Br(sponge, regs[6]);
    uint32_t brF = Br(sponge, regs[5]);
    uint32_t tmp1 = Ch(regs[4], brF, brG) + regs[7] + W + K;
    uint32_t brE = Br(sponge, regs[4]);
    uint32_t tmp2 = tmp1 + S1(brE);
    uint32_t brC = Br(sponge, regs[2]);
    uint32_t brB = Br(sponge, regs[1]);
    uint32_t brA = Br(sponge, regs[0]);
    uint32_t tmp3 = Maj(brA, brB, brC);
    tmpBr = Br(sponge, regs[0]);
    uint32_t tmp4 = tmp3 + S0(tmpBr);
    tmpBr = Br(sponge, tmp2);

    #pragma unroll 7
    for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
    regs[0] = tmp2 + tmp4;
    regs[4] += tmpBr;
}

__host__
void hefty_cpu_round(uint32_t *regs, uint32_t W, uint32_t K, uint32_t *sponge)
{
    uint32_t tmpBr;

    uint32_t brG = Br(sponge, regs[6]);
    uint32_t brF = Br(sponge, regs[5]);
    uint32_t tmp1 = Ch(regs[4], brF, brG) + regs[7] + W + K;
    uint32_t brE = Br(sponge, regs[4]);
    uint32_t tmp2 = tmp1 + S1(brE);
    uint32_t brC = Br(sponge, regs[2]);
    uint32_t brB = Br(sponge, regs[1]);
    uint32_t brA = Br(sponge, regs[0]);
    uint32_t tmp3 = Maj(brA, brB, brC);
    tmpBr = Br(sponge, regs[0]);
    uint32_t tmp4 = tmp3 + S0(tmpBr);
    tmpBr = Br(sponge, tmp2);

    for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
    regs[0] = tmp2 + tmp4;
    regs[4] += tmpBr;
}

__global__
void hefty_gpu_hash(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
#if USE_SHARED
    extern __shared__ unsigned char heftytab[];
    if(threadIdx.x < 64)
    {
        *((uint32_t*)heftytab + threadIdx.x) = hefty_gpu_constantTable[threadIdx.x];
    }

    __syncthreads();
#endif

    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        // bestimme den aktuellen Zähler
        uint32_t nounce = startNounce + thread;

        // jeder thread in diesem  Block bekommt sein eigenes W Array im Shared memory
        // reduktion von 256 byte auf 128 byte
        uint32_t W1[16];
        uint32_t W2[16];

        // Initialisiere die register a bis h mit der Hash-Tabelle
        uint32_t regs[8];
        uint32_t hash[8];
        uint32_t sponge[4];

#pragma unroll 4
        for(int k=0; k < 4; k++)
            sponge[k] = hefty_gpu_sponge[k];

        // pre
#pragma unroll 8
        for (int k=0; k < 8; k++)
        {
            regs[k] = hefty_gpu_register[k];
            hash[k] = regs[k];
        }

        //memcpy(W, &hefty_gpu_blockHeader[0], sizeof(uint32_t) * 16); // verbleibende 20 bytes aus Block 2 plus padding
#pragma unroll 16
        for(int k=0;k<16;k++)
            W1[k] = hefty_gpu_blockHeader[k];
        W1[3] = SWAB32(nounce);

        // 2. Runde
#pragma unroll 16
        for(int j=0;j<16;j++)
            Absorb(sponge, W1[j] ^ heftyLookUp(j));

// Progress W1 (Bytes 0...63)
#pragma unroll 16
        for(int j=0;j<16;j++)
        {
            Absorb(sponge, regs[3] ^ regs[7]);
            hefty_gpu_round(regs, W1[j], heftyLookUp(j), sponge);
        }

// Progress W2 (Bytes 64...127) then W3 (Bytes 128...191) ...


        for(int k=0;k<3;k++)
        {

            for(int j=0;j<2;j++)
                W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];

            for(int j=2;j<7;j++)
                W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];


            for(int j=7;j<15;j++)
                W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

            W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];


            for(int j=0;j<16;j++)
            {
                Absorb(sponge, regs[3] + regs[7]);
                hefty_gpu_round(regs, W2[j], heftyLookUp(j + ((k+1)<<4)), sponge);
            }

            for(int j=0;j<16;j++)
                W1[j] = W2[j];
        }

#pragma unroll 8
        for(int k=0;k<8;k++)
            hash[k] += regs[k];

#pragma unroll 8
        for(int k=0;k<8;k++)
            ((uint32_t*)outputHash)[(thread<<3)+k] = SWAB32(hash[k]);
    }
}

__host__
void hefty_cpu_init(int thr_id, uint32_t threads)
{
    cudaSetDevice(device_map[thr_id]);

    // Kopiere die Hash-Tabellen in den GPU-Speicher
    cudaMemcpyToSymbol( hefty_gpu_constantTable,
                        hefty_cpu_constantTable,
                        sizeof(uint32_t) * 64 );

    // Speicher für alle Hefty1 hashes belegen
    CUDA_SAFE_CALL(cudaMalloc(&heavy_heftyHashes[thr_id], (size_t) 32 * threads));
}

__host__
void hefty_cpu_free(int thr_id)
{
    cudaFree(heavy_heftyHashes[thr_id]);
}

__host__
void hefty_cpu_setBlock(int thr_id, uint32_t threads, void *data, int len)
// data muss 80/84-Byte haben!
{
    // Nachricht expandieren und setzen
    uint32_t msgBlock[32];

    memset(msgBlock, 0, sizeof(msgBlock));
    memcpy(&msgBlock[0], data, len);
    if (len == 84) {
        msgBlock[21] |= 0x80;
        msgBlock[31] = 672; // bitlen
    } else if (len == 80) {
        msgBlock[20] |= 0x80;
        msgBlock[31] = 640; // bitlen
    }

    for(int i=0;i<31;i++) // Byteorder drehen
        msgBlock[i] = SWAB32(msgBlock[i]);

    // die erste Runde wird auf der CPU durchgeführt, da diese für
    // alle Threads gleich ist. Der Hash wird dann an die Threads
    // übergeben

    // Erstelle expandierten Block W
    uint32_t W[64];
    memcpy(W, &msgBlock[0], sizeof(uint32_t) * 16);
    for(int j=16;j<64;j++)
        W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

    // Initialisiere die register a bis h mit der Hash-Tabelle
    uint32_t regs[8];
    uint32_t hash[8];
    uint32_t sponge[4];

    // pre
    memset(sponge, 0, sizeof(uint32_t) * 4);
    for (int k=0; k < 8; k++)
    {
        regs[k] = hefty_cpu_hashTable[k];
        hash[k] = regs[k];
    }

    // 1. Runde
    for(int j=0;j<16;j++)
        Absorb(sponge, W[j] ^ hefty_cpu_constantTable[j]);

    for(int j=0;j<16;j++)
    {
        Absorb(sponge, regs[3] ^ regs[7]);
        hefty_cpu_round(regs, W[j], hefty_cpu_constantTable[j], sponge);
    }

    for(int j=16;j<64;j++)
    {
        Absorb(sponge, regs[3] + regs[7]);
        hefty_cpu_round(regs, W[j], hefty_cpu_constantTable[j], sponge);
    }

    for(int k=0;k<8;k++)
        hash[k] += regs[k];

    // sponge speichern
    cudaMemcpyToSymbol(hefty_gpu_sponge, sponge, 16);
    // hash speichern
    cudaMemcpyToSymbol(hefty_gpu_register, hash, 32);
    // Blockheader setzen (korrekte Nonce fehlt da drin noch)
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(hefty_gpu_blockHeader, &msgBlock[16], 64));
}

__host__
void hefty_cpu_hash(int thr_id, uint32_t threads, int startNounce)
{
    uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
#if USE_SHARED
    int shared_size = 8 * 64 * sizeof(uint32_t);
#else
    int shared_size = 0;
#endif

    hefty_gpu_hash <<< grid, block, shared_size >>> (threads, startNounce, heavy_heftyHashes[thr_id]);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, 0, thr_id);
}

__global__
__launch_bounds__(128, 8)
void hefty_gpu_copy(const uint32_t threads, uint32_t* d_heftyhash, uint64_t* d_hash)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        const uint32_t offset = thread * 8U; // 32 / sizeof(uint32_t);
        uint4 *psrc = (uint4*) (&d_heftyhash[offset]);
        uint4 *pdst = (uint4*) (&d_hash[offset]);
        pdst[0] = psrc[0];
        pdst[1] = psrc[1];
        pdst[2] = make_uint4(0,0,0,0);
        pdst[3] = make_uint4(0,0,0,0);
    }
}

__host__
void hefty_copy_hashes(int thr_id, uint32_t threads, uint32_t* d_outputhash)
{
    const uint32_t threadsperblock = 128;
    dim3 grid((threads + threadsperblock - 1) / threadsperblock);
    dim3 block(threadsperblock);
    hefty_gpu_copy <<< grid, block >>> (threads, heavy_heftyHashes[thr_id], (uint64_t*) d_outputhash);
    cudaStreamSynchronize(NULL);
}

