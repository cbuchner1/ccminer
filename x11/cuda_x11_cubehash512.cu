#include "cuda_helper.h"

typedef unsigned char BitSequence;

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */

#if __CUDA_ARCH__ < 350
#define LROT(x,bits) ((x << bits) | (x >> (32 - bits)))
#else
#define LROT(x, bits) __funnelshift_l(x, x, bits)
#endif

#define ROTATEUPWARDS7(a)  LROT(a,7)
#define ROTATEUPWARDS11(a) LROT(a,11)

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

__device__ __constant__
static const uint32_t c_IV_512[32] = {
	0x2AEA2A61, 0x50F494D4, 0x2D538B8B,
	0x4167D83E, 0x3FEE2313, 0xC701CF8C,
	0xCC39968E, 0x50AC5695, 0x4D42C787,
	0xA647A8B3, 0x97CF0BEF, 0x825B4537,
	0xEEF864D2, 0xF22090C4, 0xD0E5CD33,
	0xA23911AE, 0xFCD398D9, 0x148FE485,
	0x1B017BEF, 0xB6444532, 0x6A536159,
	0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
	0xD65C8A2B, 0xA5A70E75, 0xB1C62456,
	0xBC796576, 0x1921C8F7, 0xE7989AF1,
	0x7795D246, 0xD43E3B44
};

__device__ __forceinline__
static void rrounds(uint32_t x[2][2][2][2][2])
{
    int r;
    int j;
    int k;
    int l;
    int m;

//#pragma unroll 16
    for (r = 0;r < CUBEHASH_ROUNDS;++r) {

        /* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[1][j][k][l][m] += x[0][j][k][l][m];

        /* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

        /* "swap x_00klm with x_01klm" */
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    SWAP(x[0][0][k][l][m],x[0][1][k][l][m])

        /* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] ^= x[1][j][k][l][m];

        /* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    SWAP(x[1][j][k][0][m],x[1][j][k][1][m])

        /* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[1][j][k][l][m] += x[0][j][k][l][m];

        /* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

        /* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    SWAP(x[0][j][0][l][m],x[0][j][1][l][m])

        /* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
#pragma unroll 2
                    for (m = 0;m < 2;++m)
                        x[0][j][k][l][m] ^= x[1][j][k][l][m];

        /* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
        for (j = 0;j < 2;++j)
#pragma unroll 2
            for (k = 0;k < 2;++k)
#pragma unroll 2
                for (l = 0;l < 2;++l)
                    SWAP(x[1][j][k][l][0],x[1][j][k][l][1])

    }
}

__device__ __forceinline__
static void block_tox(uint32_t block[16], uint32_t x[2][2][2][2][2])
{
    int k;
    int l;
    int m;
    uint32_t *in = block;

#pragma unroll 2
    for (k = 0;k < 2;++k)
#pragma unroll 2
        for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m)
                x[0][0][k][l][m] ^= *in++;
}

__device__ __forceinline__
static void hash_fromx(uint32_t hash[16], uint32_t x[2][2][2][2][2])
{
    int j;
    int k;
    int l;
    int m;
    uint32_t *out = hash;

#pragma unroll 2
    for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
            for (l = 0;l < 2;++l)
#pragma unroll 2
                for (m = 0;m < 2;++m)
                    *out++ = x[0][j][k][l][m];
}

__device__
void Init(uint32_t x[2][2][2][2][2])
{
    int i,j,k,l,m;
#if 0
    /* "the first three state words x_00000, x_00001, x_00010" */
    /* "are set to the integers h/8, b, r respectively." */
    /* "the remaining state words are set to 0." */
#pragma unroll 2
    for (i = 0;i < 2;++i)
#pragma unroll 2
      for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
          for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m)
              x[i][j][k][l][m] = 0;
    x[0][0][0][0][0] = 512/8;
    x[0][0][0][0][1] = CUBEHASH_BLOCKBYTES;
    x[0][0][0][1][0] = CUBEHASH_ROUNDS;

    /* "the state is then transformed invertibly through 10r identical rounds */
    for (i = 0;i < 10;++i) rrounds(x);
#else
    const uint32_t *iv = c_IV_512;

#pragma unroll 2
    for (i = 0;i < 2;++i)
#pragma unroll 2
      for (j = 0;j < 2;++j)
#pragma unroll 2
        for (k = 0;k < 2;++k)
#pragma unroll 2
          for (l = 0;l < 2;++l)
#pragma unroll 2
            for (m = 0;m < 2;++m)
              x[i][j][k][l][m] = *iv++;
#endif
}

__device__ __forceinline__
static void Update32(uint32_t x[2][2][2][2][2], const BitSequence *data)
{
    /* "xor the block into the first b bytes of the state" */
    /* "and then transform the state invertibly through r identical rounds" */
    block_tox((uint32_t*)data, x);
    rrounds(x);
}

__device__ __forceinline__
static void Final(uint32_t x[2][2][2][2][2], BitSequence *hashval)
{
    int i;

    /* "the integer 1 is xored into the last state word x_11111" */
    x[1][1][1][1][1] ^= 1;

    /* "the state is then transformed invertibly through 10r identical rounds" */
#pragma unroll 10
    for (i = 0;i < 10;++i) rrounds(x);

    /* "output the first h/8 bytes of the state" */
    hash_fromx((uint32_t*)hashval, x);
}


/***************************************************/

__global__
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

		uint32_t x[2][2][2][2][2];
		Init(x);

		// erste Hälfte des Hashes (32 bytes)
		Update32(x, (const BitSequence*)Hash);

		// zweite Hälfte des Hashes (32 bytes)
		Update32(x, (const BitSequence*)(Hash+8));

		// Padding Block
		uint32_t last[8];
		last[0] = 0x80;
		#pragma unroll 7
		for (int i=1; i < 8; i++) last[i] = 0;
		Update32(x, (const BitSequence*)last);

		Final(x, (BitSequence*)Hash);
	}
}

__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	x11_cubehash512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}

__host__
void x11_cubehash512_cpu_init(int thr_id, uint32_t threads) { }


/***************************************************/

#define WANT_CUBEHASH80
#ifdef WANT_CUBEHASH80

__constant__
static uint32_t c_PaddedMessage80[20];

__host__
void cubehash512_setBlock_80(int thr_id, uint32_t* endiandata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, endiandata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
void cubehash512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint64_t *g_outhash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNounce + thread;

		uint32_t x[2][2][2][2][2];
		Init(x);

		uint32_t message[8];
		// first 32 bytes
		AS_UINT4(&message[0]) = AS_UINT4(&c_PaddedMessage80[0]);
		AS_UINT4(&message[4]) = AS_UINT4(&c_PaddedMessage80[4]);
		Update32(x, (const BitSequence*)message);

		// second 32 bytes
		AS_UINT4(&message[0]) = AS_UINT4(&c_PaddedMessage80[8]);
		AS_UINT4(&message[4]) = AS_UINT4(&c_PaddedMessage80[12]);
		Update32(x, (const BitSequence*)message);

		// last 16 bytes + Padding
		AS_UINT4(&message[0]) = AS_UINT4(&c_PaddedMessage80[16]);
		message[3] = cuda_swab32(nonce);
		message[4] = 0x80;
		message[5] = 0;
		message[6] = 0;
		message[7] = 0;
		Update32(x, (const BitSequence*)message);

		BitSequence* output = (BitSequence*) (&g_outhash[(size_t)8 * thread]);
		Final(x, output);
	}
}

__host__
void cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cubehash512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, (uint64_t*) d_hash);
}

#endif