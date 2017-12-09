#include "cuda_helper.h"

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */

#if __CUDA_ARCH__ < 350
#define LROT(x,bits) ((x << bits) | (x >> (32 - bits)))
#else
#define LROT(x, bits) __funnelshift_l(x, x, bits)
#endif

#if __CUDA_ARCH__ < 500
#define TPB 576
#else
#define TPB 1024
#endif

#define ROTATEUPWARDS7(a)  LROT(a,7)
#define ROTATEUPWARDS11(a) LROT(a,11)

//#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }
#define SWAP(a,b) { a ^= b; b ^= a; a ^= b; }

__device__ __forceinline__ void rrounds(uint32_t x[2][2][2][2][2])
{
	int r;
	int j;
	int k;
	int l;
	int m;

	#pragma unroll 2
	for (r = 0; r < CUBEHASH_ROUNDS; ++r) {

		/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

		/* "swap x_00klm with x_01klm" */
#pragma unroll 2
		for (k = 0; k < 2; ++k)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][0][k][l][m], x[0][1][k][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[1][j][k][0][m], x[1][j][k][1][m])

					/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

		/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][j][0][l][m], x[0][j][1][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
					SWAP(x[1][j][k][l][0], x[1][j][k][l][1])

	}
}

__device__ __forceinline__ void block_tox(const uint32_t *in, uint32_t x[2][2][2][2][2])
{
	x[0][0][0][0][0] ^= in[0];
	x[0][0][0][0][1] ^= in[1];
	x[0][0][0][1][0] ^= in[2];
	x[0][0][0][1][1] ^= in[3];
	x[0][0][1][0][0] ^= in[4];
	x[0][0][1][0][1] ^= in[5];
	x[0][0][1][1][0] ^= in[6];
	x[0][0][1][1][1] ^= in[7];
}

__device__ __forceinline__ void hash_fromx(uint32_t *out, uint32_t x[2][2][2][2][2])
{
	out[0] = x[0][0][0][0][0];
	out[1] = x[0][0][0][0][1];
	out[2] = x[0][0][0][1][0];
	out[3] = x[0][0][0][1][1];
	out[4] = x[0][0][1][0][0];
	out[5] = x[0][0][1][0][1];
	out[6] = x[0][0][1][1][0];
	out[7] = x[0][0][1][1][1];

}

__device__ __forceinline__
void Update32(uint32_t x[2][2][2][2][2], const uint32_t *data)
{
	/* "xor the block into the first b bytes of the state" */
	/* "and then transform the state invertibly through r identical rounds" */
	block_tox(data, x);
	rrounds(x);
}

__device__ __forceinline__
void Update32_const(uint32_t x[2][2][2][2][2])
{
	x[0][0][0][0][0] ^= 0x80;
	rrounds(x);
}

__device__ __forceinline__
void Final(uint32_t x[2][2][2][2][2], uint32_t *hashval)
{
	/* "the integer 1 is xored into the last state word x_11111" */
	x[1][1][1][1][1] ^= 1U;

	/* "the state is then transformed invertibly through 10r identical rounds" */
	#pragma unroll 2
	for (int i = 0; i < 10; ++i) rrounds(x);

	/* "output the first h/8 bytes of the state" */
	hash_fromx(hashval, x);
}

#if __CUDA_ARCH__ >= 500

__global__	__launch_bounds__(TPB, 1)
void cubehash256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 Hash[4];

		Hash[0] = __ldg(&g_hash[thread]);
		Hash[1] = __ldg(&g_hash[thread + 1 * threads]);
		Hash[2] = __ldg(&g_hash[thread + 2 * threads]);
		Hash[3] = __ldg(&g_hash[thread + 3 * threads]);

		uint32_t x[2][2][2][2][2] =
		{
			0xEA2BD4B4, 0xCCD6F29F, 0x63117E71, 0x35481EAE,
			0x22512D5B, 0xE5D94E63, 0x7E624131, 0xF4CC12BE,
			0xC2D0B696, 0x42AF2070, 0xD0720C35, 0x3361DA8C,
			0x28CCECA4, 0x8EF8AD83, 0x4680AC00, 0x40E5FBAB,
			0xD89041C3, 0x6107FBD5, 0x6C859D41, 0xF0B26679,
			0x09392549, 0x5FA25603, 0x65C892FD, 0x93CB6285,
			0x2AF2B5AE, 0x9E4B4E60, 0x774ABFDD, 0x85254725,
			0x15815AEB, 0x4AB6AAD6, 0x9CDAF8AF, 0xD6032C0A
		};

		x[0][0][0][0][0] ^= Hash[0].x;
		x[0][0][0][0][1] ^= Hash[0].y;
		x[0][0][0][1][0] ^= Hash[1].x;
		x[0][0][0][1][1] ^= Hash[1].y;
		x[0][0][1][0][0] ^= Hash[2].x;
		x[0][0][1][0][1] ^= Hash[2].y;
		x[0][0][1][1][0] ^= Hash[3].x;
		x[0][0][1][1][1] ^= Hash[3].y;

		rrounds(x);
		x[0][0][0][0][0] ^= 0x80U;
		rrounds(x);

		Final(x, (uint32_t*) Hash);

		g_hash[thread] =               Hash[0];
		g_hash[1 * threads + thread] = Hash[1];
		g_hash[2 * threads + thread] = Hash[2];
		g_hash[3 * threads + thread] = Hash[3];
	}
}

#else

__global__	__launch_bounds__(TPB, 1)
void cubehash256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *d_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t Hash[8];
		uint64_t* g_hash = (uint64_t*) d_hash;

		LOHI(Hash[0], Hash[1], __ldg(&g_hash[thread]));
		LOHI(Hash[2], Hash[3], __ldg(&g_hash[thread + 1 * threads]));
		LOHI(Hash[4], Hash[5], __ldg(&g_hash[thread + 2 * threads]));
		LOHI(Hash[6], Hash[7], __ldg(&g_hash[thread + 3 * threads]));

		uint32_t x[2][2][2][2][2] =
		{
			0xEA2BD4B4, 0xCCD6F29F, 0x63117E71, 0x35481EAE,
			0x22512D5B, 0xE5D94E63, 0x7E624131, 0xF4CC12BE,
			0xC2D0B696, 0x42AF2070, 0xD0720C35, 0x3361DA8C,
			0x28CCECA4, 0x8EF8AD83, 0x4680AC00, 0x40E5FBAB,
			0xD89041C3, 0x6107FBD5, 0x6C859D41, 0xF0B26679,
			0x09392549, 0x5FA25603, 0x65C892FD, 0x93CB6285,
			0x2AF2B5AE, 0x9E4B4E60, 0x774ABFDD, 0x85254725,
			0x15815AEB, 0x4AB6AAD6, 0x9CDAF8AF, 0xD6032C0A
		};

		x[0][0][0][0][0] ^= Hash[0];
		x[0][0][0][0][1] ^= Hash[1];
		x[0][0][0][1][0] ^= Hash[2];
		x[0][0][0][1][1] ^= Hash[3];
		x[0][0][1][0][0] ^= Hash[4];
		x[0][0][1][0][1] ^= Hash[5];
		x[0][0][1][1][0] ^= Hash[6];
		x[0][0][1][1][1] ^= Hash[7];

		rrounds(x);
		x[0][0][0][0][0] ^= 0x80U;
		rrounds(x);

		Final(x, Hash);

		g_hash[thread] =               ((uint64_t*)Hash)[0];
		g_hash[1 * threads + thread] = ((uint64_t*)Hash)[1];
		g_hash[2 * threads + thread] = ((uint64_t*)Hash)[2];
		g_hash[3 * threads + thread] = ((uint64_t*)Hash)[3];
	}
}

#endif

__host__
void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order)
{
	uint32_t tpb = TPB;

	dim3 grid((threads + tpb-1)/tpb);
	dim3 block(tpb);

	cubehash256_gpu_hash_32 <<<grid, block>>> (threads, startNounce, (uint2*) d_hash);
}
