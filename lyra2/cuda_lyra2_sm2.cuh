#include <memory.h>

#ifdef __INTELLISENSE__
/* just for vstudio code colors, only uncomment that temporary, dont commit it */
//#undef __CUDA_ARCH__
//#define __CUDA_ARCH__ 500
#endif

#include "cuda_helper.h"

#define TPB30 160
#define TPB20 160

#if (__CUDA_ARCH__ >= 200 && __CUDA_ARCH__ <= 350) || !defined(__CUDA_ARCH__)
__constant__ static uint2 blake2b_IV_sm2[8] = {
	{ 0xf3bcc908, 0x6a09e667 },
	{ 0x84caa73b, 0xbb67ae85 },
	{ 0xfe94f82b, 0x3c6ef372 },
	{ 0x5f1d36f1, 0xa54ff53a },
	{ 0xade682d1, 0x510e527f },
	{ 0x2b3e6c1f, 0x9b05688c },
	{ 0xfb41bd6b, 0x1f83d9ab },
	{ 0x137e2179, 0x5be0cd19 }
};
#endif

#if __CUDA_ARCH__ >= 200 && __CUDA_ARCH__ <= 350

#define reduceDuplexRow(rowIn, rowInOut, rowOut) { \
	for (int i = 0; i < 8; i++) { \
		for (int j = 0; j < 12; j++) \
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut]; \
		round_lyra(state); \
		for (int j = 0; j < 12; j++) \
			Matrix[j + 12 * i][rowOut] ^= state[j]; \
		Matrix[0 + 12 * i][rowInOut] ^= state[11]; \
		Matrix[1 + 12 * i][rowInOut] ^= state[0]; \
		Matrix[2 + 12 * i][rowInOut] ^= state[1]; \
		Matrix[3 + 12 * i][rowInOut] ^= state[2]; \
		Matrix[4 + 12 * i][rowInOut] ^= state[3]; \
		Matrix[5 + 12 * i][rowInOut] ^= state[4]; \
		Matrix[6 + 12 * i][rowInOut] ^= state[5]; \
		Matrix[7 + 12 * i][rowInOut] ^= state[6]; \
		Matrix[8 + 12 * i][rowInOut] ^= state[7]; \
		Matrix[9 + 12 * i][rowInOut] ^= state[8]; \
		Matrix[10+ 12 * i][rowInOut] ^= state[9]; \
		Matrix[11+ 12 * i][rowInOut] ^= state[10]; \
	} \
  }

#define absorbblock(in)  { \
	state[0] ^= Matrix[0][in]; \
	state[1] ^= Matrix[1][in]; \
	state[2] ^= Matrix[2][in]; \
	state[3] ^= Matrix[3][in]; \
	state[4] ^= Matrix[4][in]; \
	state[5] ^= Matrix[5][in]; \
	state[6] ^= Matrix[6][in]; \
	state[7] ^= Matrix[7][in]; \
	state[8] ^= Matrix[8][in]; \
	state[9] ^= Matrix[9][in]; \
	state[10] ^= Matrix[10][in]; \
	state[11] ^= Matrix[11][in]; \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
	round_lyra(state); \
  }

static __device__ __forceinline__
void Gfunc(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__
static void round_lyra(uint2 *s)
{
	Gfunc(s[0], s[4], s[8],  s[12]);
	Gfunc(s[1], s[5], s[9],  s[13]);
	Gfunc(s[2], s[6], s[10], s[14]);
	Gfunc(s[3], s[7], s[11], s[15]);
	Gfunc(s[0], s[5], s[10], s[15]);
	Gfunc(s[1], s[6], s[11], s[12]);
	Gfunc(s[2], s[7], s[8],  s[13]);
	Gfunc(s[3], s[4], s[9],  s[14]);
}

__device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[16], uint2 Matrix[96][8])
{
#if __CUDA_ARCH__ > 500
	#pragma unroll
#endif
	for (int i = 0; i < 8; i++)
	{
		#pragma unroll
		for (int j = 0; j < 12; j++)
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 12; j++)
			Matrix[j + 84 - 12 * i][rowOut] = Matrix[12 * i + j][rowIn] ^ state[j];

		Matrix[0 + 12 * i][rowInOut] ^= state[11];
		Matrix[1 + 12 * i][rowInOut] ^= state[0];
		Matrix[2 + 12 * i][rowInOut] ^= state[1];
		Matrix[3 + 12 * i][rowInOut] ^= state[2];
		Matrix[4 + 12 * i][rowInOut] ^= state[3];
		Matrix[5 + 12 * i][rowInOut] ^= state[4];
		Matrix[6 + 12 * i][rowInOut] ^= state[5];
		Matrix[7 + 12 * i][rowInOut] ^= state[6];
		Matrix[8 + 12 * i][rowInOut] ^= state[7];
		Matrix[9 + 12 * i][rowInOut] ^= state[8];
		Matrix[10 + 12 * i][rowInOut] ^= state[9];
		Matrix[11 + 12 * i][rowInOut] ^= state[10];
	}
}

__global__ __launch_bounds__(TPB30, 1)
void lyra2_gpu_hash_32_sm2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 state[16];

		#pragma unroll
		for (int i = 0; i<4; i++) {
			LOHI(state[i].x, state[i].y, g_hash[threads*i + thread]);
		} //password

		#pragma unroll
		for (int i = 0; i<4; i++) {
			state[i + 4] = state[i];
		} //salt

		#pragma unroll
		for (int i = 0; i<8; i++) {
			state[i + 8] = blake2b_IV_sm2[i];
		}

		// blake2blyra x2
		//#pragma unroll 24
		for (int i = 0; i<24; i++) {
			round_lyra(state);
		} //because 12 is not enough

		uint2 Matrix[96][8]; // not cool

		// reducedSqueezeRow0
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			#pragma unroll 12
			for (int j = 0; j<12; j++) {
				Matrix[j + 84 - 12 * i][0] = state[j];
			}
			round_lyra(state);
		}

		// reducedSqueezeRow1
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			#pragma unroll 12
			for (int j = 0; j<12; j++) {
				state[j] ^= Matrix[j + 12 * i][0];
			}
			round_lyra(state);
			#pragma unroll 12
			for (int j = 0; j<12; j++) {
				Matrix[j + 84 - 12 * i][1] = Matrix[j + 12 * i][0] ^ state[j];
			}
		}

		reduceDuplexRowSetup(1, 0, 2, state, Matrix);
		reduceDuplexRowSetup(2, 1, 3, state, Matrix);
		reduceDuplexRowSetup(3, 0, 4, state, Matrix);
		reduceDuplexRowSetup(4, 3, 5, state, Matrix);
		reduceDuplexRowSetup(5, 2, 6, state, Matrix);
		reduceDuplexRowSetup(6, 1, 7, state, Matrix);

		uint32_t rowa;
		rowa = state[0].x & 7;
		reduceDuplexRow(7, rowa, 0);
		rowa = state[0].x & 7;
		reduceDuplexRow(0, rowa, 3);
		rowa = state[0].x & 7;
		reduceDuplexRow(3, rowa, 6);
		rowa = state[0].x & 7;
		reduceDuplexRow(6, rowa, 1);
		rowa = state[0].x & 7;
		reduceDuplexRow(1, rowa, 4);
		rowa = state[0].x & 7;
		reduceDuplexRow(4, rowa, 7);
		rowa = state[0].x & 7;
		reduceDuplexRow(7, rowa, 2);
		rowa = state[0].x & 7;
		reduceDuplexRow(2, rowa, 5);

		absorbblock(rowa);

		#pragma unroll
		for (int i = 0; i<4; i++) {
			g_hash[threads*i + thread] = devectorize(state[i]);
		}

	} //thread
}

#else
/* if __CUDA_ARCH__ < 200 .. host */
__global__ void lyra2_gpu_hash_32_sm2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash) {}
#endif
