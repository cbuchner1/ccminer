/* SM 2/3/3.5 Variant for lyra2REv2 */

#ifdef __INTELLISENSE__
/* just for vstudio code colors, only uncomment that temporary, dont commit it */
//#undef __CUDA_ARCH__
//#define __CUDA_ARCH__ 500
#endif

#define TPB20 64
#define TPB30 64
#define TPB35 64

#if __CUDA_ARCH__ >= 200 && __CUDA_ARCH__ < 500

#include "cuda_lyra2_vectors.h"

#define Nrow 4
#define Ncol 4

#define vectype ulonglong4
#define memshift 4

__device__ vectype *DMatrix;

static __device__ __forceinline__
void Gfunc_v35(unsigned long long &a, unsigned long long &b, unsigned long long &c, unsigned long long &d)
{
	a += b; d ^= a; d = ROTR64(d, 32);
	c += d; b ^= c; b = ROTR64(b, 24);
	a += b; d ^= a; d = ROTR64(d, 16);
	c += d; b ^= c; b = ROTR64(b, 63);
}

static __device__ __forceinline__
void round_lyra_v35(vectype* s)
{
	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__
void reduceDuplexV3(vectype state[4], uint32_t thread)
{
	vectype state1[3];
	uint32_t ps1 = (Nrow * Ncol * memshift * thread);
	uint32_t ps2 = (memshift * (Ncol - 1) * Nrow + memshift * 1 + Nrow * Ncol * memshift * thread);

	#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + Nrow * i *memshift;
		uint32_t s2 = ps2 - Nrow * i *memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);

		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];
		round_lyra_v35(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state1[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetupV3(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{
	vectype state2[3], state1[3];

	uint32_t ps1 = (memshift * rowIn    + Nrow * Ncol * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (Nrow * memshift * (Ncol - 1) + memshift *  rowOut + Nrow * Ncol * memshift * thread);

	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + Nrow*i*memshift;
		uint32_t s2 = ps2 + Nrow*i*memshift;
		uint32_t s3 = ps3 - Nrow*i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1 )[j]);
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2 )[j]);
		for (int j = 0; j < 3; j++) {
			vectype tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}

		round_lyra_v35(state);

		for (int j = 0; j < 3; j++) {
			state1[j] ^= state[j];
			(DMatrix + s3)[j] = state1[j];
		}

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowtV3(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{
	vectype state1[3], state2[3];
	uint32_t ps1 = (memshift * rowIn    + Nrow * Ncol * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (memshift * rowOut   + Nrow * Ncol * memshift * thread);

	#pragma nounroll
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + Nrow * i*memshift;
		uint32_t s2 = ps2 + Nrow * i*memshift;
		uint32_t s3 = ps3 + Nrow * i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);

		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);

		for (int j = 0; j < 3; j++)
			state1[j] += state2[j];

		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra_v35(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

		if (rowInOut != rowOut) {

			for (int j = 0; j < 3; j++)
				(DMatrix + s2)[j] = state2[j];

			for (int j = 0; j < 3; j++)
				(DMatrix + s3)[j] ^= state[j];

		} else {

			for (int j = 0; j < 3; j++)
				state2[j] ^= state[j];

			for (int j = 0; j < 3; j++)
				(DMatrix + s2)[j] = state2[j];
		}
	}
}

#if __CUDA_ARCH__ >= 300
__global__ __launch_bounds__(TPB35, 1)
void lyra2v2_gpu_hash_32_v3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	vectype state[4];
	vectype blake2b_IV[2];
	vectype padding[2];

	if (threadIdx.x == 0) {

		((uint16*)blake2b_IV)[0] = make_uint16(
			0xf3bcc908, 0x6a09e667 , 0x84caa73b, 0xbb67ae85,
			0xfe94f82b, 0x3c6ef372 , 0x5f1d36f1, 0xa54ff53a,
			0xade682d1, 0x510e527f , 0x2b3e6c1f, 0x9b05688c,
			0xfb41bd6b, 0x1f83d9ab , 0x137e2179, 0x5be0cd19
		);
		((uint16*)padding)[0] = make_uint16(
			0x20, 0x0 , 0x20, 0x0 , 0x20, 0x0 , 0x01, 0x0,
			0x04, 0x0 , 0x04, 0x0 , 0x80, 0x0 , 0x0, 0x01000000
		);
	}

	if (thread < threads)
	{
		((uint2*)state)[0] = __ldg(&outputHash[thread]);
		((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
		((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);

		state[1] = state[0];
		state[2] = shuffle4(((vectype*)blake2b_IV)[0], 0);
		state[3] = shuffle4(((vectype*)blake2b_IV)[1], 0);

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		state[0] ^= shuffle4(((vectype*)padding)[0], 0);
		state[1] ^= shuffle4(((vectype*)padding)[1], 0);

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		uint32_t ps1 = (4 * memshift * 3 + 16 * memshift * thread);

		//#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			uint32_t s1 = ps1 - 4 * memshift * i;
			for (int j = 0; j < 3; j++)
				(DMatrix + s1)[j] = (state)[j];

			round_lyra_v35(state);
		}

		reduceDuplexV3(state, thread);
		reduceDuplexRowSetupV3(1, 0, 2, state, thread);
		reduceDuplexRowSetupV3(2, 1, 3, state, thread);

		uint32_t rowa;
		int prev = 3;
		for (int i = 0; i < 4; i++)
		{
			rowa = ((uint2*)state)[0].x & 3;  reduceDuplexRowtV3(prev, rowa, i, state, thread);
			prev = i;
		}

		uint32_t shift = (memshift * rowa + 16 * memshift * thread);

		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);

		outputHash[thread] = ((uint2*)state)[0];
		outputHash[thread + threads] = ((uint2*)state)[1];
		outputHash[thread + 2 * threads] = ((uint2*)state)[2];
		outputHash[thread + 3 * threads] = ((uint2*)state)[3];

	} //thread
}
#elif __CUDA_ARCH__ >= 200
__global__ __launch_bounds__(TPB20, 1)
void lyra2v2_gpu_hash_32_v3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	vectype state[4];
	vectype blake2b_IV[2];
	vectype padding[2];

	((uint16*)blake2b_IV)[0] = make_uint16(
		0xf3bcc908, 0x6a09e667, 0x84caa73b, 0xbb67ae85,
		0xfe94f82b, 0x3c6ef372, 0x5f1d36f1, 0xa54ff53a,
		0xade682d1, 0x510e527f, 0x2b3e6c1f, 0x9b05688c,
		0xfb41bd6b, 0x1f83d9ab, 0x137e2179, 0x5be0cd19
	);
	((uint16*)padding)[0] = make_uint16(
		0x20, 0x0, 0x20, 0x0, 0x20, 0x0, 0x01, 0x0,
		0x04, 0x0, 0x04, 0x0, 0x80, 0x0, 0x0, 0x01000000
	);

	if (thread < threads)
	{

		((uint2*)state)[0] = outputHash[thread];
		((uint2*)state)[1] = outputHash[thread + threads];
		((uint2*)state)[2] = outputHash[thread + 2 * threads];
		((uint2*)state)[3] = outputHash[thread + 3 * threads];

		state[1] = state[0];
		state[2] = ((vectype*)blake2b_IV)[0];
		state[3] = ((vectype*)blake2b_IV)[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		state[0] ^= ((vectype*)padding)[0];
		state[1] ^= ((vectype*)padding)[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		uint32_t ps1 = (4 * memshift * 3 + 16 * memshift * thread);

		//#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			uint32_t s1 = ps1 - 4 * memshift * i;
			for (int j = 0; j < 3; j++)
				(DMatrix + s1)[j] = (state)[j];

			round_lyra_v35(state);
		}

		reduceDuplexV3(state, thread);
		reduceDuplexRowSetupV3(1, 0, 2, state, thread);
		reduceDuplexRowSetupV3(2, 1, 3, state, thread);

		uint32_t rowa;
		int prev = 3;
		for (int i = 0; i < 4; i++)
		{
			rowa = ((uint2*)state)[0].x & 3;  reduceDuplexRowtV3(prev, rowa, i, state, thread);
			prev = i;
		}

		uint32_t shift = (memshift * rowa + 16 * memshift * thread);

		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);

		outputHash[thread] = ((uint2*)state)[0];
		outputHash[thread + threads] = ((uint2*)state)[1];
		outputHash[thread + 2 * threads] = ((uint2*)state)[2];
		outputHash[thread + 3 * threads] = ((uint2*)state)[3];

	} //thread
}
#endif

#else
/* host & sm5+ */
__global__ void lyra2v2_gpu_hash_32_v3(uint32_t threads, uint32_t startNounce, uint2 *outputHash) {}
#endif
