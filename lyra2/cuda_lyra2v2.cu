#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define TPB52 8
#define TPB50 16

#include "cuda_lyra2v2_sm3.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#if __CUDA_ARCH__ >= 500

#include "cuda_lyra2_vectors.h"

#define Nrow 4
#define Ncol 4
#define memshift 3

__device__ uint2x4 *DMatrix;

__device__ __forceinline__
void Gfunc_v5(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__
void round_lyra_v5(uint2x4* s)
{
	Gfunc_v5(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v5(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v5(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v5(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v5(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v5(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v5(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v5(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__
void reduceDuplex(uint2x4 state[4], const uint32_t thread)
{
	uint2x4 state1[3];
	const uint32_t ps1 = (Nrow * Ncol * memshift * thread);
	const uint32_t ps2 = (memshift * (Ncol-1) + memshift * Ncol + Nrow * Ncol * memshift * thread);

	#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + i*memshift;
		uint32_t s2 = ps2 - i*memshift;

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix+s1)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra_v5(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state1[j];
	}
}

__device__ __forceinline__
void reduceDuplex50(uint2x4 state[4], const uint32_t thread)
{
	const uint32_t ps1 = (Nrow * Ncol * memshift * thread);
	const uint32_t ps2 = (memshift * (Ncol - 1) + memshift * Ncol + Nrow * Ncol * memshift * thread);

	#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const int32_t s2 = ps2 - i*memshift;

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + s1)[j]);

		round_lyra_v5(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = __ldg4(&(DMatrix + s1)[j]) ^ state[j];
	}
}

__device__ __forceinline__
void reduceDuplexRowSetupV2(const int rowIn, const int rowInOut, const int rowOut, uint2x4 state[4], const uint32_t thread)
{
	uint2x4 state2[3], state1[3];

	const uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
	const uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
	const uint32_t ps3 = (memshift * (Ncol-1) + memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);

	for (int i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 - i*memshift;

#if __CUDA_ARCH__ == 500

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] = state[j] ^ (__ldg4(&(DMatrix + s1)[j]) + __ldg4(&(DMatrix + s2)[j]));

		round_lyra_v5(state);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			state1[j] ^= state[j];
			(DMatrix + s3)[j] = state1[j];
		}

#else /* 5.2 */

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);
		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			uint2x4 tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}

		round_lyra_v5(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			state1[j] ^= state[j];
			(DMatrix + s3)[j] = state1[j];
		}

#endif
		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		#pragma unroll
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j+1] ^= ((uint2*)state)[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];
	}
}


__device__ __forceinline__
void reduceDuplexRowtV2(const int rowIn, const int rowInOut, const int rowOut, uint2x4* state, const uint32_t thread)
{
	uint2x4 state1[3], state2[3];
	const uint32_t ps1 = (memshift * Ncol * rowIn    + Nrow * Ncol * memshift * thread);
	const uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
	const uint32_t ps3 = (memshift * Ncol * rowOut   + Nrow * Ncol * memshift * thread);

	for (int i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 + i*memshift;

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);


		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] += state2[j];

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra_v5(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		#pragma unroll
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

#if __CUDA_ARCH__ == 500
		if (rowInOut != rowOut)
		{
			#pragma unroll
			for (int j = 0; j < 3; j++)
				(DMatrix + s3)[j] ^= state[j];

		}
		if (rowInOut == rowOut)
		{
			#pragma unroll
			for (int j = 0; j < 3; j++)
				state2[j] ^= state[j];
		}
#else
		if (rowInOut != rowOut)
		{
			#pragma unroll
			for (int j = 0; j < 3; j++)
				(DMatrix + s3)[j] ^= state[j];
		} else {
			#pragma unroll
			for (int j = 0; j < 3; j++)
				state2[j] ^= state[j];
		}
#endif
		#pragma unroll
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];
	}
}


#if __CUDA_ARCH__ == 500
__global__ __launch_bounds__(TPB50, 1)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void lyra2v2_gpu_hash_32(const uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint2x4 blake2b_IV[2];

	if (threadIdx.x == 0) {

		((uint16*)blake2b_IV)[0] = make_uint16(
			0xf3bcc908, 0x6a09e667, 0x84caa73b, 0xbb67ae85,
			0xfe94f82b, 0x3c6ef372, 0x5f1d36f1, 0xa54ff53a,
			0xade682d1, 0x510e527f, 0x2b3e6c1f, 0x9b05688c,
			0xfb41bd6b, 0x1f83d9ab, 0x137e2179, 0x5be0cd19
		);
	}

	if (thread < threads)
	{
		uint2x4 state[4];

		((uint2*)state)[0] = __ldg(&g_hash[thread]);
		((uint2*)state)[1] = __ldg(&g_hash[thread + threads]);
		((uint2*)state)[2] = __ldg(&g_hash[thread + threads*2]);
		((uint2*)state)[3] = __ldg(&g_hash[thread + threads*3]);

		state[1] = state[0];

		state[2] = ((blake2b_IV)[0]);
		state[3] = ((blake2b_IV)[1]);

		for (int i = 0; i<12; i++)
			round_lyra_v5(state);

		((uint2*)state)[0].x ^= 0x20;
		((uint2*)state)[1].x ^= 0x20;
		((uint2*)state)[2].x ^= 0x20;
		((uint2*)state)[3].x ^= 0x01;
		((uint2*)state)[4].x ^= 0x04;
		((uint2*)state)[5].x ^= 0x04;
		((uint2*)state)[6].x ^= 0x80;
		((uint2*)state)[7].y ^= 0x01000000;

		for (int i = 0; i<12; i++)
			round_lyra_v5(state);

		const uint32_t ps1 = (memshift * (Ncol - 1) + Nrow * Ncol * memshift * thread);

		for (int i = 0; i < Ncol; i++)
		{
			const uint32_t s1 = ps1 - memshift * i;
			DMatrix[s1] = state[0];
			DMatrix[s1+1] = state[1];
			DMatrix[s1+2] = state[2];
			round_lyra_v5(state);
		}

		reduceDuplex50(state, thread);

		reduceDuplexRowSetupV2(1, 0, 2, state, thread);
		reduceDuplexRowSetupV2(2, 1, 3, state, thread);

		uint32_t rowa;
		int prev=3;

		for (int i = 0; i < 4; i++)
		{
			rowa = ((uint2*)state)[0].x & 3;
			reduceDuplexRowtV2(prev, rowa, i, state, thread);
			prev = i;
		}

		const uint32_t shift = (memshift * Ncol * rowa + Nrow * Ncol * memshift * thread);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra_v5(state);

		g_hash[thread]             = ((uint2*)state)[0];
		g_hash[thread + threads]   = ((uint2*)state)[1];
		g_hash[thread + threads*2] = ((uint2*)state)[2];
		g_hash[thread + threads*3] = ((uint2*)state)[3];
	}
}
#else
#include "cuda_helper.h"
#if __CUDA_ARCH__ < 200
__device__ void* DMatrix;
#endif
__global__ void lyra2v2_gpu_hash_32(const uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
#endif

__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	cuda_get_arch(thr_id);
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order)
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	uint32_t tpb = TPB52;

	if (cuda_arch[dev_id] > 500) tpb = TPB52;
	else if (cuda_arch[dev_id] == 500) tpb = TPB50;
	else if (cuda_arch[dev_id] >= 350) tpb = TPB35;
	else if (cuda_arch[dev_id] >= 300) tpb = TPB30;
	else if (cuda_arch[dev_id] >= 200) tpb = TPB20;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	if (device_sm[dev_id] >= 500 && cuda_arch[dev_id] >= 500)
		lyra2v2_gpu_hash_32    <<<grid, block>>> (threads, startNounce, (uint2*)g_hash);
	else
		lyra2v2_gpu_hash_32_v3 <<<grid, block>>> (threads, startNounce, (uint2*)g_hash);

	//MyStreamSynchronize(NULL, order, thr_id);
}
