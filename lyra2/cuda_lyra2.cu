/**
 * Lyra2 (v1) cuda implementation based on djm34 work - SM 5/5.2
 * tpruvot@github 2015
 */

#include <stdio.h>
#include <memory.h>

#define TPB50 16
#define TPB52 8

#include "cuda_lyra2_sm2.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ >= 500

#include "cuda_vector_uint2x4.h"

#define memshift 3

#define Ncol 8
#define NcolMask 0x7

__device__ uint2x4* DMatrix;

static __device__ __forceinline__
void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__
void reduceDuplex(uint2x4 state[4], uint32_t thread)
{
	uint2x4 state1[3];

	const uint32_t ps1 = (256 * thread);
	const uint32_t ps2 = (memshift * 7 + memshift * 8 + 256 * thread);

	#pragma unroll 4
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 - i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix+s1)[j]);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state1[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2x4 state[4], uint32_t thread)
{
	uint2x4 state1[3], state2[3];

	const uint32_t ps1 = (             memshift*8 * rowIn    + 256 * thread);
	const uint32_t ps2 = (             memshift*8 * rowInOut + 256 * thread);
	const uint32_t ps3 = (memshift*7 + memshift*8 * rowOut   + 256 * thread);

	#pragma unroll 1
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		for (int j = 0; j < 3; j++)
			state1[j]= __ldg4(&(DMatrix + s1)[j]);
		for (int j = 0; j < 3; j++)
			state2[j]= __ldg4(&(DMatrix + s2)[j]);
		for (int j = 0; j < 3; j++) {
			uint2x4 tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}

		round_lyra(state);

		for (int j = 0; j < 3; j++) {
			const uint32_t s3 = ps3 - i*memshift;
			state1[j] ^= state[j];
			(DMatrix + s3)[j] = state1[j];
		}

		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j+1] ^= ((uint2*)state)[j];

		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2x4* state, const uint32_t thread)
{
	const uint32_t ps1 = (memshift * 8 * rowIn    + 256 * thread);
	const uint32_t ps2 = (memshift * 8 * rowInOut + 256 * thread);
	const uint32_t ps3 = (memshift * 8 * rowOut   + 256 * thread);

	#pragma unroll 1
	for (int i = 0; i < 8; i++)
	{
		uint2x4 state1[3], state2[3];

		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;

		for (int j = 0; j < 3; j++) {
			state1[j] = __ldg4(&(DMatrix + s1)[j]);
			state2[j] = __ldg4(&(DMatrix + s2)[j]);
		}

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			state1[j] += state2[j];
			state[j]  ^= state1[j];
		}

		round_lyra(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];

		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

		if (rowInOut == rowOut) {
			for (int j = 0; j < 3; j++) {
				state2[j] ^= state[j];
				(DMatrix + s2)[j]=state2[j];
			}
		} else {
			const uint32_t s3 = ps3 + i*memshift;
			for (int j = 0; j < 3; j++) {
				(DMatrix + s2)[j] = state2[j];
				(DMatrix + s3)[j] ^= state[j];
			}
		}
	}
}

#if __CUDA_ARCH__ == 500
__global__ __launch_bounds__(TPB50, 1)
#else
__global__ __launch_bounds__(TPB52, 2)
#endif
void lyra2_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	const uint2x4 blake2b_IV[2] = {
		{{ 0xf3bcc908, 0x6a09e667 }, { 0x84caa73b, 0xbb67ae85 }, { 0xfe94f82b, 0x3c6ef372 }, { 0x5f1d36f1, 0xa54ff53a }},
		{{ 0xade682d1, 0x510e527f }, { 0x2b3e6c1f, 0x9b05688c }, { 0xfb41bd6b, 0x1f83d9ab }, { 0x137e2179, 0x5be0cd19 }}
	};

	if (thread < threads)
	{
		uint2x4 state[4];

		((uint2*)state)[0] = __ldg(&g_hash[thread]);
		((uint2*)state)[1] = __ldg(&g_hash[thread + threads]);
		((uint2*)state)[2] = __ldg(&g_hash[thread + threads*2]);
		((uint2*)state)[3] = __ldg(&g_hash[thread + threads*3]);

		state[1] = state[0];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state); //because 12 is not enough

		const uint32_t ps1 = (memshift * 7  + 256 * thread);
		for (int i = 0; i < 8; i++)
		{
			const uint32_t s1 = ps1 - memshift * i;
			for (int j = 0; j < 3; j++)
				(DMatrix + s1)[j] = (state)[j];
			round_lyra(state);
		}

		reduceDuplex(state, thread);

		reduceDuplexRowSetup(1, 0, 2, state,  thread);
		reduceDuplexRowSetup(2, 1, 3, state,  thread);
		reduceDuplexRowSetup(3, 0, 4, state,  thread);
		reduceDuplexRowSetup(4, 3, 5, state,  thread);
		reduceDuplexRowSetup(5, 2, 6, state,  thread);
		reduceDuplexRowSetup(6, 1, 7, state,  thread);

		uint32_t rowa = state[0].x.x & 7;
		reduceDuplexRowt(7, rowa, 0, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(0, rowa, 3, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(3, rowa, 6, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(6, rowa, 1, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(1, rowa, 4, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(4, rowa, 7, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(7, rowa, 2, state, thread);
		rowa = state[0].x.x & 7;
		reduceDuplexRowt(2, rowa, 5, state, thread);

		const int32_t shift = (memshift * 8 * rowa + 256 * thread);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		g_hash[thread]             = ((uint2*)state)[0];
		g_hash[thread + threads]   = ((uint2*)state)[1];
		g_hash[thread + threads*2] = ((uint2*)state)[2];
		g_hash[thread + threads*3] = ((uint2*)state)[3];
	}
}
#else
/* for unsupported SM arch */
__device__ void* DMatrix;
__global__ void lyra2_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
#endif

__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t* d_matrix)
{
	cuda_get_arch(thr_id);
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order)
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	uint32_t tpb = TPB52;
	if (device_sm[dev_id] == 500) tpb = TPB50;
	if (device_sm[dev_id] == 350) tpb = TPB30; // to enhance (or not)
	if (device_sm[dev_id] <= 300) tpb = TPB30;

	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	if (device_sm[dev_id] >= 500)
		lyra2_gpu_hash_32 <<< grid, block >>> (threads, startNounce, (uint2*)d_hash);
	else
		lyra2_gpu_hash_32_sm2 <<< grid, block >>> (threads, startNounce, d_hash);

}
