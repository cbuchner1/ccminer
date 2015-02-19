/*
 * lyra2 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  djm34
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   djm34
 */
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>


extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern int compute_version[8];

#include "cuda_helper.h"


static __constant__ uint2 blake2b_IV[8] =
{
	{ 0xf3bcc908, 0x6a09e667  }, 
	{ 0x84caa73b, 0xbb67ae85  },
	{ 0xfe94f82b, 0x3c6ef372  },
	{ 0x5f1d36f1, 0xa54ff53a  },
	{ 0xade682d1, 0x510e527f  },
	{ 0x2b3e6c1f, 0x9b05688c  },
	{ 0xfb41bd6b, 0x1f83d9ab  },
	{ 0x137e2179, 0x5be0cd19  }
};

#define reduceDuplexRowSetup(rowIn, rowInOut, rowOut) \
  { \
	for (int i = 0; i < 8; i++) \
			{ \
\
		for (int j = 0; j < 12; j++) {state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut];} \
		round_lyra_v35(state); \
		for (int j = 0; j < 12; j++) {Matrix[j + 84 - 12 * i][rowOut] = Matrix[12 * i + j][rowIn] ^ state[j];} \
\
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
		Matrix[10 + 12 * i][rowInOut] ^= state[9]; \
		Matrix[11 + 12 * i][rowInOut] ^= state[10]; \
			} \
 \
  } 

#define reduceDuplexRow(rowIn, rowInOut, rowOut) \
  { \
	 for (int i = 0; i < 8; i++) \
	 	 	 	 { \
		 for (int j = 0; j < 12; j++) \
			 state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut]; \
 \
		 round_lyra_v35(state); \
		 for (int j = 0; j < 12; j++) {Matrix[j + 12 * i][rowOut] ^= state[j];} \
\
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
		 Matrix[10 + 12 * i][rowInOut] ^= state[9]; \
		 Matrix[11 + 12 * i][rowInOut] ^= state[10]; \
	 	 	 	 } \
 \
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
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
	round_lyra_v35(state); \
  } 

//// compute 30 version 
#define reduceDuplexRowSetup_v30(rowIn, rowInOut, rowOut) \
  { \
	for (int i = 0; i < 8; i++) \
				{ \
\
		for (int j = 0; j < 12; j++) {state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut];} \
		round_lyra_v30(state); \
		for (int j = 0; j < 12; j++) {Matrix[j + 84 - 12 * i][rowOut] = Matrix[12 * i + j][rowIn] ^ state[j];} \
\
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
		Matrix[10 + 12 * i][rowInOut] ^= state[9]; \
		Matrix[11 + 12 * i][rowInOut] ^= state[10]; \
				} \
 \
  } 

#define reduceDuplexRow_v30(rowIn, rowInOut, rowOut) \
  { \
	 for (int i = 0; i < 8; i++) \
	 	 	 	 	 { \
		 for (int j = 0; j < 12; j++) \
			 state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut]; \
 \
		 round_lyra_v30(state); \
		 for (int j = 0; j < 12; j++) {Matrix[j + 12 * i][rowOut] ^= state[j];} \
\
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
		 Matrix[10 + 12 * i][rowInOut] ^= state[9]; \
		 Matrix[11 + 12 * i][rowInOut] ^= state[10]; \
	 	 	 	 	 } \
 \
  } 
#define absorbblock_v30(in)  { \
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
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
	round_lyra_v30(state); \
  } 




 static __device__ __forceinline__ void Gfunc_v35(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
 {
	 a += b; d ^= a; d = ROR2(d, 32);
	 c += d; b ^= c; b = ROR2(b, 24);
	 a += b; d ^= a; d = ROR2(d, 16);
	 c += d; b ^= c; b = ROR2(b, 63);
 }


 static __device__ __forceinline__ void Gfunc_v30(uint64_t & a, uint64_t &b, uint64_t &c, uint64_t &d)
 {
	 a += b; d ^= a; d = ROTR64(d, 32);
	 c += d; b ^= c; b = ROTR64(b, 24);
	 a += b; d ^= a; d = ROTR64(d, 16);
	 c += d; b ^= c; b = ROTR64(b, 63);
 }

 
static __device__ __forceinline__ void round_lyra_v35(uint2 *s) 
{
	Gfunc_v35(s[0], s[4], s[8],  s[12]);
	Gfunc_v35(s[1], s[5], s[9],  s[13]);
	Gfunc_v35(s[2], s[6], s[10], s[14]);
	Gfunc_v35(s[3], s[7], s[11], s[15]);
	Gfunc_v35(s[0], s[5], s[10], s[15]);
	Gfunc_v35(s[1], s[6], s[11], s[12]);
	Gfunc_v35(s[2], s[7], s[8],  s[13]);
	Gfunc_v35(s[3], s[4], s[9],  s[14]);
}

static __device__ __forceinline__ void round_lyra_v30(uint64_t *s)
{
	Gfunc_v30(s[0], s[4], s[8], s[12]);
	Gfunc_v30(s[1], s[5], s[9], s[13]);
	Gfunc_v30(s[2], s[6], s[10], s[14]);
	Gfunc_v30(s[3], s[7], s[11], s[15]);
	Gfunc_v30(s[0], s[5], s[10], s[15]);
	Gfunc_v30(s[1], s[6], s[11], s[12]);
	Gfunc_v30(s[2], s[7], s[8], s[13]);
	Gfunc_v30(s[3], s[4], s[9], s[14]);
}



__global__ void __launch_bounds__(160, 1) lyra2_gpu_hash_32_v30(int threads, uint32_t startNounce, uint64_t *outputHash)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t state[16];
#pragma unroll
		for (int i = 0; i<4; i++) { state[i] = outputHash[threads*i + thread]; } //password
#pragma unroll
		for (int i = 0; i<4; i++) { state[i + 4] = state[i]; } //salt 
#pragma unroll
		for (int i = 0; i<8; i++) { state[i + 8] = devectorize(blake2b_IV[i]); }

		//     blake2blyra x2 
#pragma unroll 24
		for (int i = 0; i<24; i++) { round_lyra_v30(state); } //because 12 is not enough

		uint64_t Matrix[96][8]; // not cool
		/// reducedSqueezeRow0
#pragma unroll 8 
		for (int i = 0; i < 8; i++)
		{
int idx = 84-12*i;
#pragma unroll 12
			for (int j = 0; j<12; j++) { Matrix[j + idx][0] = state[j]; }
			round_lyra_v30(state);
		}

		/// reducedSqueezeRow1
#pragma unroll 8 
		for (int i = 0; i < 8; i++)
		{
int idx0= 12*i;
int idx1= 84-idx0; 
#pragma unroll 12
			for (int j = 0; j<12; j++) { state[j] ^= Matrix[j + idx0][0]; }
			round_lyra_v30(state);
#pragma unroll 12  
			for (int j = 0; j<12; j++) { Matrix[j + idx1][1] = Matrix[j + idx0][0] ^ state[j]; }
		}


		reduceDuplexRowSetup_v30(1, 0, 2);
		reduceDuplexRowSetup_v30(2, 1, 3);
		reduceDuplexRowSetup_v30(3, 0, 4);
		reduceDuplexRowSetup_v30(4, 3, 5);
		reduceDuplexRowSetup_v30(5, 2, 6);
		reduceDuplexRowSetup_v30(6, 1, 7);



		uint64_t rowa;
		rowa = state[0] & 7;
		reduceDuplexRow_v30(7, rowa, 0);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(0, rowa, 3);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(3, rowa, 6);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(6, rowa, 1);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(1, rowa, 4);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(4, rowa, 7);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(7, rowa, 2);
		rowa = state[0] & 7;
		reduceDuplexRow_v30(2, rowa, 5);

		absorbblock_v30(rowa);


#pragma unroll
		for (int i = 0; i<4; i++) {
			outputHash[threads*i + thread] = state[i];
		} //password


	} //thread
}


__global__ void __launch_bounds__(160, 1) lyra2_gpu_hash_32(int threads, uint32_t startNounce, uint64_t *outputHash)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 state[16];
#pragma unroll
		for (int i = 0; i<4; i++) { LOHI(state[i].x, state[i].y, outputHash[threads*i + thread]); } //password
#pragma unroll
		for (int i = 0; i<4; i++) { state[i + 4] = state[i]; } //salt 
#pragma unroll
		for (int i = 0; i<8; i++) { state[i + 8] = blake2b_IV[i]; }

		//     blake2blyra x2 
#pragma unroll 24
		for (int i = 0; i<24; i++) { round_lyra_v35(state); } //because 12 is not enough

		uint2 Matrix[96][8]; // not cool

		/// reducedSqueezeRow0
#pragma unroll 8 
		for (int i = 0; i < 8; i++)
		{
#pragma unroll 12
			for (int j = 0; j<12; j++) { Matrix[j + 84 - 12 * i][0] = state[j]; }
			round_lyra_v35(state);
		}

		/// reducedSqueezeRow1
#pragma unroll 8 
		for (int i = 0; i < 8; i++)
		{
#pragma unroll 12
			for (int j = 0; j<12; j++) { state[j] ^= Matrix[j + 12 * i][0]; }
			round_lyra_v35(state);
#pragma unroll 12  
			for (int j = 0; j<12; j++) { Matrix[j + 84 - 12 * i][1] = Matrix[j + 12 * i][0] ^ state[j]; }
		}

		reduceDuplexRowSetup(1, 0, 2);
		reduceDuplexRowSetup(2, 1, 3);
		reduceDuplexRowSetup(3, 0, 4);
		reduceDuplexRowSetup(4, 3, 5);
		reduceDuplexRowSetup(5, 2, 6);
		reduceDuplexRowSetup(6, 1, 7);



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
			outputHash[threads*i + thread] = devectorize(state[i]);
		} //password


	} //thread
}

   
void lyra2_cpu_init(int thr_id, int threads)
{
//not used    	
} 


__host__ void lyra2_cpu_hash_32(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	
	const int threadsperblock = 160;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;
	
	if (compute_version[thr_id]>=35) {
	lyra2_gpu_hash_32 << <grid, block, shared_size >> >(threads, startNounce, d_outputHash);
	}
	else {  // kernel for compute30 card
	lyra2_gpu_hash_32_v30 << <grid, block, shared_size >> >(threads, startNounce, d_outputHash);
	}
    
	MyStreamSynchronize(NULL, order, thr_id);

}

