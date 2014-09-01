/*
 * ripemd-160 djm34
 * 
 */

/*
 * ripemd-160 kernel implementation.
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
 * @author   phm <phm@inbox.com>
 */
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>



#include "cuda_helper.h"

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))
#define ROTL    SPH_ROTL32

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);


 __constant__ uint32_t c_PaddedMessage80[32]; // padded message (80 bytes + padding)
static __constant__ uint32_t gpu_IV[5];
static __constant__ uint32_t bufo[5];
static const uint32_t IV[5] = {
	SPH_C32(0x67452301), SPH_C32(0xEFCDAB89), SPH_C32(0x98BADCFE),
	SPH_C32(0x10325476), SPH_C32(0xC3D2E1F0)
};

/*
 * Round functions for RIPEMD-128 and RIPEMD-160.
 */
#define F1(x, y, z)   ((x) ^ (y) ^ (z))
#define F2(x, y, z)   ((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z)   (((x) | ~(y)) ^ (z))
#define F4(x, y, z)   ((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z)   ((x) ^ ((y) | ~(z)))

/*
 * Round constants for RIPEMD-160.
 */
#define K11    SPH_C32(0x00000000)
#define K12    SPH_C32(0x5A827999)
#define K13    SPH_C32(0x6ED9EBA1)
#define K14    SPH_C32(0x8F1BBCDC)
#define K15    SPH_C32(0xA953FD4E)

#define K21    SPH_C32(0x50A28BE6)
#define K22    SPH_C32(0x5C4DD124)
#define K23    SPH_C32(0x6D703EF3)
#define K24    SPH_C32(0x7A6D76E9)
#define K25    SPH_C32(0x00000000)

#define RR(a, b, c, d, e, f, s, r, k)    { \
		a = SPH_T32(ROTL(SPH_T32(a + f(b, c, d) + r + k), s) + e); \
		c = ROTL(c, 10); \
	} 

#define ROUND1(a, b, c, d, e, f, s, r, k)  \
	RR(a ## 1, b ## 1, c ## 1, d ## 1, e ## 1, f, s, r, K1 ## k)

#define ROUND2(a, b, c, d, e, f, s, r, k)  \
	RR(a ## 2, b ## 2, c ## 2, d ## 2, e ## 2, f, s, r, K2 ## k)



#define RIPEMD160_ROUND_BODY(in, h)   { \
		uint32_t A1, B1, C1, D1, E1; \
		uint32_t A2, B2, C2, D2, E2; \
		uint32_t tmp; \
 \
		A1 = A2 = (h)[0]; \
		B1 = B2 = (h)[1]; \
		C1 = C2 = (h)[2]; \
		D1 = D2 = (h)[3]; \
		E1 = E2 = (h)[4]; \
 \
		ROUND1(A, B, C, D, E, F1, 11, in[ 0],  1); \
		ROUND1(E, A, B, C, D, F1, 14, in[ 1],  1); \
		ROUND1(D, E, A, B, C, F1, 15, in[ 2],  1); \
		ROUND1(C, D, E, A, B, F1, 12, in[ 3],  1); \
		ROUND1(B, C, D, E, A, F1,  5, in[ 4],  1); \
		ROUND1(A, B, C, D, E, F1,  8, in[ 5],  1); \
		ROUND1(E, A, B, C, D, F1,  7, in[ 6],  1); \
		ROUND1(D, E, A, B, C, F1,  9, in[ 7],  1); \
		ROUND1(C, D, E, A, B, F1, 11, in[ 8],  1); \
		ROUND1(B, C, D, E, A, F1, 13, in[ 9],  1); \
		ROUND1(A, B, C, D, E, F1, 14, in[10],  1); \
		ROUND1(E, A, B, C, D, F1, 15, in[11],  1); \
		ROUND1(D, E, A, B, C, F1,  6, in[12],  1); \
		ROUND1(C, D, E, A, B, F1,  7, in[13],  1); \
		ROUND1(B, C, D, E, A, F1,  9, in[14],  1); \
		ROUND1(A, B, C, D, E, F1,  8, in[15],  1); \
 \
		ROUND1(E, A, B, C, D, F2,  7, in[ 7],  2); \
		ROUND1(D, E, A, B, C, F2,  6, in[ 4],  2); \
		ROUND1(C, D, E, A, B, F2,  8, in[13],  2); \
		ROUND1(B, C, D, E, A, F2, 13, in[ 1],  2); \
		ROUND1(A, B, C, D, E, F2, 11, in[10],  2); \
		ROUND1(E, A, B, C, D, F2,  9, in[ 6],  2); \
		ROUND1(D, E, A, B, C, F2,  7, in[15],  2); \
		ROUND1(C, D, E, A, B, F2, 15, in[ 3],  2); \
		ROUND1(B, C, D, E, A, F2,  7, in[12],  2); \
		ROUND1(A, B, C, D, E, F2, 12, in[ 0],  2); \
		ROUND1(E, A, B, C, D, F2, 15, in[ 9],  2); \
		ROUND1(D, E, A, B, C, F2,  9, in[ 5],  2); \
		ROUND1(C, D, E, A, B, F2, 11, in[ 2],  2); \
		ROUND1(B, C, D, E, A, F2,  7, in[14],  2); \
		ROUND1(A, B, C, D, E, F2, 13, in[11],  2); \
		ROUND1(E, A, B, C, D, F2, 12, in[ 8],  2); \
 \
		ROUND1(D, E, A, B, C, F3, 11, in[ 3],  3); \
		ROUND1(C, D, E, A, B, F3, 13, in[10],  3); \
		ROUND1(B, C, D, E, A, F3,  6, in[14],  3); \
		ROUND1(A, B, C, D, E, F3,  7, in[ 4],  3); \
		ROUND1(E, A, B, C, D, F3, 14, in[ 9],  3); \
		ROUND1(D, E, A, B, C, F3,  9, in[15],  3); \
		ROUND1(C, D, E, A, B, F3, 13, in[ 8],  3); \
		ROUND1(B, C, D, E, A, F3, 15, in[ 1],  3); \
		ROUND1(A, B, C, D, E, F3, 14, in[ 2],  3); \
		ROUND1(E, A, B, C, D, F3,  8, in[ 7],  3); \
		ROUND1(D, E, A, B, C, F3, 13, in[ 0],  3); \
		ROUND1(C, D, E, A, B, F3,  6, in[ 6],  3); \
		ROUND1(B, C, D, E, A, F3,  5, in[13],  3); \
		ROUND1(A, B, C, D, E, F3, 12, in[11],  3); \
		ROUND1(E, A, B, C, D, F3,  7, in[ 5],  3); \
		ROUND1(D, E, A, B, C, F3,  5, in[12],  3); \
 \
		ROUND1(C, D, E, A, B, F4, 11, in[ 1],  4); \
		ROUND1(B, C, D, E, A, F4, 12, in[ 9],  4); \
		ROUND1(A, B, C, D, E, F4, 14, in[11],  4); \
		ROUND1(E, A, B, C, D, F4, 15, in[10],  4); \
		ROUND1(D, E, A, B, C, F4, 14, in[ 0],  4); \
		ROUND1(C, D, E, A, B, F4, 15, in[ 8],  4); \
		ROUND1(B, C, D, E, A, F4,  9, in[12],  4); \
		ROUND1(A, B, C, D, E, F4,  8, in[ 4],  4); \
		ROUND1(E, A, B, C, D, F4,  9, in[13],  4); \
		ROUND1(D, E, A, B, C, F4, 14, in[ 3],  4); \
		ROUND1(C, D, E, A, B, F4,  5, in[ 7],  4); \
		ROUND1(B, C, D, E, A, F4,  6, in[15],  4); \
		ROUND1(A, B, C, D, E, F4,  8, in[14],  4); \
		ROUND1(E, A, B, C, D, F4,  6, in[ 5],  4); \
		ROUND1(D, E, A, B, C, F4,  5, in[ 6],  4); \
		ROUND1(C, D, E, A, B, F4, 12, in[ 2],  4); \
 \
		ROUND1(B, C, D, E, A, F5,  9, in[ 4],  5); \
		ROUND1(A, B, C, D, E, F5, 15, in[ 0],  5); \
		ROUND1(E, A, B, C, D, F5,  5, in[ 5],  5); \
		ROUND1(D, E, A, B, C, F5, 11, in[ 9],  5); \
		ROUND1(C, D, E, A, B, F5,  6, in[ 7],  5); \
		ROUND1(B, C, D, E, A, F5,  8, in[12],  5); \
		ROUND1(A, B, C, D, E, F5, 13, in[ 2],  5); \
		ROUND1(E, A, B, C, D, F5, 12, in[10],  5); \
		ROUND1(D, E, A, B, C, F5,  5, in[14],  5); \
		ROUND1(C, D, E, A, B, F5, 12, in[ 1],  5); \
		ROUND1(B, C, D, E, A, F5, 13, in[ 3],  5); \
		ROUND1(A, B, C, D, E, F5, 14, in[ 8],  5); \
		ROUND1(E, A, B, C, D, F5, 11, in[11],  5); \
		ROUND1(D, E, A, B, C, F5,  8, in[ 6],  5); \
		ROUND1(C, D, E, A, B, F5,  5, in[15],  5); \
		ROUND1(B, C, D, E, A, F5,  6, in[13],  5); \
 \
		ROUND2(A, B, C, D, E, F5,  8, in[ 5],  1); \
		ROUND2(E, A, B, C, D, F5,  9, in[14],  1); \
		ROUND2(D, E, A, B, C, F5,  9, in[ 7],  1); \
		ROUND2(C, D, E, A, B, F5, 11, in[ 0],  1); \
		ROUND2(B, C, D, E, A, F5, 13, in[ 9],  1); \
		ROUND2(A, B, C, D, E, F5, 15, in[ 2],  1); \
		ROUND2(E, A, B, C, D, F5, 15, in[11],  1); \
		ROUND2(D, E, A, B, C, F5,  5, in[ 4],  1); \
		ROUND2(C, D, E, A, B, F5,  7, in[13],  1); \
		ROUND2(B, C, D, E, A, F5,  7, in[ 6],  1); \
		ROUND2(A, B, C, D, E, F5,  8, in[15],  1); \
		ROUND2(E, A, B, C, D, F5, 11, in[ 8],  1); \
		ROUND2(D, E, A, B, C, F5, 14, in[ 1],  1); \
		ROUND2(C, D, E, A, B, F5, 14, in[10],  1); \
		ROUND2(B, C, D, E, A, F5, 12, in[ 3],  1); \
		ROUND2(A, B, C, D, E, F5,  6, in[12],  1); \
 \
		ROUND2(E, A, B, C, D, F4,  9, in[ 6],  2); \
		ROUND2(D, E, A, B, C, F4, 13, in[11],  2); \
		ROUND2(C, D, E, A, B, F4, 15, in[ 3],  2); \
		ROUND2(B, C, D, E, A, F4,  7, in[ 7],  2); \
		ROUND2(A, B, C, D, E, F4, 12, in[ 0],  2); \
		ROUND2(E, A, B, C, D, F4,  8, in[13],  2); \
		ROUND2(D, E, A, B, C, F4,  9, in[ 5],  2); \
		ROUND2(C, D, E, A, B, F4, 11, in[10],  2); \
		ROUND2(B, C, D, E, A, F4,  7, in[14],  2); \
		ROUND2(A, B, C, D, E, F4,  7, in[15],  2); \
		ROUND2(E, A, B, C, D, F4, 12, in[ 8],  2); \
		ROUND2(D, E, A, B, C, F4,  7, in[12],  2); \
		ROUND2(C, D, E, A, B, F4,  6, in[ 4],  2); \
		ROUND2(B, C, D, E, A, F4, 15, in[ 9],  2); \
		ROUND2(A, B, C, D, E, F4, 13, in[ 1],  2); \
		ROUND2(E, A, B, C, D, F4, 11, in[ 2],  2); \
 \
		ROUND2(D, E, A, B, C, F3,  9, in[15],  3); \
		ROUND2(C, D, E, A, B, F3,  7, in[ 5],  3); \
		ROUND2(B, C, D, E, A, F3, 15, in[ 1],  3); \
		ROUND2(A, B, C, D, E, F3, 11, in[ 3],  3); \
		ROUND2(E, A, B, C, D, F3,  8, in[ 7],  3); \
		ROUND2(D, E, A, B, C, F3,  6, in[14],  3); \
		ROUND2(C, D, E, A, B, F3,  6, in[ 6],  3); \
		ROUND2(B, C, D, E, A, F3, 14, in[ 9],  3); \
		ROUND2(A, B, C, D, E, F3, 12, in[11],  3); \
		ROUND2(E, A, B, C, D, F3, 13, in[ 8],  3); \
		ROUND2(D, E, A, B, C, F3,  5, in[12],  3); \
		ROUND2(C, D, E, A, B, F3, 14, in[ 2],  3); \
		ROUND2(B, C, D, E, A, F3, 13, in[10],  3); \
		ROUND2(A, B, C, D, E, F3, 13, in[ 0],  3); \
		ROUND2(E, A, B, C, D, F3,  7, in[ 4],  3); \
		ROUND2(D, E, A, B, C, F3,  5, in[13],  3); \
 \
		ROUND2(C, D, E, A, B, F2, 15, in[ 8],  4); \
		ROUND2(B, C, D, E, A, F2,  5, in[ 6],  4); \
		ROUND2(A, B, C, D, E, F2,  8, in[ 4],  4); \
		ROUND2(E, A, B, C, D, F2, 11, in[ 1],  4); \
		ROUND2(D, E, A, B, C, F2, 14, in[ 3],  4); \
		ROUND2(C, D, E, A, B, F2, 14, in[11],  4); \
		ROUND2(B, C, D, E, A, F2,  6, in[15],  4); \
		ROUND2(A, B, C, D, E, F2, 14, in[ 0],  4); \
		ROUND2(E, A, B, C, D, F2,  6, in[ 5],  4); \
		ROUND2(D, E, A, B, C, F2,  9, in[12],  4); \
		ROUND2(C, D, E, A, B, F2, 12, in[ 2],  4); \
		ROUND2(B, C, D, E, A, F2,  9, in[13],  4); \
		ROUND2(A, B, C, D, E, F2, 12, in[ 9],  4); \
		ROUND2(E, A, B, C, D, F2,  5, in[ 7],  4); \
		ROUND2(D, E, A, B, C, F2, 15, in[10],  4); \
		ROUND2(C, D, E, A, B, F2,  8, in[14],  4); \
 \
		ROUND2(B, C, D, E, A, F1,  8, in[12],  5); \
		ROUND2(A, B, C, D, E, F1,  5, in[15],  5); \
		ROUND2(E, A, B, C, D, F1, 12, in[10],  5); \
		ROUND2(D, E, A, B, C, F1,  9, in[ 4],  5); \
		ROUND2(C, D, E, A, B, F1, 12, in[ 1],  5); \
		ROUND2(B, C, D, E, A, F1,  5, in[ 5],  5); \
		ROUND2(A, B, C, D, E, F1, 14, in[ 8],  5); \
		ROUND2(E, A, B, C, D, F1,  6, in[ 7],  5); \
		ROUND2(D, E, A, B, C, F1,  8, in[ 6],  5); \
		ROUND2(C, D, E, A, B, F1, 13, in[ 2],  5); \
		ROUND2(B, C, D, E, A, F1,  6, in[13],  5); \
		ROUND2(A, B, C, D, E, F1,  5, in[14],  5); \
		ROUND2(E, A, B, C, D, F1, 15, in[ 0],  5); \
		ROUND2(D, E, A, B, C, F1, 13, in[ 3],  5); \
		ROUND2(C, D, E, A, B, F1, 11, in[ 9],  5); \
		ROUND2(B, C, D, E, A, F1, 11, in[11],  5); \
 \
		tmp = SPH_T32((h)[1] + C1 + D2); \
		(h)[1] = SPH_T32((h)[2] + D1 + E2); \
		(h)[2] = SPH_T32((h)[3] + E1 + A2); \
		(h)[3] = SPH_T32((h)[4] + A1 + B2); \
		(h)[4] = SPH_T32((h)[0] + B1 + C2); \
		(h)[0] = tmp; \
	} 


__global__ void m7_ripemd160_gpu_hash_120(int threads, uint32_t startNounce, uint64_t *outputHash)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        
        uint32_t nounce = startNounce + thread ;
union {
uint8_t h1[64];
uint32_t h4[16];
uint64_t h8[8];
} hash;  

#undef F1
#undef F2
#undef F3
#undef F4
#undef F5

#define F1(x, y, z)   xor3(x,y,z)
#define F2(x, y, z)   xandx(x,y,z)
#define F3(x, y, z)   xornot64(x,y,z)
#define F4(x, y, z)   xandx(z,x,y)
#define F5(x, y, z)   xornt64(x,y,z)
        uint32_t in2[16],in3[16];
        uint32_t in[16],buf[5]; 
        #pragma unroll 16
        for (int i=0;i<16;i++) {if ((i+16)<29)  {in2[i]= c_PaddedMessage80[i+16];} 
						   else if ((i+16)==29) {in2[i]= nounce;}
						   else if ((i+16)==30) {in2[i]= c_PaddedMessage80[i+16];}
						   else                 {in2[i]= 0;}}
		#pragma unroll 16
		for (int i=0;i<16;i++) {in3[i]=0;}
		                        in3[14]=0x3d0;
         #pragma unroll 5
		 for (int i=0;i<5;i++) {buf[i]=bufo[i];}
		 RIPEMD160_ROUND_BODY(in2, buf);		 
         RIPEMD160_ROUND_BODY(in3, buf);

  
hash.h4[5]=0; 
#pragma unroll 5
for (int i=0;i<5;i++) 
{hash.h4[i]=buf[i];
}

#pragma unroll 3
for (int i=0;i<3;i++) {outputHash[i*threads+thread]=hash.h8[i];}

 }
}


void ripemd160_cpu_init(int thr_id, int threads)
{

    cudaMemcpyToSymbol(gpu_IV,IV,sizeof(IV),0, cudaMemcpyHostToDevice);
	
}

__host__ void ripemd160_setBlock_120(void *pdata)
{
	unsigned char PaddedMessage[128];
	uint8_t ending =0x80;
	memcpy(PaddedMessage, pdata, 122);
	memset(PaddedMessage+122,ending,1); 
	memset(PaddedMessage+123, 0, 5); //useless
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 32*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);

#undef F1
#undef F2
#undef F3
#undef F4
#undef F5
#define F1(x, y, z)   ((x) ^ (y) ^ (z))
#define F2(x, y, z)   ((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z)   (((x) | ~(y)) ^ (z))
#define F4(x, y, z)   ((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z)   ((x) ^ ((y) | ~(z)))	
	uint32_t* alt_data =(uint32_t*)pdata;
        uint32_t in[16],buf[5];

	    
		for (int i=0;i<16;i++) {in[i]= alt_data[i];}
        
		
		for (int i=0;i<5;i++) {buf[i]=IV[i];}
		
		 RIPEMD160_ROUND_BODY(in, buf); //no need to calculate it several time (need to moved)
	cudaMemcpyToSymbol(bufo, buf, 5*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__ void m7_ripemd160_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{

	const int threadsperblock = 256; // Alignment mit mixtab Grösse. NICHT ÄNDERN


dim3 grid(threads/threadsperblock);
dim3 block(threadsperblock);
//dim3 grid(1);
//dim3 block(1);
	size_t shared_size =0;
	m7_ripemd160_gpu_hash_120<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}
