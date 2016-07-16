/*
 * ripemd-160 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014, 2016  djm34, tpruvot
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
 */
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>

static __constant__ uint32_t c_IV[5] = {
	0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
};

__device__ __forceinline__
uint32_t xor3b(const uint32_t a, const uint32_t b, const uint32_t c) {
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0x96; // xor3b"  //0x96 = 0xF0 ^ 0xCC ^ 0xAA
		: "=r"(result) : "r"(a), "r"(b),"r"(c));
#else
	result = a^b^c;
#endif
	return result;
}

//__host__
//uint64_t xornot64(uint64_t a, uint64_t b, uint64_t c) {
//	return c ^ (a | !b);
//}

__forceinline__ __device__
uint64_t xornot64(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{  .reg .u64 m,n; // xornot64\n\t"
		"not.b64 m,%2; \n\t"
		"or.b64 n, %1,m;\n\t"
		"xor.b64 %0, n,%3;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

//__host__
//uint64_t xornt64(uint64_t a, uint64_t b, uint64_t c) {
//	return a ^ (b | !c);
//}

__device__ __forceinline__
uint64_t xornt64(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{  .reg .u64 m,n; // xornt64\n\t"
		"not.b64 m,%3; \n\t"
		"or.b64 n, %2,m;\n\t"
		"xor.b64 %0, %1,n;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

/*
 * Round functions for RIPEMD-128 and RIPEMD-160.
 */
#if 1
#define F1(x, y, z)   ((x) ^ (y) ^ (z))
#define F2(x, y, z)   ((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z)   (((x) | ~(y)) ^ (z))
#define F4(x, y, z)   ((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z)   ((x) ^ ((y) | ~(z)))
#else
#define F1(x, y, z)   xor3b(x,y,z)
#define F2(x, y, z)   xandx(x,y,z)
#define F3(x, y, z)   xornot64(x,y,z)
#define F4(x, y, z)   xandx(z,x,y)
#define F5(x, y, z)   xornt64(x,y,z)
#endif

/*
 * Round constants for RIPEMD-160.
 */
#define K11 0x00000000u
#define K12 0x5A827999u
#define K13 0x6ED9EBA1u
#define K14 0x8F1BBCDCu
#define K15 0xA953FD4Eu

#define K21 0x50A28BE6u
#define K22 0x5C4DD124u
#define K23 0x6D703EF3u
#define K24 0x7A6D76E9u
#define K25 0x00000000u

#define RR(a, b, c, d, e, f, s, r, k) { \
	a = SPH_T32(ROTL32(SPH_T32(a + f(b, c, d) + r + k), s) + e); \
	c = ROTL32(c, 10); \
}

#define ROUND1(a, b, c, d, e, f, s, r, k) \
	RR(a ## 1, b ## 1, c ## 1, d ## 1, e ## 1, f, s, r, K1 ## k)

#define ROUND2(a, b, c, d, e, f, s, r, k) \
	RR(a ## 2, b ## 2, c ## 2, d ## 2, e ## 2, f, s, r, K2 ## k)

#define RIPEMD160_ROUND_BODY(in, h) { \
	uint32_t A1, B1, C1, D1, E1; \
	uint32_t A2, B2, C2, D2, E2; \
	uint32_t tmp; \
\
	A1 = A2 = h[0]; \
	B1 = B2 = h[1]; \
	C1 = C2 = h[2]; \
	D1 = D2 = h[3]; \
	E1 = E2 = h[4]; \
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
	tmp  = (h[1] + C1 + D2); \
	h[1] = (h[2] + D1 + E2); \
	h[2] = (h[3] + E1 + A2); \
	h[3] = (h[4] + A1 + B2); \
	h[4] = (h[0] + B1 + C2); \
	h[0] = tmp; \
}

__global__
void lbry_ripemd160_gpu_hash_32x2(const uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *hash = (uint32_t*) (&g_hash[thread * 8U]);

		uint32_t in[16];
		#pragma unroll
		for (int i=0; i<8; i++)
			in[i] = (hash[i]);
		in[8] = 0x80;

		#pragma unroll
		for (int i=9;i<16;i++) in[i] = 0;

		in[14] = 0x100; // size in bits

		uint32_t h[5];
		#pragma unroll
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(in, h);

		#pragma unroll
		for (int i=0; i<5; i++)
			hash[i] = h[i];

#ifdef PAD_ZEROS
		// 20 bytes hash on 32 output space
		hash[5] = 0;
		hash[6] = 0;
		hash[7] = 0;
#endif
		// second 32 bytes block hash
		hash += 8;

		#pragma unroll
		for (int i=0; i<8; i++)
			in[i] = (hash[i]);
		in[8] = 0x80;

		#pragma unroll
		for (int i=9;i<16;i++) in[i] = 0;

		in[14] = 0x100; // size in bits

		#pragma unroll
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(in, h);

		#pragma unroll
		for (int i=0; i<5; i++)
			hash[i] = h[i];

#ifdef PAD_ZEROS
		// 20 bytes hash on 32 output space
		hash[5] = 0;
		hash[6] = 0;
		hash[7] = 0;
#endif
	}
}

__host__
void lbry_ripemd160_hash_32x2(int thr_id, uint32_t threads, uint32_t *g_Hash, cudaStream_t stream)
{
	const uint32_t threadsperblock = 128;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_ripemd160_gpu_hash_32x2 <<<grid, block, 0, stream>>> (threads, (uint64_t*) g_Hash);
}

void lbry_ripemd160_init(int thr_id)
{
	//cudaMemcpyToSymbol(c_IV, IV, sizeof(IV), 0, cudaMemcpyHostToDevice);
}
