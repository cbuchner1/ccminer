/*
 * tiger-192 djm34
 * 
 */

/*
 * tiger-192 kernel implementation.
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


// aus heavy.cu

extern int device_major[8];
extern int device_minor[8];
extern int compute_version[8];
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

static __forceinline__ __device__ void mul_unroll1_core_test(int threads, int thread, uint64_t* am, uint64_t* bm, uint64_t *w)
{

	uint32_t B0, B1, B2, B3, B4, B5;
	LOHI(B0, B1, am[thread]);
	LOHI(B2, B3, am[threads + thread]);
	LOHI(B4, B5, am[2 * threads + thread]);



#pragma unroll
	for (int i = 0; i<35; i++) { w[i*threads + thread] = 0; }
#if __CUDA_ARCH__ < 500
#pragma unroll    
#endif
	for (int i = 0; i<32; i++) {
		uint32_t Q0;
		uint32_t Q1;
		LOHI(Q0, Q1, bm[i*threads + thread]);
		//		uint32_t W0,W1,W2,W3,W4,W5,W6,W7;
		uint4 Wa, Wb;
		LOHI(Wa.x, Wa.y, w[i*threads + thread]);
		LOHI(Wa.z, Wa.w, w[(i + 1)*threads + thread]);
		LOHI(Wb.x, Wb.y, w[(i + 2)*threads + thread]);
		LOHI(Wb.z, Wb.w, w[(i + 3)*threads + thread]);


		asm("{\n\t"
			".reg .u32 b0,b1; \n\t"
			"mad.lo.cc.u32      b0,%7,%13,%0; \n\t"
			"madc.hi.cc.u32     b1,%7,%13,0; \n\t"
			"mov.u32 %0,b0; \n\t"
			"madc.lo.cc.u32  b1,%8,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%8,%13,0; \n\t"
			"add.cc.u32      b1,b1,%1;      \n\t"
			"mov.u32 %1,b1; \n\t"
			"madc.lo.cc.u32 b0,%9,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%9,%13,0; \n\t"
			"add.cc.u32      b0,b0,%2;      \n\t"
			"mov.u32 %2,b0; \n\t"
			"madc.lo.cc.u32 b1,%10,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%10,%13,0; \n\t"
			"add.cc.u32      b1,b1,%3;      \n\t"
			"mov.u32 %3,b1; \n\t"
			"madc.lo.cc.u32 b0,%11,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%11,%13,0; \n\t"
			"add.cc.u32      b0,b0,%4;      \n\t"
			"mov.u32 %4,b0; \n\t"
			"madc.lo.cc.u32 b1,%12,%13,b1; \n\t"
			"madc.hi.cc.u32 %6,%12,%13,0; \n\t"
			"add.cc.u32      b1,b1,%5;      \n\t"
			"addc.u32     %6,%6,0;   \n\t"
			"mov.u32 %5,b1; \n\t"
			"}\n\t"
			: "+r"(Wa.x), "+r"(Wa.y), "+r"(Wa.z), "+r"(Wa.w), "+r"(Wb.x), "+r"(Wb.y), "+r"(Wb.z)
			: "r"(B0), "r"(B1), "r"(B2), "r"(B3), "r"(B4), "r"(B5), "r"(Q0));
		///////////////////////////
		asm("{\n\t"
			".reg .u32 b0,b1; \n\t"
			"mad.lo.cc.u32      b0,%7,%13,%0; \n\t"
			"madc.hi.cc.u32     b1,%7,%13,0; \n\t"
			"mov.u32 %0,b0; \n\t"
			"madc.lo.cc.u32  b1,%8,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%8,%13,0; \n\t"
			"add.cc.u32      b1,b1,%1;      \n\t"
			"mov.u32 %1,b1; \n\t"
			"madc.lo.cc.u32 b0,%9,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%9,%13,0; \n\t"
			"add.cc.u32      b0,b0,%2;      \n\t"
			"mov.u32 %2,b0; \n\t"
			"madc.lo.cc.u32 b1,%10,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%10,%13,0; \n\t"
			"add.cc.u32      b1,b1,%3;      \n\t"
			"mov.u32 %3,b1; \n\t"
			"madc.lo.cc.u32 b0,%11,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%11,%13,0; \n\t"
			"add.cc.u32      b0,b0,%4;      \n\t"
			"mov.u32 %4,b0; \n\t"
			"madc.lo.cc.u32 b1,%12,%13,b1; \n\t"
			"madc.hi.cc.u32 %6,%12,%13,0; \n\t"
			"add.cc.u32      b1,b1,%5;      \n\t"
			"addc.u32     %6,%6,0;   \n\t"
			"mov.u32 %5,b1; \n\t"
			"}\n\t"
			: "+r"(Wa.y), "+r"(Wa.z), "+r"(Wa.w), "+r"(Wb.x), "+r"(Wb.y), "+r"(Wb.z), "+r"(Wb.w)
			: "r"(B0), "r"(B1), "r"(B2), "r"(B3), "r"(B4), "r"(B5), "r"(Q1));

		w[i*threads + thread] = MAKE_ULONGLONG(Wa.x, Wa.y);
		w[(i + 1)*threads + thread] = MAKE_ULONGLONG(Wa.z, Wa.w);
		w[(i + 2)*threads + thread] = MAKE_ULONGLONG(Wb.x, Wb.y);
		w[(i + 3)*threads + thread] = MAKE_ULONGLONG(Wb.z, Wb.w);



	}

}

static __forceinline__ __device__ void mul_unroll2_core_test(int threads, int thread, uint64_t* am, uint64_t* bm, uint64_t *w)
{

	uint32_t B0, B1, B2, B3, B4, B5;
	LOHI(B0, B1, am[thread]);
	LOHI(B2, B3, am[threads + thread]);
	LOHI(B4, B5, am[2 * threads + thread]);



#pragma unroll
	for (int i = 0; i<38; i++) { w[i*threads + thread] = 0; }
#if __CUDA_ARCH__ < 500
#pragma unroll    
#endif
	for (int i = 0; i<35; i++) {
		uint32_t Q0;
		uint32_t Q1;
		LOHI(Q0, Q1, bm[i*threads + thread]);
		//		uint32_t W0, W1, W2, W3, W4, W5, W6, W7;
		uint4 Wa, Wb;
		LOHI(Wa.x, Wa.y, w[i*threads + thread]);
		LOHI(Wa.z, Wa.w, w[(i + 1)*threads + thread]);
		LOHI(Wb.x, Wb.y, w[(i + 2)*threads + thread]);
		LOHI(Wb.z, Wb.w, w[(i + 3)*threads + thread]);


		asm("{\n\t"
			".reg .u32 b0,b1; \n\t"
			"mad.lo.cc.u32      b0,%7,%13,%0; \n\t"
			"madc.hi.cc.u32     b1,%7,%13,0; \n\t"
			"mov.u32 %0,b0; \n\t"
			"madc.lo.cc.u32  b1,%8,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%8,%13,0; \n\t"
			"add.cc.u32      b1,b1,%1;      \n\t"
			"mov.u32 %1,b1; \n\t"
			"madc.lo.cc.u32 b0,%9,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%9,%13,0; \n\t"
			"add.cc.u32      b0,b0,%2;      \n\t"
			"mov.u32 %2,b0; \n\t"
			"madc.lo.cc.u32 b1,%10,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%10,%13,0; \n\t"
			"add.cc.u32      b1,b1,%3;      \n\t"
			"mov.u32 %3,b1; \n\t"
			"madc.lo.cc.u32 b0,%11,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%11,%13,0; \n\t"
			"add.cc.u32      b0,b0,%4;      \n\t"
			"mov.u32 %4,b0; \n\t"
			"madc.lo.cc.u32 b1,%12,%13,b1; \n\t"
			"madc.hi.cc.u32 %6,%12,%13,0; \n\t"
			"add.cc.u32      b1,b1,%5;      \n\t"
			"addc.u32     %6,%6,0;   \n\t"
			"mov.u32 %5,b1; \n\t"
			"}\n\t"
			: "+r"(Wa.x), "+r"(Wa.y), "+r"(Wa.z), "+r"(Wa.w), "+r"(Wb.x), "+r"(Wb.y), "+r"(Wb.z)
			: "r"(B0), "r"(B1), "r"(B2), "r"(B3), "r"(B4), "r"(B5), "r"(Q0));
		///////////////////////////
		asm("{\n\t"
			".reg .u32 b0,b1; \n\t"
			"mad.lo.cc.u32      b0,%7,%13,%0; \n\t"
			"madc.hi.cc.u32     b1,%7,%13,0; \n\t"
			"mov.u32 %0,b0; \n\t"
			"madc.lo.cc.u32  b1,%8,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%8,%13,0; \n\t"
			"add.cc.u32      b1,b1,%1;      \n\t"
			"mov.u32 %1,b1; \n\t"
			"madc.lo.cc.u32 b0,%9,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%9,%13,0; \n\t"
			"add.cc.u32      b0,b0,%2;      \n\t"
			"mov.u32 %2,b0; \n\t"
			"madc.lo.cc.u32 b1,%10,%13,b1; \n\t"
			"madc.hi.cc.u32 b0,%10,%13,0; \n\t"
			"add.cc.u32      b1,b1,%3;      \n\t"
			"mov.u32 %3,b1; \n\t"
			"madc.lo.cc.u32 b0,%11,%13,b0; \n\t"
			"madc.hi.cc.u32 b1,%11,%13,0; \n\t"
			"add.cc.u32      b0,b0,%4;      \n\t"
			"mov.u32 %4,b0; \n\t"
			"madc.lo.cc.u32 b1,%12,%13,b1; \n\t"
			"madc.hi.cc.u32 %6,%12,%13,0; \n\t"
			"add.cc.u32      b1,b1,%5;      \n\t"
			"addc.u32     %6,%6,0;   \n\t"
			"mov.u32 %5,b1; \n\t"
			"}\n\t"
			: "+r"(Wa.y), "+r"(Wa.z), "+r"(Wa.w), "+r"(Wb.x), "+r"(Wb.y), "+r"(Wb.z), "+r"(Wb.w)
			: "r"(B0), "r"(B1), "r"(B2), "r"(B3), "r"(B4), "r"(B5), "r"(Q1));

		w[i*threads + thread] = MAKE_ULONGLONG(Wa.x, Wa.y);
		w[(i + 1)*threads + thread] = MAKE_ULONGLONG(Wa.z, Wa.w);
		w[(i + 2)*threads + thread] = MAKE_ULONGLONG(Wb.x, Wb.y);
		w[(i + 3)*threads + thread] = MAKE_ULONGLONG(Wb.z, Wb.w);



	}

}


__global__ void __launch_bounds__(512, 3) m7_bigmul_unroll1_gpu(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{

		mul_unroll1_core_test(threads, thread, am, bm, w);
	} // threads
}

__global__ void __launch_bounds__(256, 2) m7_bigmul_unroll1_gpu_50(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		mul_unroll1_core_test(threads, thread, am, bm, w);
	} // threads
}

__global__ void __launch_bounds__(256, 4) m7_bigmul_unroll1_gpu_80(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		mul_unroll1_core_test(threads, thread, am, bm, w);
	} // threads
}


__global__ void __launch_bounds__(512, 2) m7_bigmul_unroll2_gpu(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		mul_unroll2_core_test(threads, thread, am, bm, w);

	} //// threads
}

__global__ void __launch_bounds__(512, 2) m7_bigmul_unroll2_gpu_50(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		mul_unroll2_core_test(threads, thread, am, bm, w);
	} //// threads
}




__host__ void m7_bigmul_unroll1_cpu(int thr_id, int threads, uint64_t* Hash1, uint64_t* Hash2, uint64_t *finalHash, int order)
{

	int threadsperblock = 512;
	if (compute_version[thr_id] >= 50) { threadsperblock = 256; }
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;
	if (compute_version[thr_id]==50) {
		m7_bigmul_unroll1_gpu_50 << <grid, block, shared_size >> >(threads, Hash1, Hash2, finalHash);
	}
	else if (compute_version[thr_id]==52) {
		m7_bigmul_unroll1_gpu_80 << <grid, block, shared_size >> >(threads, Hash1, Hash2, finalHash);
	}
	else {
		m7_bigmul_unroll1_gpu << <grid, block, shared_size >> >(threads, Hash1, Hash2, finalHash);
	}

}

__host__ void m7_bigmul_unroll2_cpu(int thr_id, int threads, uint64_t* Hash1, uint64_t* Hash2, uint64_t *finalHash, int order)
{

	const int threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	if (compute_version[thr_id] >= 50) {
		m7_bigmul_unroll2_gpu << <grid, block, shared_size >> >(threads, Hash1, Hash2, finalHash);
	}
	else {
		m7_bigmul_unroll2_gpu << <grid, block, shared_size >> >(threads, Hash1, Hash2, finalHash);
	}

}




__host__ void m7_bigmul_init(int thr_id, int threads)
{
	// why I am here ?
}