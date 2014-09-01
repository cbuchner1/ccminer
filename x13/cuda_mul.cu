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

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);


__global__ void __launch_bounds__(512,2) m7_bigmul_unroll1_gpu(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{

//__shared__ uint64_t ams[38],bms[38],ws[38]
//ams
	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {

#pragma unroll 32 
	for (int i=0;i<32;i++) {w[i*threads+thread]=0;}
#if __CUDA_ARCH__ < 500
#pragma unroll 32   
#endif
for (int i=0;i<32;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
    #pragma unroll 3
	for (int j=0;j<3;j++) {  
    muladd128(u,v,am[j*threads+thread],bm[i*threads+thread],w[(i+j)*threads+thread],c);	
    w[(i+j)*threads+thread]=v;
    c=u; 
	}
   w[(i+3)*threads+thread]=u;
 }
							 } // threads
}

__global__ void __launch_bounds__(512,4) m7_bigmul_unroll1_gpu_50(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{

//__shared__ uint64_t ams[38],bms[38],ws[38]
//ams
	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {

#pragma unroll 32 
	for (int i=0;i<32;i++) {w[i*threads+thread]=0;}
#if __CUDA_ARCH__ < 500
#pragma unroll 32   
#endif
for (int i=0;i<32;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
    #pragma unroll 3
	for (int j=0;j<3;j++) {  
    muladd128(u,v,am[j*threads+thread],bm[i*threads+thread],w[(i+j)*threads+thread],c);	
    w[(i+j)*threads+thread]=v;
    c=u; 
	}
   w[(i+3)*threads+thread]=u;
 }
							 } // threads
}


__global__ void __launch_bounds__(512,2) m7_bigmul_unroll2_gpu(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {


#if __CUDA_ARCH__ < 500
#pragma unroll
#endif
	for (int i=0;i<38;i++) {w[i*threads+thread]=0;}
#if __CUDA_ARCH__ < 500
#pragma unroll    
#endif
for (int i=0;i<35;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
#if __CUDA_ARCH__ < 500
    #pragma unroll
#endif
	for (int j=0;j<3;j++) {  
    muladd128(u,v,am[j*threads+thread],bm[i*threads+thread],w[(i+j)*threads+thread],c);	
    w[(i+j)*threads+thread]=v;
    c=u; 
	}
   w[(i+3)*threads+thread]=u;
 }
//////////////////////////////////////////////////////////////////////////////////////////////////	  

 } //// threads
}

__global__ void __launch_bounds__(512,4) m7_bigmul_unroll2_gpu_50(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {


#if __CUDA_ARCH__ < 500
#pragma unroll
#endif
	for (int i=0;i<38;i++) {w[i*threads+thread]=0;}
#if __CUDA_ARCH__ < 500
#pragma unroll    
#endif
for (int i=0;i<35;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
#if __CUDA_ARCH__ < 500
    #pragma unroll
#endif
	for (int j=0;j<3;j++) {  
    muladd128(u,v,am[j*threads+thread],bm[i*threads+thread],w[(i+j)*threads+thread],c);	
    w[(i+j)*threads+thread]=v;
    c=u; 
	}
   w[(i+3)*threads+thread]=u;
 }
//////////////////////////////////////////////////////////////////////////////////////////////////	  

 } //// threads
}



__host__ void m7_bigmul_unroll1_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order)
{

	const int threadsperblock = 512; 

dim3 grid(threads/threadsperblock);
dim3 block(threadsperblock);
  
	size_t shared_size =0;
	if (device_major[thr_id]==5) {
		m7_bigmul_unroll1_gpu_50<<<grid, block, shared_size>>>(threads,Hash1,Hash2,finalHash);}
	else {
		m7_bigmul_unroll1_gpu<<<grid, block, shared_size>>>(threads,Hash1,Hash2,finalHash);}
}

__host__ void m7_bigmul_unroll2_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order)
{

	const int threadsperblock = 512;

dim3 grid(threads/threadsperblock);
dim3 block(threadsperblock);
  
	size_t shared_size =0;

	if (device_major[thr_id]==5) {
		m7_bigmul_unroll2_gpu_50<<<grid, block, shared_size>>>(threads,Hash1,Hash2,finalHash);}
	else {
		m7_bigmul_unroll2_gpu<<<grid, block, shared_size>>>(threads,Hash1,Hash2,finalHash);}

}


__host__ void m7_bigmul_init(int thr_id, int threads)
{
	// why I am here ?
}