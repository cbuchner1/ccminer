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
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__forceinline__  __device__  void bigmul(uint64_t *w, uint64_t* am, uint64_t* bm, int sizea, int sizeb, int thread)
{

	
	int threads = 256*256*8*2;
#pragma unroll
	for (int i=0;i<sizea+sizeb;i++) {w[i*threads+thread]=0;}
#pragma unroll
for (int i=0;i<sizeb;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
    #pragma unroll
	for (int j=0;j<sizea;j++) {
    muladd128(u,v,am[j*threads+thread],bm[i*threads+thread],w[(i+j)*threads+thread],c);	
    w[(i+j)*threads+thread]=v;
    c=u;
	}
   w[(i+sizea)*threads+thread]=u;
 }
  
}



__global__ void m7_bigmul_gpu(int threads, uint64_t* Hash1, uint64_t* Hash2, uint64_t* Hash3, uint64_t* Hash4, uint64_t *Hash5, uint64_t *Hash6, uint64_t *Hash7, uint32_t foundNonce, uint32_t startNonce)
{

	int count=0;

	                        int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
uint64_t stage1[16];
uint64_t stage2[24]; 
uint64_t stage3[28];
uint64_t stage4[32];
uint64_t stage5[35]; 
uint64_t stage6[38];
     if (thread== foundNonce-startNonce) {          
for (int i=0;i<8;i++) {
int idx = i*threads+thread;

printf("%d Sha256 %08x %08x Sha512 %08x %08x Keccak %08x %08x Whirlpool %08x %08x Haval %08x %08x Tiger %08x %08x Ripemd %08x %08x \n",
i, HIWORD(Hash1[idx]),LOWORD(Hash1[idx]),HIWORD(Hash2[idx]),LOWORD(Hash2[idx]),HIWORD(Hash3[idx]),LOWORD(Hash3[idx]),HIWORD(Hash4[idx]),LOWORD(Hash4[idx]),
   HIWORD(Hash5[idx]),LOWORD(Hash5[idx]),HIWORD(Hash6[idx]),LOWORD(Hash6[idx]),HIWORD(Hash7[idx]),LOWORD(Hash7[idx]));
}}



//bigmul(stage1,Hash1,Hash2,8,8,count);
//bigmul(stage2, stage1,Hash3,16,8,count);
//bigmul(stage3, stage2,Hash4,24,4,count);
//bigmul(stage4, stage3,Hash5,28,4,count);
//bigmul(stage5, stage4,Hash6,32,3,count);
//bigmul(stage6, stage5,Hash7,35,3,count);

//for(int i=0;i<38;i++)  {finalHash[i*threads+thread]=stage6[i];}
//printf("stage6[i] %08x %08x\n",HIWORD(stage6[0]),LOWORD(stage6[0]));
//////////////////////////////////////////////////////////////////////////////////////////////////	  


 } //// threads
}

__global__ void m7_bigmul1_gpu(int threads, int sizea, int sizeb, uint64_t* am, uint64_t* bm, uint64_t *w)
{


	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {



#pragma unroll
	for (int i=0;i<sizea+sizeb;i++) {w[i*threads+thread]=0;}
#pragma unroll    
for (int i=0;i<sizeb;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
    #pragma unroll
	for (int j=0;j<sizea;j++) {  
    muladd128(u,v,am[j*threads+thread],bm[i*threads+thread],w[(i+j)*threads+thread],c);	
    w[(i+j)*threads+thread]=v;
    c=u; 
	}
   w[(i+sizea)*threads+thread]=u;
 }


//////////////////////////////////////////////////////////////////////////////////////////////////	  

 } //// threads
}



__global__ void m7_bigmul_unroll1_gpu(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
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

__global__ void m7_bigmul_unroll1_gpu_std(int threads, uint64_t* amg, uint64_t* bmg, uint64_t *wg)
{

//__shared__ uint64_t ams[38],bms[38],ws[38]
//ams
	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {

uint64_t * am = amg + 8*thread;
uint64_t * bm = bmg + 38*thread;
uint64_t * w  = wg +  38*thread;

#pragma unroll 32 
	for (int i=0;i<32;i++) {w[i]=0;}
#if __CUDA_ARCH__ < 500
#pragma unroll 32   
#endif
for (int i=0;i<32;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;  
    #pragma unroll 3
	for (int j=0;j<3;j++) {  
    muladd128(u,v,am[j],bm[i],w[(i+j)],c);	
    w[(i+j)]=v;
    c=u; 
	}
   w[(i+3)]=u;
 }
							 } // threads
}


__global__ void m7_bigmul_unroll2_gpu(int threads, uint64_t* am, uint64_t* bm, uint64_t *w)
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

__global__ void m7_bigmul_unroll2_gpu_std(int threads, uint64_t* amg, uint64_t* bmg, uint64_t *wg)
{


	                         int thread = (blockDim.x * blockIdx.x + threadIdx.x);

                             if (thread < threads)
                           {

uint64_t * am = amg + 8*thread;
uint64_t * bm = bmg + 38*thread;
uint64_t * w  = wg +  38*thread;

#if __CUDA_ARCH__ < 500
#pragma unroll
#endif
	for (int i=0;i<38;i++) {w[i]=0;}
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
    muladd128(u,v,am[j],bm[i],w[(i+j)],c);	
    w[(i+j)]=v;
    c=u; 
	}
   w[(i+3)]=u;
 }
//////////////////////////////////////////////////////////////////////////////////////////////////	  

 } //// threads
}



__host__ void m7_bigmul_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2, uint64_t* Hash3, uint64_t* Hash4, 
	                                                uint64_t *Hash5, uint64_t *Hash6, uint64_t *Hash7,uint32_t foundNonce, uint32_t StartNonce,int order)
{

	const int threadsperblock = 256;
	 

dim3 grid((threads + threadsperblock-1)/threadsperblock);
dim3 block(threadsperblock);
  
	size_t shared_size =0;
	m7_bigmul_gpu<<<grid, block, shared_size>>>(threads,Hash1,Hash2,Hash3,Hash4,
		                                                Hash5,Hash6,Hash7, foundNonce, StartNonce);
 
	
//	MyStreamSynchronize(NULL, order, thr_id); 
	 
	 

}

__host__ void m7_bigmul1_cpu(int thr_id, int threads,int len1,int len2,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order)
{

	const int threadsperblock = 256;
	 

dim3 grid((threads + threadsperblock-1)/threadsperblock);
dim3 block(threadsperblock);
  
	size_t shared_size =0;
	m7_bigmul1_gpu<<<grid, block, shared_size>>>(threads,len1,len2,Hash1,Hash2,finalHash);
 
	
//	MyStreamSynchronize(NULL, order, thr_id); 
	 
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaThreadSynchronize());
}


__host__ void m7_bigmul_unroll1_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order)
{

	const int threadsperblock = 256;
	 

dim3 grid((threads + threadsperblock-1)/threadsperblock);
dim3 block(threadsperblock);
  
	size_t shared_size =0;
	m7_bigmul_unroll1_gpu<<<grid, block, shared_size>>>(threads,Hash1,Hash2,finalHash);
}

__host__ void m7_bigmul_unroll2_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order)
{

	const int threadsperblock = 256;
	 

dim3 grid((threads + threadsperblock-1)/threadsperblock);
dim3 block(threadsperblock);
  
	size_t shared_size =0;
	m7_bigmul_unroll2_gpu<<<grid, block, shared_size>>>(threads,Hash1,Hash2,finalHash);
}


__host__ void m7_bigmul_init(int thr_id, int threads)
{
	// why I am here ?
}