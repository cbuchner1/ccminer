/*
 * sha256 djm34, catia
 * 
 */

/*
 * sha-256 kernel implementation.
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

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>


#include "cuda_helper.h"

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);



typedef struct t4_t{
	uint64_t high,low;
} t4_t;

__device__ __forceinline__ 
ulonglong2 umul64wide (unsigned long long int a, 
                       unsigned long long int b)
{
    ulonglong2 res;
    asm ("{\n\t"
         ".reg .u32 r0, r1, r2, r3, alo, ahi, blo, bhi;\n\t"
         "mov.b64         {alo,ahi}, %2;   \n\t"
         "mov.b64         {blo,bhi}, %3;   \n\t"
         "mul.lo.u32      r0, alo, blo;    \n\t"
         "mul.hi.u32      r1, alo, blo;    \n\t"
         "mad.lo.cc.u32   r1, alo, bhi, r1;\n\t"
         "madc.hi.u32     r2, alo, bhi,  0;\n\t"
         "mad.lo.cc.u32   r1, ahi, blo, r1;\n\t"
         "madc.hi.cc.u32  r2, ahi, blo, r2;\n\t"
         "madc.hi.u32     r3, ahi, bhi,  0;\n\t"
         "mad.lo.cc.u32   r2, ahi, bhi, r2;\n\t"
         "addc.u32        r3, r3,  0;      \n\t"
         "mov.b64         %0, {r0,r1};     \n\t"  
         "mov.b64         %1, {r2,r3};     \n\t"
         "}"
         : "=l"(res.x), "=l"(res.y)
         : "l"(a), "l"(b));
    return res;
}

#define umul_ppmm(h,l,m,n) \
{ \
	ulonglong2 foom = umul64wide(m,n); \
	h = foom.y; \
	l = foom.x; \
}


__device__ __forceinline__ void umul_ppmmT4(t4_t *h, t4_t *l, t4_t m, t4_t n)
{
    asm ("{\n\t"
         ".reg .u32 o0, o1, o2, o3, o4;    \n\t"
         ".reg .u32 o5, o6, o7, i8, i9;    \n\t"
         ".reg .u32 i10, i11, i12, i13;    \n\t"
         ".reg .u32 i14, i15, i16, i17;    \n\t"
         ".reg .u32 i18, i19, i20, i21;    \n\t"
         ".reg .u32 i22, i23;              \n\t"
         "mov.b64         { i8, i9}, %4;   \n\t"
         "mov.b64         {i10,i11}, %5;   \n\t"
         "mov.b64         {i12,i13}, %6;   \n\t"
         "mov.b64         {i14,i15}, %7;   \n\t"
         "mov.b64         {i16,i17}, %8;   \n\t"
         "mov.b64         {i18,i19}, %9;   \n\t"
         "mov.b64         {i20,i21},%10;   \n\t"
         "mov.b64         {i22,i23},%11;   \n\t"
         "mul.lo.u32      o0,  i8, i16;    \n\t"
         "mul.hi.u32      o1,  i8, i16;    \n\t"
         "mad.lo.cc.u32   o1,  i8, i17, o1;\n\t"
         "madc.hi.u32     o2,  i8, i17,  0;\n\t"
         "mad.lo.cc.u32   o1,  i9, i16, o1;\n\t"
         "madc.hi.cc.u32  o2,  i9, i16, o2;\n\t"
         "madc.hi.u32     o3,  i8, i18,  0;\n\t"
         "mad.lo.cc.u32   o2,  i8, i18, o2;\n\t"
         "madc.hi.cc.u32  o3,  i9, i17, o3;\n\t"
         "madc.hi.u32     o4,  i8, i19,  0;\n\t"
         "mad.lo.cc.u32   o2,  i9, i17, o2;\n\t"
         "madc.hi.cc.u32  o3, i10, i16, o3;\n\t"
         "madc.hi.cc.u32  o4,  i9, i18, o4;\n\t"
         "addc.u32        o5,   0,   0;\n\t"
         "mad.lo.cc.u32   o2, i10, i16, o2;\n\t"
	 "madc.lo.cc.u32  o3,  i8, i19, o3;\n\t"
         "madc.hi.cc.u32  o4, i10, i17, o4;\n\t"
         "madc.hi.cc.u32  o5,  i9, i19, o5;\n\t"
         "addc.u32        o6,   0,   0;\n\t"
         "mad.lo.cc.u32   o3,  i9, i18, o3;\n\t"
         "madc.hi.cc.u32  o4, i11, i16, o4;\n\t"
         "madc.hi.cc.u32  o5, i10, i18, o5;\n\t"
         "addc.u32        o6,   0,  o6;\n\t"
         "mad.lo.cc.u32   o3, i10, i17, o3;\n\t"
         "addc.u32        o4,   0,  o4;\n\t"
         "mad.hi.cc.u32   o5, i11, i17, o5;\n\t"
         "madc.hi.cc.u32  o6, i10, i19, o6;\n\t"
         "addc.u32        o7,   0,   0;\n\t"
         "mad.lo.cc.u32   o3, i11, i16, o3;\n\t"
         "madc.lo.cc.u32  o4,  i9, i19, o4;\n\t"
         "addc.u32        o5,   0,  o5;\n\t"
         "mad.hi.cc.u32   o6, i11, i18, o6;\n\t"
         "addc.u32        o7,   0,  o7;\n\t"
         "mad.lo.cc.u32   o4, i10, i18, o4;\n\t"
         "addc.u32        o5,   0,  o5;\n\t"
         "mad.hi.u32      o7, i11, i19, o7;\n\t"
         "mad.lo.cc.u32   o4, i11, i17, o4;\n\t"
         "addc.u32        o5,   0,  o5;\n\t"
         "mad.lo.cc.u32   o5, i10, i19, o5;\n\t"
         "addc.u32        o6,   0,  o6;\n\t"
         "mad.lo.cc.u32   o5, i11, i18, o5;\n\t"
         "addc.u32        o6,   0,  o6;\n\t"
         "mad.lo.cc.u32   o6, i11, i19, o6;\n\t"
         "addc.u32        o7,   0,  o7;\n\t"
         "mov.b64         %0, {o0,o1};     \n\t"
         "mov.b64         %1, {o2,o3};     \n\t"
         "mov.b64         %2, {o4,o5};     \n\t"
         "mov.b64         %3, {o6,o7};     \n\t"
         "}"
         : "=l"(l->low), "=l"(l->high), "=l"(h->low), "=l"(h->high)
         : "l"(m.low), "l"(m.high), "l"(0ULL), "l"(0ULL),
           "l"(n.low), "l"(n.high), "l"(0ULL), "l"(0ULL));
}

#if 0
__device__ __forceinline__ void umul_ppmmT4(t4_t *h, t4_t *l, t4_t m, t4_t n){
	uint64_t th,tl;
	uint32_t c,c2;
	umul_ppmm(l->high,l->low,m.low,n.low);

	umul_ppmm(th,tl,m.high,n.low);
	l->high += tl;
	c = (l->high < tl);
	h->low = th + c;
	c = (h->low < c);
	h->high = c;

	//Second word
	umul_ppmm(th,tl,m.low,n.high);
	l->high += tl;
	c = l->high < tl;
	h->low += th;
	c2 = h->low < th;
	h->low += c;
	c2 += h->low < c;
	h->high += c2;

	umul_ppmm(th,tl,m.high,n.high);
	h->low += tl;
	c = h->low < tl;
	h->high += th + c;
}
#endif


__device__ __forceinline__ t4_t T4(uint32_t thread, uint32_t threads, uint32_t idx, uint64_t *g){
	t4_t ret;
	ret.high = g[(idx*2 + 1)*threads + thread];
	ret.low = g[(idx*2)*threads + thread];

	

	return ret;
}

__device__ __forceinline__ void T4_store(uint32_t thread, uint32_t threads, uint32_t idx, uint64_t *g, t4_t val){
	g[(idx*2 + 1)*threads + thread]=val.high;
	g[(idx*2)*threads + thread]=val.low;

	

}

__device__ __forceinline__ void T4_set(t4_t *d, uint64_t v){
	d->high = 0;
	d->low = v;
}

__device__ __forceinline__ t4_t T4_add(t4_t a, t4_t b){
	t4_t ret;
	uint32_t c=0;
	ret.low = a.low + b.low;
	if(ret.low < a.low)
	    c=1;
	ret.high = a.high + b.high + c;
	return ret;
}

__device__ __forceinline__ t4_t T4_add(uint64_t a, t4_t b){
	t4_t ret;
	uint32_t c=0;
	ret.low = a + b.low;
	if(ret.low < a)
	    c=1;
	ret.high = b.high + c;
	return ret;
}


__device__ __forceinline__ uint32_t T4_lt(t4_t a, t4_t b){
	if(a.high < b.high)
		return 1;
	if(a.high == b.high && a.low < b.low)
		return 1;
	return 0;
}

__device__ __forceinline__ uint32_t T4_gt(t4_t a, uint64_t b){
	if(a.high)
		return 1;
	if(a.low > b)
		return 1;
	return 0;
}


__device__ void mulScalarT4(uint32_t thread, uint32_t threads, uint32_t len, uint64_t* g_p, uint64_t* g_v, t4_t sml, uint32_t *size){
  t4_t ul, cl, hpl, lpl;
  uint32_t i;
  T4_set(&cl,0);
  for(i=0; i < len; i++) {
      ul = T4(thread,threads,i,g_v);
      umul_ppmmT4 (&hpl, &lpl, ul, sml);

      lpl = T4_add(lpl,cl);
      cl = T4_add(T4_lt(lpl,cl),hpl);

      T4_store(thread,threads,i,g_p,lpl);
    }

    T4_store(thread,threads,len,g_p,cl);
    *size = len + T4_gt(cl,0);
}


__device__ void mulScalar(uint32_t thread, uint32_t threads, uint32_t len, uint64_t* g_p, uint64_t* g_v, uint64_t sml, uint32_t *size){
  uint64_t ul, cl, hpl, lpl;
  uint32_t i;
  cl = 0;
  for(i=0; i < len; i++) {
      ul = g_v[i*threads + thread];
      umul_ppmm (hpl, lpl, ul, sml);

      lpl += cl;
      cl = (lpl < cl) + hpl;

      g_p[i*threads + thread] = lpl;
    }

    g_p[len*threads + thread] = cl;
    *size = len + (cl != 0);
}

uint64_t __device__ addmul_1g (uint32_t thread, uint32_t threads, uint64_t *sum, uint32_t sofst, uint64_t *x, uint64_t xsz, uint64_t a){
	uint64_t carry=0;
	uint32_t i;
	uint64_t ul,lpl,hpl,rl;

	for(i=0; i < xsz; i++){
		
      		ul = x[i*threads + thread];
      		umul_ppmm (hpl, lpl, ul, a);

      		lpl += carry;
      		carry = (lpl < carry) + hpl;

      		rl = sum[(i+sofst) * threads + thread];
      		lpl = rl + lpl;
      		carry += lpl < rl;
      		sum[(i+sofst)*threads + thread] = lpl;
    	}

  	return carry;
}

t4_t __device__ addmul_1gT4 (uint32_t thread, uint32_t threads, uint64_t *sum, uint32_t sofst, uint64_t *x, uint64_t xsz, t4_t a){
	t4_t carry;
	uint32_t i;
	t4_t ul,lpl,hpl,rl;
	T4_set(&carry,0);
	for(i=0; i < xsz; i++){
		
      		ul = T4(thread,threads,i,x);
      		umul_ppmmT4 (&hpl, &lpl, ul, a);

      		lpl = T4_add(lpl,carry);
      		carry = T4_add(T4_lt(lpl,carry), hpl);

      		rl = T4(thread,threads,i+sofst,sum);
      		lpl = T4_add(rl,lpl);
      		carry = T4_add(T4_lt(lpl,rl),carry);
      		T4_store(thread,threads,i+sofst,sum,lpl);
    	}

  	return carry;
}



__global__ void gpu_mul(int threads, uint32_t ulegs, uint32_t vlegs, uint64_t *g_u, uint64_t *g_v, uint64_t *g_p)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
	if(ulegs < vlegs){
		uint64_t t1=ulegs;
		ulegs = vlegs;
		vlegs = t1;

		uint64_t *t2 = g_u;
		g_u = g_v;
		g_v = t2;
	}

	uint32_t vofst=1,rofst=1,psize=0;
	mulScalar(thread,threads,ulegs,g_p,g_u,g_v[thread],&psize);   

#if 1

  	while (vofst < vlegs) {

	    	g_p[(psize+0)*threads+thread] = 0;

            	g_p[(ulegs+rofst)*threads + thread] = addmul_1g (thread, threads, g_p ,rofst , g_u, ulegs,  g_v[vofst*threads+thread]);

	    	vofst++; rofst++;
	    	psize++;
        }




#endif
    }
}

__global__ void  gpu_mulT4(int threads, uint32_t ulegs, uint32_t vlegs, uint64_t *g_u, uint64_t *g_v, uint64_t *g_p)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {

	if(ulegs < vlegs){  
		uint64_t t1=ulegs;
		ulegs = vlegs;   
		vlegs = t1;

		uint64_t *t2 = g_u;
		g_u = g_v;
		g_v = t2;
	}

	ulegs >>= 1; vlegs >>= 1;

	

	uint32_t vofst=1,rofst=1,psize=0;
	mulScalarT4(thread,threads,ulegs,g_p,g_u,T4(thread,threads,0,g_v),&psize);

#if 1
	t4_t zero;
	T4_set(&zero,0);
	

#pragma unroll
	    for (vofst=1;vofst<vlegs;vofst++) {  
	    	T4_store(thread,threads,psize,g_p,zero);

            	T4_store(thread,threads,ulegs+rofst,g_p,addmul_1gT4 (thread, threads, g_p ,rofst , g_u, ulegs,T4(thread,threads,vofst,g_v)));
			rofst++;
	    	psize++;
        }


#endif
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__host__ void cpu_mul(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p,int order)
{

	const int threadsperblock = 512; // Alignment mit mixtab Gr\F6sse. NICHT \C4NDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size =0;
  	gpu_mul<<<grid, block, shared_size>>>(threads, alegs, blegs, g_a, g_b, g_p) ;

}

__host__ void cpu_mulT4(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p, int order)
{

	const int threadsperblock = 256; 

	dim3 grid(2*(threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size =0;
  	
	gpu_mulT4<<<grid, block, shared_size>>>(threads, blegs, alegs, g_b, g_a, g_p) ;
}

__host__ void mul_init(){

}
