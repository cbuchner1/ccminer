/*
 * "pluck" kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2015  djm34
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

__device__  uint8_t *  hashbuffer;
uint32_t *d_PlNonce[8];
__constant__  uint32_t pTarget[8];
__constant__  uint32_t  c_data[20];
#include "cuda_vector.h" 


#define HASH_MEMORY_8bit 131072
#define HASH_MEMORY_32bit 32768
#define HASH_MEMORY 4096

static __constant__  uint32_t H256[8] = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372,
	0xA54FF53A, 0x510E527F, 0x9B05688C,
	0x1F83D9AB, 0x5BE0CD19
};

static  __constant__  uint32_t Ksha[64] = {
	0x428A2F98, 0x71374491,
	0xB5C0FBCF, 0xE9B5DBA5,
	0x3956C25B, 0x59F111F1,
	0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01,
	0x243185BE, 0x550C7DC3,
	0x72BE5D74, 0x80DEB1FE,
	0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786,
	0x0FC19DC6, 0x240CA1CC,
	0x2DE92C6F, 0x4A7484AA,
	0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D,
	0xB00327C8, 0xBF597FC7,
	0xC6E00BF3, 0xD5A79147,
	0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138,
	0x4D2C6DFC, 0x53380D13,
	0x650A7354, 0x766A0ABB,
	0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B,
	0xC24B8B70, 0xC76C51A3,
	0xD192E819, 0xD6990624,
	0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08,
	0x2748774C, 0x34B0BCB5,
	0x391C0CB3, 0x4ED8AA4A,
	0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F,
	0x84C87814, 0x8CC70208,
	0x90BEFFFA, 0xA4506CEB,
	0xBEF9A3F7, 0xC67178F2
};


#define SALSA(a,b,c,d) { \
    t =a+d; b^=rotate(t,  7);    \
    t =b+a; c^=rotate(t,  9);    \
    t =c+b; d^=rotate(t, 13);    \
    t =d+c; a^=rotate(t, 18);     \
}


#define SALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s5,state.s9,state.sd,state.s1); \
SALSA(state.sa,state.se,state.s2,state.s6); \
SALSA(state.sf,state.s3,state.s7,state.sb); \
SALSA(state.s0,state.s1,state.s2,state.s3); \
SALSA(state.s5,state.s6,state.s7,state.s4); \
SALSA(state.sa,state.sb,state.s8,state.s9); \
SALSA(state.sf,state.sc,state.sd,state.se); \
	} 


static __device__ __forceinline__ uint16 xor_salsa8(const uint16 &Bx)
{
	uint32_t t;
	uint16 state = Bx;
	SALSA_CORE(state);
	SALSA_CORE(state);
	SALSA_CORE(state);
	SALSA_CORE(state);
	return(state+Bx);
}



// sha256

static __device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	uint32_t r1 = SPH_ROTR32(x, 2);
	uint32_t r2 = SPH_ROTR32(x, 13);
	uint32_t r3 = SPH_ROTR32(x, 22);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	uint32_t r1 = SPH_ROTR32(x, 6);
	uint32_t r2 = SPH_ROTR32(x, 11);
	uint32_t r3 = SPH_ROTR32(x, 25);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	uint64_t r1 = SPH_ROTR32(x, 7);
	uint64_t r2 = SPH_ROTR32(x, 18);
	uint64_t r3 = shr_t32(x, 3);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	uint64_t r1 = SPH_ROTR32(x, 17);
	uint64_t r2 = SPH_ROTR32(x, 19);
	uint64_t r3 = shr_t32(x, 10);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ void sha2_step1(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d, const uint32_t e, 
const uint32_t f, const uint32_t g, uint32_t &h, const uint32_t in, const uint32_t Kshared)
{
	uint32_t t1, t2;
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a, b, c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

static __device__ __forceinline__ void sha2_step2(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d, const uint32_t e, 
const uint32_t f, const uint32_t g, uint32_t &h, uint32_t* in, const uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1, t2;

	int pcidx1 = (pc - 2) & 0xF;
	int pcidx2 = (pc - 7) & 0xF;
	int pcidx3 = (pc - 15) & 0xF;
	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];


	uint32_t ssg21 = ssg2_1(inx1);
	uint32_t ssg20 = ssg2_0(inx3);
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a, b, c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;

}


static __device__ __forceinline__ void sha2_round_body(uint32_t* in, uint32_t* r)
{
	uint32_t a = r[0];
	uint32_t b = r[1];
	uint32_t c = r[2];
	uint32_t d = r[3];
	uint32_t e = r[4];
	uint32_t f = r[5];
	uint32_t g = r[6];
	uint32_t h = r[7];

	sha2_step1(a, b, c, d, e, f, g, h, in[0], Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[1], Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[2], Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[3], Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[4], Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[5], Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[6], Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[7], Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[8], Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[9], Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[10], Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[11], Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[12], Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[13], Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[14], Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[15], Ksha[15]);

#pragma unroll 3
	for (int i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, in, 15, Ksha[31 + 16 * i]);

	}



	r[0] += a;
	r[1] += b;
	r[2] += c;
	r[3] += d;
	r[4] += e;
	r[5] += f;
	r[6] += g;
	r[7] += h;
}


static __device__ __forceinline__ uint8 sha256_64(uint32_t *data)
{

	uint32_t __align__(64) in[16];
    uint32_t __align__(32) buf[8];
	
	((uint16 *)in)[0] = swapvec((uint16*)data);

	((uint8*)buf)[0] = ((uint8*)H256)[0];

	sha2_round_body(in, buf);

#pragma unroll 14
	for (int i = 0; i<14; i++) { in[i + 1] = 0; }
	in[0] = 0x80000000;
	in[15] = 0x200;


	sha2_round_body(in, buf);
	return swapvec((uint8*)buf);
}


static __device__ __forceinline__ uint8 sha256_80(uint32_t nonce)
{

//	uint32_t in[16], buf[8];
	uint32_t __align__(64) in[16];
	uint32_t __align__(32) buf[8];
	((uint16 *)in)[0] = swapvec((uint16*)c_data);

	((uint8*)buf)[0] = ((uint8*)H256)[0];

	sha2_round_body(in, buf);


#pragma unroll 3
	for (int i = 0; i<3; i++) { in[i] = cuda_swab32(c_data[i + 16]); }
//	in[3] = cuda_swab32(nonce);
    in[3] = nonce;
	in[4] = 0x80000000;
	in[15] = 0x280;

#pragma unroll 10
	for (int i = 5; i<15; i++) { in[i] = 0; }

	sha2_round_body(in, buf);
	return swapvec((uint8*)buf);
}


#define SHIFT 32 * 1024 * 4
__global__ __launch_bounds__(256, 1) void pluck_gpu_hash0_v50(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		int shift = SHIFT * thread; //uint32_t
		((uint8*)(hashbuffer + shift))[0] = sha256_80(nonce);
		((uint8*)(hashbuffer + shift))[1] = make_uint8(0, 0, 0, 0, 0, 0, 0, 0);
		for (int i = 2; i < 5; i++)
		{
			uint32_t randmax = i * 32 - 4;
			uint32_t randseed[16];
			uint32_t randbuffer[16];
			uint32_t joint[16];
			uint8 Buffbuffer[2];

			((uint8*)randseed)[0] = __ldg8(&(hashbuffer + shift)[32 * i - 64]);
			((uint8*)randseed)[1] = __ldg8(&(hashbuffer + shift)[32 * i - 32]);

			

			((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);

//			((uint8*)joint)[0] = __ldg8(&(hashbuffer + shift)[(i - 1) << 5]);
			((uint8*)joint)[0] = ((uint8*)randseed)[1];
#pragma unroll
			for (int j = 0; j < 8; j++)
			{
				uint32_t rand = randbuffer[j] % (randmax - 32);
				joint[j + 8] = __ldgtoint_unaligned(&(hashbuffer + shift)[rand]); 
			}

			uint8 truc = sha256_64(joint);
			((uint8*)(hashbuffer + shift))[i] = truc;
			((uint8*)randseed)[0] = ((uint8*)joint)[0];
			((uint8*)randseed)[1] = truc;


			((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);


			for (int j = 0; j < 32; j += 2)
			{

				uint32_t rand = randbuffer[j / 2] % randmax;
				(hashbuffer + shift)[rand] = __ldg(&(hashbuffer + shift)[randmax + j]);
				(hashbuffer + shift)[rand + 1] = __ldg(&(hashbuffer + shift)[randmax + j + 1]);
				(hashbuffer + shift)[rand + 2] = __ldg(&(hashbuffer + shift)[randmax + j + 2]);
				(hashbuffer + shift)[rand + 3] = __ldg(&(hashbuffer + shift)[randmax + j + 3]);
			}

		} // main loop

} 
}
__global__ __launch_bounds__(256, 1) void pluck_gpu_hash_v50(int threads, uint32_t startNonce, uint32_t *nonceVector)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
 
		int shift = SHIFT * thread; //uint32_t

		for (int i = 5; i < HASH_MEMORY - 1; i++)
		{
			uint32_t randmax = i*32-4;
			uint32_t randseed[16];
			uint32_t randbuffer[16];  
			uint32_t joint[16];
			uint8 Buffbuffer[2];
            
			((uint8*)randseed)[0] = __ldg8(&(hashbuffer + shift)[32*i-64]);
			((uint8*)randseed)[1] = __ldg8(&(hashbuffer + shift)[32*i-32]);           	
			

                Buffbuffer[0] = __ldg8(&(hashbuffer + shift)[32*i - 128]);
				Buffbuffer[1] = __ldg8(&(hashbuffer + shift)[32*i - 96]);
				((uint16*)randseed)[0] ^= ((uint16*)Buffbuffer)[0];
 
			((uint16*)randbuffer)[0]= xor_salsa8(((uint16*)randseed)[0]);

			((uint8*)joint)[0] = __ldg8(&(hashbuffer + shift)[(i-1)<<5]);

#pragma unroll
			for (int j = 0; j < 8; j++)
			{
				uint32_t rand = randbuffer[j] % (randmax - 32); 
				joint[j+8] = __ldgtoint_unaligned(&(hashbuffer + shift)[rand]); 
			}
	
			uint8 truc =  sha256_64(joint);
			((uint8*)(hashbuffer + shift))[i] = truc;
			((uint8*)randseed)[0] = ((uint8*)joint)[0];
			((uint8*)randseed)[1] = truc;


	 ((uint16*)randseed)[0] ^= ((uint16*)Buffbuffer)[0];


 ((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);


			for (int j = 0; j < 32; j += 2)
			{
 
				uint32_t rand = randbuffer[j / 2] % randmax;
				
				(hashbuffer+shift)[rand] =       __ldg(&(hashbuffer+shift)[randmax+j]);
				(hashbuffer + shift)[rand + 1] = __ldg(&(hashbuffer + shift)[randmax + j + 1]);
				(hashbuffer + shift)[rand + 2] = __ldg(&(hashbuffer + shift)[randmax + j + 2]);
				(hashbuffer + shift)[rand + 3] = __ldg(&(hashbuffer + shift)[randmax + j + 3]);
			}
 
		} // main loop

		uint32_t outbuf =  __ldgtoint(&(hashbuffer + shift)[28]);

		if (outbuf <= pTarget[7]) {
			nonceVector[0] = nonce;
		}

	}
}

__global__ __launch_bounds__(128, 3) void pluck_gpu_hash0(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		int shift = SHIFT * thread; //uint32_t
		((uint8*)(hashbuffer + shift))[0] = sha256_80(nonce);
		((uint8*)(hashbuffer + shift))[1] = make_uint8(0, 0, 0, 0, 0, 0, 0, 0);
		for (int i = 2; i < 5; i++)
		{
			uint32_t randmax = i * 32 - 4;
			uint32_t randseed[16];
			uint32_t randbuffer[16];
			uint32_t joint[16];
			uint8 Buffbuffer[2];

			((uint8*)randseed)[0] = __ldg8(&(hashbuffer + shift)[32 * i - 64]);
			((uint8*)randseed)[1] = __ldg8(&(hashbuffer + shift)[32 * i - 32]);



			((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);

			//			((uint8*)joint)[0] = __ldg8(&(hashbuffer + shift)[(i - 1) << 5]);
			((uint8*)joint)[0] = ((uint8*)randseed)[1];
#pragma unroll
			for (int j = 0; j < 8; j++)
			{
				uint32_t rand = randbuffer[j] % (randmax - 32);
				joint[j + 8] = __ldgtoint_unaligned(&(hashbuffer + shift)[rand]);
			}

			uint8 truc = sha256_64(joint);
			((uint8*)(hashbuffer + shift))[i] = truc;
			((uint8*)randseed)[0] = ((uint8*)joint)[0];
			((uint8*)randseed)[1] = truc;


			((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);


			for (int j = 0; j < 32; j += 2)
			{

				uint32_t rand = randbuffer[j / 2] % randmax;
				(hashbuffer + shift)[rand] = __ldg(&(hashbuffer + shift)[randmax + j]);
				(hashbuffer + shift)[rand + 1] = __ldg(&(hashbuffer + shift)[randmax + j + 1]);
				(hashbuffer + shift)[rand + 2] = __ldg(&(hashbuffer + shift)[randmax + j + 2]);
				(hashbuffer + shift)[rand + 3] = __ldg(&(hashbuffer + shift)[randmax + j + 3]);
			}

		} // main loop

	}
}
__global__ __launch_bounds__(128, 3) void pluck_gpu_hash(int threads, uint32_t startNonce, uint32_t *nonceVector)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		int shift = SHIFT * thread; //uint32_t

		for (int i = 5; i < HASH_MEMORY - 1; i++)
		{
			uint32_t randmax = i * 32 - 4;
			uint32_t randseed[16];
			uint32_t randbuffer[16];
			uint32_t joint[16];
			uint8 Buffbuffer[2];

			((uint8*)randseed)[0] = __ldg8(&(hashbuffer + shift)[32 * i - 64]);
			((uint8*)randseed)[1] = __ldg8(&(hashbuffer + shift)[32 * i - 32]);


			Buffbuffer[0] = __ldg8(&(hashbuffer + shift)[32 * i - 128]);
			Buffbuffer[1] = __ldg8(&(hashbuffer + shift)[32 * i - 96]);
			((uint16*)randseed)[0] ^= ((uint16*)Buffbuffer)[0];

			((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);

			((uint8*)joint)[0] = __ldg8(&(hashbuffer + shift)[(i - 1) << 5]);

#pragma unroll
			for (int j = 0; j < 8; j++)
			{
				uint32_t rand = randbuffer[j] % (randmax - 32);
				joint[j + 8] = __ldgtoint_unaligned(&(hashbuffer + shift)[rand]);
			}

			uint8 truc = sha256_64(joint);
			((uint8*)(hashbuffer + shift))[i] = truc;
			((uint8*)randseed)[0] = ((uint8*)joint)[0];
			((uint8*)randseed)[1] = truc;


			((uint16*)randseed)[0] ^= ((uint16*)Buffbuffer)[0];


			((uint16*)randbuffer)[0] = xor_salsa8(((uint16*)randseed)[0]);


			for (int j = 0; j < 32; j += 2)
			{

				uint32_t rand = randbuffer[j / 2] % randmax;

				(hashbuffer + shift)[rand] = __ldg(&(hashbuffer + shift)[randmax + j]);
				(hashbuffer + shift)[rand + 1] = __ldg(&(hashbuffer + shift)[randmax + j + 1]);
				(hashbuffer + shift)[rand + 2] = __ldg(&(hashbuffer + shift)[randmax + j + 2]);
				(hashbuffer + shift)[rand + 3] = __ldg(&(hashbuffer + shift)[randmax + j + 3]);
			}

		} // main loop

		uint32_t outbuf = __ldgtoint(&(hashbuffer + shift)[28]);

		if (outbuf <= pTarget[7]) {
			nonceVector[0] = nonce;
		}

	}
}


void pluck_cpu_init(int thr_id, int threads, uint32_t* hash)
{
    
	cudaMemcpyToSymbol(hashbuffer, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_PlNonce[thr_id], sizeof(uint32_t)); 

} 


__host__ uint32_t pluck_cpu_hash(int thr_id, int threads, uint32_t startNounce,  int order)
{
	uint32_t result[8] = {0xffffffff};
	cudaMemset(d_PlNonce[thr_id], 0xffffffff, sizeof(uint32_t));

 
	const int threadsperblock = 128;
	
 
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	dim3 grid50((threads + 256 - 1) / 256);
	dim3 block50(256);

	if (compute_version[thr_id]==50) {
	pluck_gpu_hash0_v50 << <grid50, block50 >> >(threads, startNounce);
	pluck_gpu_hash_v50  << <grid50, block50 >> >(threads, startNounce, d_PlNonce[thr_id]);
	}
	else {
		pluck_gpu_hash0 << <grid, block >> >(threads, startNounce);
		pluck_gpu_hash << <grid, block >> >(threads, startNounce, d_PlNonce[thr_id]);
	}

	MyStreamSynchronize(NULL, order, thr_id);
	cudaMemcpy(&result[thr_id], d_PlNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

return result[thr_id];
}



__host__ void pluck_setBlockTarget(const void *pdata, const void *ptarget)
{
	unsigned char PaddedMessage[80];
	memcpy(PaddedMessage, pdata, 80);
	cudaMemcpyToSymbol(c_data, PaddedMessage, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(pTarget, ptarget, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}