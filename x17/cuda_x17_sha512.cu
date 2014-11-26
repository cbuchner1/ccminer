/**
 * sha512 djm34
 * (cleaned by tpruvot)
 */

/*
 * sha-512 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014 djm34
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
 * @author phm <phm@inbox.com>
 */
#include <stdio.h>

#define USE_SHARED 1

#include "cuda_helper.h"

#define SWAP64(u64) cuda_swab64(u64)

#define SPH_ROTL32(x, n)  SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define SPH_ROTR32(x, n)  SPH_ROTL32(x, (32 - (n)))

#define SPH_T32(x)  ((x) & SPH_C32(0xFFFFFFFF))
#define SPH_T64(x)  ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))

static __constant__ uint64_t H_512[8];

static const uint64_t H512[8] = {
	SPH_C64(0x6A09E667F3BCC908), SPH_C64(0xBB67AE8584CAA73B),
	SPH_C64(0x3C6EF372FE94F82B), SPH_C64(0xA54FF53A5F1D36F1),
	SPH_C64(0x510E527FADE682D1), SPH_C64(0x9B05688C2B3E6C1F),
	SPH_C64(0x1F83D9ABFB41BD6B), SPH_C64(0x5BE0CD19137E2179)
};
static __constant__ uint64_t K_512[80];

static const uint64_t K512[80] = {
	SPH_C64(0x428A2F98D728AE22), SPH_C64(0x7137449123EF65CD),
	SPH_C64(0xB5C0FBCFEC4D3B2F), SPH_C64(0xE9B5DBA58189DBBC),
	SPH_C64(0x3956C25BF348B538), SPH_C64(0x59F111F1B605D019),
	SPH_C64(0x923F82A4AF194F9B), SPH_C64(0xAB1C5ED5DA6D8118),
	SPH_C64(0xD807AA98A3030242), SPH_C64(0x12835B0145706FBE),
	SPH_C64(0x243185BE4EE4B28C), SPH_C64(0x550C7DC3D5FFB4E2),
	SPH_C64(0x72BE5D74F27B896F), SPH_C64(0x80DEB1FE3B1696B1),
	SPH_C64(0x9BDC06A725C71235), SPH_C64(0xC19BF174CF692694),
	SPH_C64(0xE49B69C19EF14AD2), SPH_C64(0xEFBE4786384F25E3),
	SPH_C64(0x0FC19DC68B8CD5B5), SPH_C64(0x240CA1CC77AC9C65),
	SPH_C64(0x2DE92C6F592B0275), SPH_C64(0x4A7484AA6EA6E483),
	SPH_C64(0x5CB0A9DCBD41FBD4), SPH_C64(0x76F988DA831153B5),
	SPH_C64(0x983E5152EE66DFAB), SPH_C64(0xA831C66D2DB43210),
	SPH_C64(0xB00327C898FB213F), SPH_C64(0xBF597FC7BEEF0EE4),
	SPH_C64(0xC6E00BF33DA88FC2), SPH_C64(0xD5A79147930AA725),
	SPH_C64(0x06CA6351E003826F), SPH_C64(0x142929670A0E6E70),
	SPH_C64(0x27B70A8546D22FFC), SPH_C64(0x2E1B21385C26C926),
	SPH_C64(0x4D2C6DFC5AC42AED), SPH_C64(0x53380D139D95B3DF),
	SPH_C64(0x650A73548BAF63DE), SPH_C64(0x766A0ABB3C77B2A8),
	SPH_C64(0x81C2C92E47EDAEE6), SPH_C64(0x92722C851482353B),
	SPH_C64(0xA2BFE8A14CF10364), SPH_C64(0xA81A664BBC423001),
	SPH_C64(0xC24B8B70D0F89791), SPH_C64(0xC76C51A30654BE30),
	SPH_C64(0xD192E819D6EF5218), SPH_C64(0xD69906245565A910),
	SPH_C64(0xF40E35855771202A), SPH_C64(0x106AA07032BBD1B8),
	SPH_C64(0x19A4C116B8D2D0C8), SPH_C64(0x1E376C085141AB53),
	SPH_C64(0x2748774CDF8EEB99), SPH_C64(0x34B0BCB5E19B48A8),
	SPH_C64(0x391C0CB3C5C95A63), SPH_C64(0x4ED8AA4AE3418ACB),
	SPH_C64(0x5B9CCA4F7763E373), SPH_C64(0x682E6FF3D6B2B8A3),
	SPH_C64(0x748F82EE5DEFB2FC), SPH_C64(0x78A5636F43172F60),
	SPH_C64(0x84C87814A1F0AB72), SPH_C64(0x8CC702081A6439EC),
	SPH_C64(0x90BEFFFA23631E28), SPH_C64(0xA4506CEBDE82BDE9),
	SPH_C64(0xBEF9A3F7B2C67915), SPH_C64(0xC67178F2E372532B),
	SPH_C64(0xCA273ECEEA26619C), SPH_C64(0xD186B8C721C0C207),
	SPH_C64(0xEADA7DD6CDE0EB1E), SPH_C64(0xF57D4F7FEE6ED178),
	SPH_C64(0x06F067AA72176FBA), SPH_C64(0x0A637DC5A2C898A6),
	SPH_C64(0x113F9804BEF90DAE), SPH_C64(0x1B710B35131C471B),
	SPH_C64(0x28DB77F523047D84), SPH_C64(0x32CAAB7B40C72493),
	SPH_C64(0x3C9EBE0A15C9BEBC), SPH_C64(0x431D67C49C100D4C),
	SPH_C64(0x4CC5D4BECB3E42B6), SPH_C64(0x597F299CFC657E2A),
	SPH_C64(0x5FCB6FAB3AD6FAEC), SPH_C64(0x6C44198C4A475817)
};


#define SHA3_STEP(ord,r,i) { \
		uint64_t T1, T2; \
		int a = 8-ord; \
		T1 = SPH_T64(r[(7+a)%8] + BSG5_1(r[(4+a)%8]) + CH(r[(4+a)%8], r[(5+a)%8], r[(6+a)%8]) + K_512[i] + W[i]); \
		T2 = SPH_T64(BSG5_0(r[(0+a)%8]) + MAJ(r[(0+a)%8], r[(1+a)%8], r[(2+a)%8])); \
		r[(3+a)%8] = SPH_T64(r[(3+a)%8] + T1); \
		r[(7+a)%8] = SPH_T64(T1 + T2); \
	}

#define SHA3_STEP2(truc,ord,r,i) { \
		uint64_t T1, T2; \
		int a = 8-ord; \
		T1 = Tone(truc,r,W,a,i); \
		T2 = SPH_T64(BSG5_0(r[(0+a)%8]) + MAJ(r[(0+a)%8], r[(1+a)%8], r[(2+a)%8])); \
		r[(3+a)%8] = SPH_T64(r[(3+a)%8] + T1); \
		r[(7+a)%8] = SPH_T64(T1 + T2); \
	}
//#define BSG5_0(x)      (ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39))
#define BSG5_0(x)        xor3(ROTR64(x, 28),ROTR64(x, 34),ROTR64(x, 39))

//#define BSG5_1(x)      (ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41))
#define BSG5_1(x)      xor3(ROTR64(x, 14),ROTR64(x, 18),ROTR64(x, 41))

//#define SSG5_0(x)      (ROTR64(x, 1) ^ ROTR64(x, 8) ^ SPH_T64((x) >> 7))
#define SSG5_0(x)      xor3(ROTR64(x, 1),ROTR64(x, 8),shr_t64(x,7))

//#define SSG5_1(x)      (ROTR64(x, 19) ^ ROTR64(x, 61) ^ SPH_T64((x) >> 6))
#define SSG5_1(x)      xor3(ROTR64(x, 19),ROTR64(x, 61),shr_t64(x,6))

//#define CH(X, Y, Z)    ((((Y) ^ (Z)) & (X)) ^ (Z))
#define CH(x, y, z)    xandx(x,y,z)
//#define MAJ(X, Y, Z)   (((X) & (Y)) | (((X) | (Y)) & (Z)))
#define MAJ(x, y, z)   andor(x,y,z)

__device__ __forceinline__
uint64_t Tone(const uint64_t* sharedMemory, uint64_t r[8], uint64_t W[80], uint32_t a, uint32_t i)
{
	uint64_t h = r[(7+a)%8];
	uint64_t e = r[(4+a)%8];
	uint64_t f = r[(5+a)%8];
	uint64_t g = r[(6+a)%8];
	//uint64_t BSG51 = ROTR64(e, 14) ^ ROTR64(e, 18) ^ ROTR64(e, 41);
	uint64_t BSG51 = xor3(ROTR64(e, 14),ROTR64(e, 18),ROTR64(e, 41));
	//uint64_t CHl     = (((f) ^ (g)) & (e)) ^ (g);
	uint64_t CHl = xandx(e,f,g);
	uint64_t result = SPH_T64(h+BSG51+CHl+sharedMemory[i]+W[i]);
	return result;
}

__global__
void x17_sha512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);
		int hashPosition = nounce - startNounce;
		uint32_t *inpHash = (uint32_t*)&g_hash[8 * hashPosition];
		union {
			uint8_t h1[64];
			uint32_t h4[16];
			uint64_t h8[8];
		} hash;

		#pragma unroll
		for (int i=0;i<16;i++) {
			hash.h4[i]= inpHash[i];
		}
		uint64_t W[80];
		uint64_t r[8];

		#pragma unroll 71
		for (int i=9;i<80;i++) {
			W[i]=0;
		}

		#pragma unroll
		for (int i = 0; i < 8; i ++) {
			W[i] = SWAP64(hash.h8[i]);
			r[i] = H_512[i];
		}

		W[8] = 0x8000000000000000;
		W[15]= 0x0000000000000200;

		#pragma unroll 64
		for (int i = 16; i < 80; i ++)
			W[i] = SPH_T64(SSG5_1(W[i - 2]) + W[i - 7]
			     + SSG5_0(W[i - 15]) + W[i - 16]);

		#pragma unroll 10
		for (int i = 0; i < 80; i += 8) {
			#pragma unroll 8
			for (int ord=0;ord<8;ord++) {
				SHA3_STEP2(K_512,ord,r,i+ord);
			}
		}

		#pragma unroll 8
		for (int i = 0; i < 8; i++) {
			r[i] = SPH_T64(r[i] + H_512[i]);
		}

		#pragma unroll 8
		for(int i=0;i<8;i++) {
			hash.h8[i] = SWAP64(r[i]);
		}

		#pragma unroll 16
		for (int u = 0; u < 16; u ++) {
			inpHash[u] = hash.h4[u];
		}
	}
}

__host__
void x17_sha512_cpu_init(int thr_id, int threads)
{
	cudaMemcpyToSymbol(K_512,K512,80*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(H_512,H512,sizeof(H512),0, cudaMemcpyHostToDevice);
}

__host__
void x17_sha512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size =0;
	x17_sha512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	MyStreamSynchronize(NULL, order, thr_id);
}
