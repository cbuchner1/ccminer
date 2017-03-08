/**
 * Based on the SPH implementation of blake2s
 * Provos Alexis - 2016
 */

#include "miner.h"

#include <string.h>
#include <stdint.h>

#include "sph/blake2s.h"
#include "sph/sph_types.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

#include "cuda_helper.h"

#ifdef __CUDA_ARCH__

__device__ __forceinline__
uint32_t ROR8(const uint32_t a) {
	return __byte_perm(a, 0, 0x0321);
}

__device__ __forceinline__
uint32_t ROL16(const uint32_t a) {
	return __byte_perm(a, 0, 0x1032);
}

#else
#define ROR8(u)  (u >> 8)
#define ROL16(u) (u << 16)
#endif

__device__ __forceinline__
uint32_t xor3x(uint32_t a, uint32_t b, uint32_t c)
{
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
#else
	result = a^b^c;
#endif
	return result;
}

static const uint32_t blake2s_IV[8] = {
	0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
	0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
};

static const uint8_t blake2s_sigma[10][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
};

#define G(r,i,a,b,c,d) \
	do { \
		a = a + b + m[blake2s_sigma[r][2*i+0]]; \
		d = SPH_ROTR32(d ^ a, 16); \
		c = c + d; \
		b = SPH_ROTR32(b ^ c, 12); \
		a = a + b + m[blake2s_sigma[r][2*i+1]]; \
		d = SPH_ROTR32(d ^ a, 8); \
		c = c + d; \
		b = SPH_ROTR32(b ^ c, 7); \
	} while(0)
#define ROUND(r)  \
	do { \
		G(r,0,v[0],v[4],v[ 8],v[12]); \
		G(r,1,v[1],v[5],v[ 9],v[13]); \
		G(r,2,v[2],v[6],v[10],v[14]); \
		G(r,3,v[3],v[7],v[11],v[15]); \
		G(r,4,v[0],v[5],v[10],v[15]); \
		G(r,5,v[1],v[6],v[11],v[12]); \
		G(r,6,v[2],v[7],v[ 8],v[13]); \
		G(r,7,v[3],v[4],v[ 9],v[14]); \
	} while(0)

extern "C" void blake2s_hash(void *output, const void *input)
{
	uint32_t m[16];
	uint32_t v[16];
	uint32_t h[8];

	uint32_t *in = (uint32_t*)input;
//	COMPRESS
	for(int i = 0; i < 16; ++i )
		m[i] = in[i];

	h[0] = 0x01010020 ^ blake2s_IV[0];
	h[1] = blake2s_IV[1];
	h[2] = blake2s_IV[2];
	h[3] = blake2s_IV[3];
	h[4] = blake2s_IV[4];
	h[5] = blake2s_IV[5];
	h[6] = blake2s_IV[6];
	h[7] = blake2s_IV[7];

	for(int i = 0; i < 8; ++i )
		v[i] = h[i];

	v[ 8] = blake2s_IV[0];		v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];		v[11] = blake2s_IV[3];
	v[12] = 64 ^ blake2s_IV[4];	v[13] = blake2s_IV[5];
	v[14] = blake2s_IV[6];		v[15] = blake2s_IV[7];

	ROUND( 0 ); ROUND( 1 );
	ROUND( 2 ); ROUND( 3 );
	ROUND( 4 ); ROUND( 5 );
	ROUND( 6 ); ROUND( 7 );
	ROUND( 8 ); ROUND( 9 );

	for(size_t i = 0; i < 8; ++i)
		h[i] ^= v[i] ^ v[i + 8];

//	COMPRESS
	m[0] = in[16]; m[1] = in[17];
	m[2] = in[18]; m[3] = in[19];
	for(size_t i = 4; i < 16; ++i)
		m[i] = 0;

	for(size_t i = 0; i < 8; ++i)
		v[i] = h[i];

	v[ 8] = blake2s_IV[0];		v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];		v[11] = blake2s_IV[3];
	v[12] = 0x50 ^ blake2s_IV[4];	v[13] = blake2s_IV[5];
	v[14] = ~blake2s_IV[6];		v[15] = blake2s_IV[7];

	ROUND( 0 ); ROUND( 1 );
	ROUND( 2 ); ROUND( 3 );
	ROUND( 4 ); ROUND( 5 );
	ROUND( 6 ); ROUND( 7 );
	ROUND( 8 ); ROUND( 9 );

	for(size_t i = 0; i < 8; ++i)
		h[i] ^= v[i] ^ v[i + 8];

	memcpy(output, h, 32);
}

#define TPB 1024
#define NPT 256
#define maxResults 16
#define NBN 1

__constant__ uint32_t _ALIGN(32) midstate[20];

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

#define GS4(a,b,c,d,e,f,a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2,a3,b3,c3,d3,e3,f3){ \
	a += b + e;		a1+= b1 + e1;	 	a2+= b2 + e2;		a3+= b3 + e3; \
	d  = ROL16( d ^ a);	d1 = ROL16(d1 ^ a1);	d2 = ROL16(d2 ^ a2);	d3 = ROL16(d3 ^ a3); \
	c +=d; 			c1+=d1;			c2+=d2;			c3+=d3;\
	b  = ROTR32(b ^ c, 12); b1 = ROTR32(b1^c1, 12);	b2 = ROTR32(b2^c2, 12);	b3 = ROTR32(b3^c3, 12); \
	a += b + f;		a1+= b1 + f1;		a2+= b2 + f2;		a3+= b3 + f3; \
	d  = ROR8(d ^ a);	d1 = ROR8(d1^a1);	d2 = ROR8(d2^a2);	d3 = ROR8(d3^a3); \
	c  += d;		c1 += d1;		c2 += d2;		c3 += d3;\
	b  = ROTR32(b ^ c, 7);	b1 = ROTR32(b1^c1, 7);	b2 = ROTR32(b2^c2, 7);	b3 = ROTR32(b3^c3, 7); \
}

__global__ __launch_bounds__(TPB,1)
void blake2s_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint32_t ptarget7)
{
	const uint32_t step = gridDim.x * blockDim.x;

	uint32_t m[ 3];
	uint32_t v[16];

	m[0] = midstate[16];
	m[1] = midstate[17];
	m[2] = midstate[18];

	const uint32_t h7 = midstate[19];

	for(uint32_t thread   = blockDim.x * blockIdx.x + threadIdx.x ; thread <threads; thread+=step){
		#pragma unroll
		for(int i=0;i<16;i++){
			v[ i] = midstate[ i];
		}

		uint32_t nonce = cuda_swab32(startNonce + thread);
//		Round( 0 );
		v[ 1] += nonce;
		v[13] = ROR8(v[13] ^ v[ 1]);
		v[ 9] += v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[ 9], 7);

		v[ 1]+= v[ 6];
		v[ 0]+= v[ 5];

		v[12] = ROL16(v[12] ^ v[ 1]);
		v[13] = ROL16(v[13] ^ v[ 2]);
		v[15] = ROL16(v[15] ^ v[ 0]);

		v[11]+= v[12];				v[ 8]+= v[13];				v[ 9]+= v[14];				v[10]+= v[15];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 12);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 12);	v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 1]+= v[ 6];				v[ 2]+= v[ 7];				v[ 3]+= v[ 4];				v[ 0]+= v[ 5];
		v[12] = ROR8(v[12] ^ v[ 1]);		v[13] = ROR8(v[13] ^ v[ 2]);		v[14] = ROR8(v[14] ^ v[ 3]);		v[15] = ROR8(v[15] ^ v[ 0]);
		v[11]+= v[12]; 				v[ 8]+= v[13];				v[ 9]+= v[14];				v[10]+= v[15];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 7);	v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 7);	v[ 5] = ROTR32(v[ 5] ^ v[10], 7);

		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],m[ 1],0,	v[ 1],v[ 6],v[11],v[12],m[ 0],m[ 2],	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],0,nonce);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,m[ 0],	v[ 2],v[ 6],v[10],v[14],0,m[ 2],	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,0,	v[ 1],v[ 6],v[11],v[12],nonce,0,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 1],	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],nonce,m[ 1],	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],m[ 2],0,	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 0],	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,m[ 0],	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],m[ 2],0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 1],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],nonce,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],m[ 2],0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],m[ 0],0,	v[ 3],v[ 7],v[11],v[15],0,nonce);
		GS4(v[ 0],v[ 5],v[10],v[15],0,0,	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],m[ 1],0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],m[ 1],0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],m[ 0],0,	v[ 1],v[ 6],v[11],v[12],0,nonce,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 2],	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,m[ 1],	v[ 3],v[ 7],v[11],v[15],nonce,0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 0],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],m[ 2],0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,nonce,	v[ 3],v[ 7],v[11],v[15],m[ 0],0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 2],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],m[ 1],0,	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,m[ 2],	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],m[ 1],0);

//		GS(9,4,v[ 0],v[ 5],v[10],v[15]);
		v[ 0] += v[ 5];
		v[ 2] += v[ 7] + nonce;
		v[15] = ROL16(v[15] ^ v[ 0]);
		v[13] = ROL16(v[13] ^ v[ 2]);
		v[10] += v[15];
		v[ 8] += v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
		v[ 0] += v[ 5];
		v[ 2] += v[ 7];
		v[15] = ROR8(v[15] ^ v[ 0]);
		v[13] = ROR8(v[13] ^ v[ 2]);

		v[ 8] += v[13];
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 7);

		if (xor3x(h7,v[7],v[15]) <= ptarget7){
			uint32_t pos = atomicInc(&resNonce[0],0xffffffff)+1;
			if(pos < maxResults)
				resNonce[pos] = nonce;
			return;
		}
	}
}

__global__ __launch_bounds__(TPB,1)
void blake2s_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce)
{
	const uint32_t step = gridDim.x * blockDim.x;

	uint32_t m[ 3];
	uint32_t v[16];

	m[0] = midstate[16];
	m[1] = midstate[17];
	m[2] = midstate[18];

	const uint32_t h7 = midstate[19];

	for(uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x ; thread <threads; thread+=step)
	{
		#pragma unroll
		for(int i=0;i<16;i++){
			v[ i] = midstate[ i];
		}

		uint32_t nonce = cuda_swab32(startNonce+thread);

//		Round( 0 );
		v[ 1] += nonce;
		v[13] = ROR8(v[13] ^ v[ 1]);
		v[ 9] += v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[ 9], 7);

		v[ 1]+= v[ 6];
		v[ 0]+= v[ 5];

		v[13] = ROL16(v[13] ^ v[ 2]);		v[12] = ROL16(v[12] ^ v[ 1]);		v[15] = ROL16(v[15] ^ v[ 0]);

		v[ 8]+= v[13];				v[11]+= v[12];				v[ 9]+= v[14];				v[10]+= v[15];
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);	v[ 6] = ROTR32(v[ 6] ^ v[11], 12);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 12);	v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 2]+= v[ 7];				v[ 1]+= v[ 6];				v[ 3]+= v[ 4];				v[ 0]+= v[ 5];
		v[13] = ROR8(v[13] ^ v[ 2]);		v[12] = ROR8(v[12] ^ v[ 1]);		v[14] = ROR8(v[14] ^ v[ 3]);		v[15] = ROR8(v[15] ^ v[ 0]);
		v[ 8]+= v[13];				v[11]+= v[12];				v[ 9]+= v[14];				v[10]+= v[15];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 7);	v[ 7] = ROTR32(v[ 7] ^ v[8], 7);	v[ 4] = ROTR32(v[ 4] ^ v[ 9], 7);	v[ 5] = ROTR32(v[ 5] ^ v[10], 7);

		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],m[ 1],0,	v[ 1],v[ 6],v[11],v[12],m[ 0],m[ 2],	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],0,nonce);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,m[ 0],	v[ 2],v[ 6],v[10],v[14],0,m[ 2],	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,0,	v[ 1],v[ 6],v[11],v[12],nonce,0,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 1],	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],nonce,m[ 1],	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],m[ 2],0,	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 0],	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,m[ 0],	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],m[ 2],0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 1],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],nonce,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],m[ 2],0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],m[ 0],0,	v[ 3],v[ 7],v[11],v[15],0,nonce);
		GS4(v[ 0],v[ 5],v[10],v[15],0,0,	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],m[ 1],0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],m[ 1],0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],0,0);
		GS4(v[ 0],v[ 5],v[10],v[15],m[ 0],0,	v[ 1],v[ 6],v[11],v[12],0,nonce,	v[ 2],v[ 7],v[ 8],v[13],0,m[ 2],	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,m[ 1],	v[ 3],v[ 7],v[11],v[15],nonce,0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 0],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],0,0,	v[ 3],v[ 4],v[ 9],v[14],m[ 2],0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,0,	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,nonce,	v[ 3],v[ 7],v[11],v[15],m[ 0],0);
		GS4(v[ 0],v[ 5],v[10],v[15],0,m[ 2],	v[ 1],v[ 6],v[11],v[12],0,0,	v[ 2],v[ 7],v[ 8],v[13],m[ 1],0,	v[ 3],v[ 4],v[ 9],v[14],0,0);
		GS4(v[ 0],v[ 4],v[ 8],v[12],0,m[ 2],	v[ 1],v[ 5],v[ 9],v[13],0,0,	v[ 2],v[ 6],v[10],v[14],0,0,	v[ 3],v[ 7],v[11],v[15],m[ 1],0);

		v[ 0] += v[ 5];
		v[ 2] += v[ 7] + nonce;
		v[15] = ROL16(v[15] ^ v[ 0]);
		v[13] = ROL16(v[13] ^ v[ 2]);
		v[10] += v[15];
		v[ 8] += v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
		v[ 0] += v[ 5];
		v[ 2] += v[ 7];
		v[15] = ROTR32(v[15] ^ v[ 0],1);
		v[13] = ROR8(v[13] ^ v[ 2]);

		v[ 8] += v[13];

		if(xor3x(v[ 7],h7,v[ 8])==v[15]){
			uint32_t pos = atomicInc(&resNonce[0],0xffffffff)+1;
			if(pos < maxResults)
				resNonce[pos]=nonce;
			return;
		}
	}
}

static void blake2s_setBlock(const uint32_t* input,const uint32_t ptarget7)
{
	uint32_t _ALIGN(64) m[16];
	uint32_t _ALIGN(64) v[16];
	uint32_t _ALIGN(64) h[21];

//	COMPRESS
	for(int i = 0; i < 16; ++i )
		m[i] = input[i];

	h[0] = 0x01010020 ^ blake2s_IV[0];
	h[1] = blake2s_IV[1];
	h[2] = blake2s_IV[2]; h[3] = blake2s_IV[3];
	h[4] = blake2s_IV[4]; h[5] = blake2s_IV[5];
	h[6] = blake2s_IV[6]; h[7] = blake2s_IV[7];

	for(int i = 0; i < 8; ++i )
		v[i] = h[i];

	v[ 8] = blake2s_IV[0];		v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];		v[11] = blake2s_IV[3];
	v[12] = 64 ^ blake2s_IV[4];	v[13] = blake2s_IV[5];
	v[14] = blake2s_IV[6];		v[15] = blake2s_IV[7];

	ROUND( 0 ); ROUND( 1 );
	ROUND( 2 ); ROUND( 3 );
	ROUND( 4 ); ROUND( 5 );
	ROUND( 6 ); ROUND( 7 );
	ROUND( 8 ); ROUND( 9 );

	for(int i = 0; i < 8; ++i )
		h[i] ^= v[i] ^ v[i + 8];

	h[16] = input[16];
	h[17] = input[17];
	h[18] = input[18];

	h[ 8] = 0x6A09E667; h[ 9] = 0xBB67AE85;
	h[10] = 0x3C6EF372; h[11] = 0xA54FF53A;
	h[12] = 0x510E522F; h[13] = 0x9B05688C;
	h[14] =~0x1F83D9AB; h[15] = 0x5BE0CD19;

	h[ 0]+= h[ 4] + h[16];
	h[12] = SPH_ROTR32(h[12] ^ h[ 0],16);
	h[ 8]+= h[12];
	h[ 4] = SPH_ROTR32(h[ 4] ^ h[ 8],12);
	h[ 0]+= h[ 4] + h[17];
	h[12] = SPH_ROTR32(h[12] ^ h[ 0],8);
	h[ 8]+= h[12];
	h[ 4] = SPH_ROTR32(h[ 4] ^ h[ 8],7);

	h[ 1]+= h[ 5] + h[18];
	h[13] = SPH_ROTR32(h[13] ^ h[ 1], 16);
	h[ 9]+= h[13];
	h[ 5] = ROTR32(h[ 5] ^ h[ 9], 12);

	h[ 2]+= h[ 6];
	h[14] = SPH_ROTR32(h[14] ^ h[ 2],16);
	h[10]+= h[14];
	h[ 6] = SPH_ROTR32(h[ 6] ^ h[10], 12);
	h[ 2]+= h[ 6];
	h[14] = SPH_ROTR32(h[14] ^ h[ 2],8);
	h[10]+= h[14];
	h[ 6] = SPH_ROTR32(h[ 6] ^ h[10], 7);

	h[19] = h[7]; //constant h[7] for nonce check

	h[ 3]+= h[ 7];
	h[15] = SPH_ROTR32(h[15] ^ h[ 3],16);
	h[11]+= h[15];
	h[ 7] = SPH_ROTR32(h[ 7] ^ h[11], 12);
	h[ 3]+= h[ 7];
	h[15] = SPH_ROTR32(h[15] ^ h[ 3],8);
	h[11]+= h[15];
	h[ 7] = SPH_ROTR32(h[ 7] ^ h[11], 7);

	h[ 1]+= h[ 5];
	h[ 3]+= h[ 4];
	h[14] = SPH_ROTR32(h[14] ^ h[ 3],16);

	h[ 2]+= h[ 7];
	if(ptarget7==0){
		h[19] = SPH_ROTL32(h[19],7); //align the rotation with v[7] v[15];
	}
	cudaMemcpyToSymbol(midstate, h, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake2s(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *resNonces;

	const uint32_t first_nonce = pdata[19];

	const int dev_id = device_map[thr_id];
	int rc = 0;
	int intensity = is_windows() ? 25 : 28;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], maxResults * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], maxResults * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}
	resNonces = h_resNonce[thr_id];

	for (int i=0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}
	blake2s_setBlock(endiandata,ptarget[7]);

	cudaMemset(d_resNonce[thr_id], 0x00, maxResults*sizeof(uint32_t));

	do {
		if(ptarget[7]) {
			blake2s_gpu_hash_nonce<<<grid, block>>>(throughput,pdata[19],d_resNonce[thr_id],ptarget[7]);
		} else {
			blake2s_gpu_hash_nonce<<<grid, block>>>(throughput,pdata[19],d_resNonce[thr_id]);
		}
		cudaMemcpy(resNonces, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if(resNonces[0])
		{
			cudaMemcpy(resNonces, d_resNonce[thr_id], maxResults*sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

			if(resNonces[0] >= maxResults) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", resNonces[0]);
				resNonces[0] = maxResults-1;
			}

			uint32_t vhashcpu[8];
			uint32_t nonce = sph_bswap32(resNonces[1]);
			be32enc(&endiandata[19], nonce);
			blake2s_hash(vhashcpu, endiandata);

			*hashes_done = pdata[19] - first_nonce + throughput;

			if(vhashcpu[6] <= ptarget[6] && fulltest(vhashcpu, ptarget))
			{
				work_set_target_ratio(work, vhashcpu);
				work->nonces[0] = nonce;
				rc = work->valid_nonces = 1;

				// search for 2nd best nonce
				for(uint32_t j=2; j <= resNonces[0]; j++)
				{
					nonce = sph_bswap32(resNonces[j]);
					be32enc(&endiandata[19], nonce);
					blake2s_hash(vhashcpu, endiandata);
					if(vhashcpu[6] <= ptarget[6] && fulltest(vhashcpu, ptarget))
					{
						gpulog(LOG_DEBUG, thr_id, "Multiple nonces: 1/%08x - %u/%08x", work->nonces[0], j, nonce);

						work->nonces[1] = nonce;
						if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio[0]) {
							work->shareratio[1] = work->shareratio[0];
							work->sharediff[1] = work->sharediff[0];
							xchg(work->nonces[1], work->nonces[0]);
							work_set_target_ratio(work, vhashcpu);
						} else if (work->valid_nonces == 1) {
							bn_set_target_ratio(work, vhashcpu, 1);
						}

						work->valid_nonces++;
						rc = 2;
						break;
					}
				}
				pdata[19] = max(work->nonces[0], work->nonces[1]); // next scan start
				return rc;
			} else if (vhashcpu[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", resNonces[0]);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + pdata[19]);

	*hashes_done = pdata[19] - first_nonce;

	return rc;
}

// cleanup
extern "C" void free_blake2s(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

