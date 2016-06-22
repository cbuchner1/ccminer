/**
 * Blake-256 Decred 180-Bytes input Cuda Kernel (Tested on SM 5/5.2)
 *
 * Tanguy Pruvot - Feb 2016
 *
 * Merged 8-round blake (XVC) tweaks
 * Further improved by: ~2.72%
 * Alexis Provos - Jun 2016
 */

#include <stdint.h>
#include <memory.h>
#include <miner.h>

extern "C" {
#include <sph/sph_blake.h>
}

/* threads per block */
#define TPB 768
#define NPT 192
#define maxResults 8
/* max count of found nonces in one call */
#define NBN 2

/* hash by cpu with blake 256 */
extern "C" void decred_hash(void *output, const void *input)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 180);
	sph_blake256_close(&ctx, output);
}

#include <cuda_helper.h>

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#define atomicInc(p, max) (*p)
#endif

__constant__ uint32_t c_m[3];
__constant__ uint32_t _ALIGN(8)  c_h[2];
__constant__ uint32_t _ALIGN(32) c_v[16];
__constant__ uint32_t _ALIGN(32) c_x[90];

/* Buffers of candidate nonce(s) */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

__device__ __forceinline__
uint32_t ROR8(const uint32_t a) {
	return __byte_perm(a, 0, 0x0321);
}

__device__ __forceinline__
uint32_t ROL16(const uint32_t a) {
	return __byte_perm(a, 0, 0x1032);
}

__device__ __forceinline__
uint32_t xor3x(uint32_t a, uint32_t b, uint32_t c) {
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
#else
	result = a^b^c;
#endif
	return result;
}

#define GSn(a,b,c,d,x,y) { \
	v[a]+= x + v[b]; \
	v[d] = ROL16(v[d] ^ v[a]); \
	v[c]+= v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a]+= y + v[b]; \
	v[d] = ROR8(v[d] ^ v[a]); \
	v[c]+= v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

#define GSn3(a,b,c,d,x,y, a1,b1,c1,d1,x1,y1, a2,b2,c2,d2,x2,y2) { \
	v[ a]+= x + v[ b];                  v[a1]+= x1 + v[b1];                 v[a2]+= x2 + v[b2];\
	v[ d] = ROL16(v[ d] ^ v[ a]);       v[d1] = ROL16(v[d1] ^ v[a1]);       v[d2] = ROL16(v[d2] ^ v[a2]);\
	v[ c]+= v[ d];                      v[c1]+= v[d1];                      v[c2]+= v[d2];\
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);  v[b1] = ROTR32(v[b1] ^ v[c1], 12);  v[b2] = ROTR32(v[b2] ^ v[c2], 12);\
	v[ a]+= y + v[ b];                  v[a1]+= y1 + v[b1];                 v[a2]+= y2 + v[b2];\
	v[ d] = ROR8(v[ d] ^ v[ a]);        v[d1] = ROR8(v[d1] ^ v[a1]);        v[d2] = ROR8(v[d2] ^ v[a2]);\
	v[ c]+= v[ d];                      v[c1]+= v[d1];                      v[c2]+= v[d2];\
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);   v[b1] = ROTR32(v[b1] ^ v[c1], 7);   v[b2] = ROTR32(v[b2] ^ v[c2], 7);\
}

#define GSn4(a,b,c,d,x,y, a1,b1,c1,d1,x1,y1, a2,b2,c2,d2,x2,y2, a3,b3,c3,d3,x3,y3) { \
	v[ a]+= x + v[ b];                  v[a1]+= x1 + v[b1];                 v[a2]+= x2 + v[b2];                 v[a3]+= x3 + v[b3]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);       v[d1] = ROL16(v[d1] ^ v[a1]);       v[d2] = ROL16(v[d2] ^ v[a2]);       v[d3] = ROL16(v[d3] ^ v[a3]); \
	v[ c]+= v[ d];                      v[c1]+= v[d1];                      v[c2]+= v[d2];                      v[c3]+= v[d3]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);  v[b1] = ROTR32(v[b1] ^ v[c1], 12);  v[b2] = ROTR32(v[b2] ^ v[c2], 12);  v[b3] = ROTR32(v[b3] ^ v[c3], 12); \
	v[ a]+= y + v[ b];                  v[a1]+= y1 + v[b1];                 v[a2]+= y2 + v[b2];                 v[a3]+= y3 + v[b3]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);        v[d1] = ROR8(v[d1] ^ v[a1]);        v[d2] = ROR8(v[d2] ^ v[a2]);        v[d3] = ROR8(v[d3] ^ v[a3]); \
	v[ c]+= v[ d];                      v[c1]+= v[d1];                      v[c2]+= v[d2];                      v[c3]+= v[d3]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);   v[b1] = ROTR32(v[b1] ^ v[c1], 7);   v[b2] = ROTR32(v[b2] ^ v[c2], 7);   v[b3] = ROTR32(v[b3] ^ v[c3], 7); \
}

__global__ __launch_bounds__(TPB,1)
void decred_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce)
{
	      uint64_t m3       = startNonce + blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t step     = gridDim.x * blockDim.x;
	const uint64_t maxNonce = startNonce + threads;

	const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	uint32_t v[16];
	uint32_t m[16];

	#pragma unroll
	for(int i=0;i<3;i++) {
		m[i] = c_m[i];
	}
	m[13] = 0x80000001;
	m[15] = 0x000005a0;

	const uint32_t    m130 = z[12] ^ m[13];
	const uint32_t    m131 = m[13] ^ z[ 6];
	const uint32_t    m132 = z[15] ^ m[13];
	const uint32_t    m133 = z[ 3] ^ m[13];
	const uint32_t    m134 = z[ 4] ^ m[13];
	const uint32_t    m135 = z[14] ^ m[13];
	const uint32_t    m136 = m[13] ^ z[11];
	const uint32_t    m137 = m[13] ^ z[ 7];
	const uint32_t    m138 = m[13] ^ z[ 0];

	volatile uint32_t m150 = z[14] ^ m[15];
	volatile uint32_t m151 = z[ 9] ^ m[15];
	volatile uint32_t m152 = m[15] ^ z[13];
	volatile uint32_t m153 = m[15] ^ z[ 8];
	const uint32_t    m154 = z[10] ^ m[15];
	const uint32_t    m155 = z[ 1] ^ m[15];
	const uint32_t    m156 = m[15] ^ z[ 4];
	const uint32_t    m157 = z[ 6] ^ m[15];
	const uint32_t    m158 = m[15] ^ z[11];

	const uint32_t    h7   = c_h[ 0];

	for( ; m3<maxNonce ; m3+=step) {

		m[ 3] = m3;

		#pragma unroll 16
		for(int i=0; i<16; i++) {
			v[i] = c_v[i];
		}

		uint32_t xors[16];
		uint32_t i = 0;

		// round 1 {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } partial
		xors[ 5] = z[2] ^ m[3];
		xors[ 9] = c_x[i++]; xors[10] = c_x[i++];
		xors[11] = z[15];
		xors[12] = c_x[i++]; xors[13] = c_x[i++];
		xors[14] = m130;
		xors[15] = m150;

		v[ 1] += xors[ 5];			v[13] = ROR8(v[13] ^ v[1]);
		v[ 9] += v[13];				v[ 5] = ROTR32(v[5] ^ v[9], 7);
		v[ 0] += v[5];				v[15] = ROL16(v[15] ^ v[0]);
		v[10] += v[15];				v[ 5] = ROTR32(v[5] ^ v[10], 12);
		v[ 0] += xors[12] + v[5];	v[15] = ROR8(v[15] ^ v[0]);
		v[10] += v[15];				v[ 5] = ROTR32(v[5] ^ v[10], 7);

		GSn3(1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 2 { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
		xors[ 0] = z[10]; xors[ 1] = c_x[i++]; xors[ 2] = c_x[i++]; xors[ 3] = m131;
		xors[ 8] = m[ 1]^z[12]; xors[ 9] = m[ 0]^z[ 2]; xors[10] = c_x[i++]; xors[11] = c_x[i++];
		xors[ 4] = c_x[i++]; xors[ 5] = c_x[i++]; xors[ 6] = m151; xors[ 7] = c_x[i++];
		xors[12] = c_x[i++]; xors[13] = z[ 0]^m[ 2]; xors[14] = c_x[i++]; xors[15] = z[ 5]^m[ 3];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 3 { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }
		xors[ 0] = c_x[i++]; xors[ 1] = c_x[i++]; xors[ 2] = c_x[i++]; xors[ 3] = m152;
		xors[ 8] = c_x[i++]; xors[ 9] = m[ 3]^z[ 6]; xors[10] = c_x[i++]; xors[11] = c_x[i++];
		xors[ 4] = c_x[i++]; xors[ 5] = z[12]^m[ 0]; xors[ 6] = z[ 5]^m[ 2]; xors[ 7] = m132;
		xors[12] = z[10]; xors[13] = c_x[i++]; xors[14] = z[ 7]^m[ 1]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 4 { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 }
		xors[ 0] = c_x[i++]; xors[ 1] = m[ 3]^z[ 1]; xors[ 2] = m130; xors[ 3] = c_x[i++];
		xors[ 8] = m[ 2]^z[ 6]; xors[ 9] = c_x[i++]; xors[10] = c_x[i++]; xors[11] = m153;
		xors[ 4] = c_x[i++]; xors[ 5] = z[ 3]^m[ 1]; xors[ 6] = c_x[i++]; xors[ 7] = z[11];
		xors[12] = c_x[i++]; xors[13] = c_x[i++]; xors[14] = z[ 4]^m[ 0]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 5 { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 }
		xors[ 0] = c_x[i++]; xors[ 1] = c_x[i++]; xors[ 2] = m[ 2]^z[ 4]; xors[ 3] = c_x[i++];
		xors[ 8] = z[ 1]; xors[ 9] = c_x[i++]; xors[10] = c_x[i++]; xors[11] = m[ 3]^z[13];
		xors[ 4] = z[ 9]^m[ 0]; xors[ 5] = c_x[i++]; xors[ 6] = c_x[i++]; xors[ 7] = m154;
		xors[12] = z[14]^m[ 1]; xors[13] = c_x[i++]; xors[14] = c_x[i++]; xors[15] = m133;

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 6 { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
		xors[ 0] = m[ 2]^z[12]; xors[ 1] = c_x[i++]; xors[ 2] = m[ 0]^z[11]; xors[ 3] = c_x[i++];
		xors[ 8] = c_x[i++]; xors[ 9] = c_x[i++]; xors[10] = m150; xors[11] = m[ 1]^z[ 9];
		xors[ 4] = c_x[i++]; xors[ 5] = c_x[i++]; xors[ 6] = c_x[i++]; xors[ 7] = z[ 8]^m[ 3];
		xors[12] = m134; xors[13] = c_x[i++]; xors[14] = z[15]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 7 { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 }
		xors[ 0] = c_x[i++]; xors[ 1] = m[ 1]^z[15]; xors[ 2] = z[13]; xors[ 3] = c_x[i++];
		xors[ 8] = m[ 0]^z[ 7]; xors[ 9] = c_x[i++]; xors[10] = c_x[i++]; xors[11] = c_x[i++];
		xors[ 4] = c_x[i++]; xors[ 5] = m155; xors[ 6] = m135; xors[ 7] = c_x[i++];
		xors[12] = c_x[i++]; xors[13] = z[ 6]^m[ 3]; xors[14] = z[ 9]^m[ 2]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 8 { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 }
		xors[ 0] = m136; xors[ 1] = c_x[i++]; xors[ 2] = c_x[i++]; xors[ 3] = m[ 3]^z[ 9];
		xors[ 8] = c_x[i++]; xors[ 9] = m156; xors[10] = c_x[i++]; xors[11] = m[ 2]^z[10];
		xors[ 4] = c_x[i++]; xors[ 5] = z[ 7]; xors[ 6] = z[12]^m[ 1]; xors[ 7] = c_x[i++];
		xors[12] = z[ 5]^m[ 0]; xors[13] = c_x[i++]; xors[14] = c_x[i++]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 9 { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 }
		xors[ 0] = c_x[i++]; xors[ 1] = z[ 9]; xors[ 2] = c_x[i++]; xors[ 3] = m[ 0]^z[ 8];
		xors[ 8] = c_x[i++]; xors[ 9] = m137; xors[10] = m[ 1]^z[ 4]; xors[11] = c_x[i++];
		xors[ 4] = m157; xors[ 5] = c_x[i++]; xors[ 6] = z[11]^m[ 3]; xors[ 7] = c_x[i++];
		xors[12] = z[12]^m[ 2]; xors[13] = c_x[i++]; xors[14] = c_x[i++]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 10 { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 }
		xors[ 0] = c_x[i++]; xors[ 1] = c_x[i++]; xors[ 2] = c_x[i++]; xors[ 3] = m[ 1]^z[ 5];
		xors[ 8] = m158; xors[ 9] = c_x[i++]; xors[10] = m[ 3]^z[12]; xors[11] = m138;
		xors[ 4] = z[10]^m[ 2]; xors[ 5] = c_x[i++]; xors[ 6] = c_x[i++]; xors[ 7] = c_x[i++];
		xors[12] = c_x[i++]; xors[13] = z[ 9]; xors[14] = c_x[i++]; xors[15] = z[13]^m[ 0];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 11 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
		xors[ 0] = m[ 0]^z[ 1]; xors[ 1] = m[ 2]^z[ 3]; xors[ 2] = c_x[i++]; xors[ 3] = c_x[i++];
		xors[ 8] = c_x[i++]; xors[ 9] = c_x[ 0]; xors[10] = c_x[ 1]; xors[11] = z[15];
		xors[ 4] = z[ 0]^m[ 1]; xors[ 5] = z[ 2]^m[ 3]; xors[ 6] = c_x[i++]; xors[ 7] = c_x[i++];
		xors[12] = c_x[ 2]; xors[13] = c_x[ 3]; xors[14] = m130; xors[15] = m150;

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);
		//i=90
		i=4;

		// round 12 { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
		xors[ 0] = z[10]; xors[ 1] = c_x[i++]; xors[ 2] = c_x[i++]; xors[ 3] = m131;
		xors[ 8] = m[ 1]^z[12]; xors[ 9] = m[ 0]^z[ 2]; xors[10] = c_x[i++]; xors[11] = c_x[i++];
		xors[ 4] = c_x[i++]; xors[ 5] = c_x[i++]; xors[ 6] = m151; xors[ 7] = c_x[i++];
		xors[12] = c_x[i++]; xors[13] = z[ 0]^m[ 2]; xors[14] = c_x[i++]; xors[15] = z[ 5]^m[ 3];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 13 { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 }
		xors[ 0] = c_x[i++]; xors[ 1] = c_x[i++]; xors[ 2] = c_x[i++]; xors[ 3] = m152;
		xors[ 8] = c_x[i++]; xors[ 9] = m[ 3]^z[ 6]; xors[10] = c_x[i++]; xors[11] = c_x[i++];
		xors[ 4] = c_x[i++]; xors[ 5] = z[12]^m[ 0]; xors[ 6] = z[ 5]^m[ 2]; xors[ 7] = m132;
		xors[12] = z[10]; xors[13] = c_x[i++]; xors[14] = z[ 7]^m[ 1]; xors[15] = c_x[i++];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);
		GSn4(0, 5,10,15, xors[ 8], xors[12], 1, 6,11,12, xors[ 9], xors[13], 2, 7, 8,13, xors[10], xors[14], 3, 4, 9,14, xors[11], xors[15]);

		// round 14 { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 }
		xors[ 0] = c_x[i++]; xors[ 1] = m[ 3]^z[ 1]; xors[ 2] = m130; xors[ 3] = c_x[i++];
		xors[ 8] = m[ 2]^z[ 6]; i++; xors[10] = c_x[i++];
		xors[ 4] = c_x[i++]; xors[ 5] = z[ 3]^m[ 1]; xors[ 6] = c_x[i++]; xors[ 7] = z[11];
		xors[12] = c_x[i++]; xors[14] = z[ 4]^m[ 0];

		GSn4(0, 4, 8,12, xors[ 0], xors[ 4], 1, 5, 9,13, xors[ 1], xors[ 5], 2, 6,10,14, xors[ 2], xors[ 6], 3, 7,11,15, xors[ 3], xors[ 7]);

		v[ 0]+= xors[ 8] + v[ 5];
		v[ 2]+= xors[10] + v[ 7];
		v[15] = ROL16(v[15] ^ v[ 0]);
		v[13] = ROL16(v[13] ^ v[ 2]);
		v[10]+= v[15];
		v[ 8]+= v[13];
		v[ 5] = ROTR32(v[ 5] ^ v[10], 12);
		v[ 7] = ROTR32(v[ 7] ^ v[ 8], 12);
		v[ 0]+= xors[12] + v[ 5];
		v[ 2]+= xors[14] + v[ 7];
		v[15] = ROTR32(v[15] ^ v[ 0],1);
		v[13] = ROR8(v[13] ^ v[ 2]);
		v[ 8]+= v[13];
		if(xor3x(v[ 7],h7,v[ 8])==v[15]) {
			uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX)+1;
			if(pos < maxResults)
				resNonce[pos] = m[3];
			return;
		}
	}
}

__host__
void decred_cpu_setBlock_52(const int thr_id,const uint32_t *input, const uint32_t *pend)
{
	const uint32_t z[16] = {
		0x243F6A88UL, 0x85A308D3UL, 0x13198A2EUL, 0x03707344UL,
		0xA4093822UL, 0x299F31D0UL, 0x082EFA98UL, 0xEC4E6C89UL,
		0x452821E6UL, 0x38D01377UL, 0xBE5466CFUL, 0x34E90C6CUL,
		0xC0AC29B7UL, 0xC97C50DDUL, 0x3F84D5B5UL, 0xB5470917UL
	};

	sph_u32 _ALIGN(64) v[16];
	sph_u32 _ALIGN(64) m[16];
	sph_u32 _ALIGN(64) h[ 2];

	sph_blake256_context ctx;
	sph_blake256_set_rounds(14);
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 128);

	v[ 0] = ctx.H[0];  v[ 1] = ctx.H[1];
	v[ 2] = ctx.H[2];  v[ 3] = ctx.H[3];
	v[ 4] = ctx.H[4];  v[ 5] = ctx.H[5];
	v[ 8] = ctx.H[6];  v[12] = swab32(input[35]);
	v[13] = ctx.H[7];

	// pre swab32
	m[ 0] = swab32(input[32]);	m[ 1] = swab32(input[33]);
	m[ 2] = swab32(input[34]);	m[ 3] = 0;
	m[ 4] = swab32(input[36]);	m[ 5] = swab32(input[37]);
	m[ 6] = swab32(input[38]);	m[ 7] = swab32(input[39]);
	m[ 8] = swab32(input[40]);	m[ 9] = swab32(input[41]);
	m[10] = swab32(input[42]);	m[11] = swab32(input[43]);
	m[12] = swab32(input[44]);	m[13] = 0x80000001;
	m[14] = 0;
	m[15] = 0x000005a0;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_m, m, 3*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

	h[ 0] = v[ 8];
	h[ 1] = v[13];

	v[ 0]+= (m[ 0] ^ z[1]) + v[ 4];
	v[12]  = SPH_ROTR32(z[4] ^ SPH_C32(0x5A0) ^ v[ 0], 16);

	v[ 8] = z[0]+v[12];
	v[ 4] = SPH_ROTR32(v[ 4] ^ v[ 8], 12);
	v[ 0]+= (m[ 1] ^ z[0]) + v[ 4];
	v[12] = SPH_ROTR32(v[12] ^ v[ 0],8);
	v[ 8]+= v[12];
	v[ 4] = SPH_ROTR32(v[ 4] ^ v[ 8], 7);

	v[ 1]+= (m[ 2] ^ z[3]) + v[ 5];
	v[13] = SPH_ROTR32((z[5] ^ SPH_C32(0x5A0)) ^ v[ 1], 16);
	v[ 9] = z[1]+v[13];
	v[ 5] = SPH_ROTR32(v[ 5] ^ v[ 9], 12);
	v[ 1]+= v[ 5]; //+nonce ^ ...

	v[ 2]+= (m[ 4] ^ z[5]) + h[ 0];
	v[14] = SPH_ROTR32(z[6] ^ v[ 2],16);
	v[10] = z[2] + v[14];
	v[ 6] = SPH_ROTR32(h[ 0] ^ v[10], 12);
	v[ 2]+= (m[ 5] ^ z[4]) + v[ 6];
	v[14] = SPH_ROTR32(v[14] ^ v[ 2], 8);
	v[10]+= v[14];
	v[ 6] = SPH_ROTR32(v[ 6] ^ v[10], 7);

	v[ 3]+= (m[ 6] ^ z[7]) + h[ 1];
	v[15] = SPH_ROTR32(z[7] ^ v[ 3],16);
	v[11] = z[3] + v[15];
	v[ 7] = SPH_ROTR32(h[ 1] ^ v[11], 12);
	v[ 3]+= (m[ 7] ^ z[6]) + v[ 7];
	v[15] = SPH_ROTR32(v[15] ^ v[ 3],8);
	v[11]+= v[15];
	v[ 7] = SPH_ROTR32(v[11] ^ v[ 7], 7);
	v[ 0]+= m[ 8] ^ z[9];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_v, v,16*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

	h[ 0] = SPH_ROTL32(h[ 1], 7); //align the rotation with v[7] v[15];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_h, h, 1*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

	uint32_t x[90];
	int i=0;

	x[i++] = m[10]^z[11];  x[i++] = m[12]^z[13];  x[i++] = m[ 9]^z[ 8];  x[i++] = z[10]^m[11];  x[i++] = m[ 4]^z[ 8];  x[i++] = m[ 9]^z[15];  x[i++] = m[11]^z[ 7];  x[i++] = m[ 5]^z[ 3];
	x[i++] = z[14]^m[10];  x[i++] = z[ 4]^m[ 8];  x[i++] = z[13]^m[ 6];  x[i++] = z[ 1]^m[12];  x[i++] = z[11]^m[ 7];  x[i++] = m[11]^z[ 8];  x[i++] = m[12]^z[ 0];  x[i++] = m[ 5]^z[ 2];
	x[i++] = m[10]^z[14];  x[i++] = m[ 7]^z[ 1];  x[i++] = m[ 9]^z[ 4];  x[i++] = z[11]^m[ 8];  x[i++] = z[ 3]^m[ 6];  x[i++] = z[ 9]^m[ 4];  x[i++] = m[ 7]^z[ 9];  x[i++] = m[11]^z[14];
	x[i++] = m[ 5]^z[10];  x[i++] = m[ 4]^z[ 0];  x[i++] = z[ 7]^m[ 9];  x[i++] = z[13]^m[12];  x[i++] = z[ 2]^m[ 6];  x[i++] = z[ 5]^m[10];  x[i++] = z[15]^m[ 8];  x[i++] = m[ 9]^z[ 0];
	x[i++] = m[ 5]^z[ 7];  x[i++] = m[10]^z[15];  x[i++] = m[11]^z[12];  x[i++] = m[ 6]^z[ 8];  x[i++] = z[ 5]^m[ 7];  x[i++] = z[ 2]^m[ 4];  x[i++] = z[11]^m[12];  x[i++] = z[ 6]^m[ 8];
	x[i++] = m[ 6]^z[10];  x[i++] = m[ 8]^z[ 3];  x[i++] = m[ 4]^z[13];  x[i++] = m[ 7]^z[ 5];  x[i++] = z[ 2]^m[12];  x[i++] = z[ 6]^m[10];  x[i++] = z[ 0]^m[11];  x[i++] = z[ 7]^m[ 5];
	x[i++] = z[ 1]^m[ 9];  x[i++] = m[12]^z[ 5];  x[i++] = m[ 4]^z[10];  x[i++] = m[ 6]^z[ 3];  x[i++] = m[ 9]^z[ 2];  x[i++] = m[ 8]^z[11];  x[i++] = z[12]^m[ 5];  x[i++] = z[ 4]^m[10];
	x[i++] = z[ 0]^m[ 7];  x[i++] = z[ 8]^m[11];  x[i++] = m[ 7]^z[14];  x[i++] = m[12]^z[ 1];  x[i++] = m[ 5]^z[ 0];  x[i++] = m[ 8]^z[ 6];  x[i++] = z[13]^m[11];  x[i++] = z[ 3]^m[ 9];
	x[i++] = z[15]^m[ 4];  x[i++] = z[ 8]^m[ 6];  x[i++] = z[ 2]^m[10];  x[i++] = m[ 6]^z[15];  x[i++] = m[11]^z[ 3];  x[i++] = m[12]^z[ 2];  x[i++] = m[10]^z[ 5];  x[i++] = z[14]^m[ 9];
	x[i++] = z[ 0]^m[ 8];  x[i++] = z[13]^m[ 7];  x[i++] = z[ 1]^m[ 4];  x[i++] = z[10]^m[ 5];  x[i++] = m[10]^z[ 2];  x[i++] = m[ 8]^z[ 4];  x[i++] = m[ 7]^z[ 6];  x[i++] = m[ 9]^z[14];
	x[i++] = z[ 8]^m[ 4];  x[i++] = z[ 7]^m[ 6];  x[i++] = z[ 1]^m[ 5];  x[i++] = z[15]^m[11];  x[i++] = z[ 3]^m[12];  x[i++] = m[ 4]^z[ 5];  x[i++] = m[ 6]^z[ 7];  x[i++] = m[ 8]^z[ 9];
	x[i++] = z[ 4]^m[ 5];  x[i++] = z[ 6]^m[ 7];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_x, x, i*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

/* ############################################################################################################################### */

static bool init[MAX_GPUS] = { 0 };

// nonce position is different in decred
#define DCR_NONCE_OFT32 35

extern "C" int scanhash_decred(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[48];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = &pdata[DCR_NONCE_OFT32];

	const uint32_t first_nonce = *pnonce;
	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 29 : 25;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);

	if (opt_benchmark) {
		ptarget[6] = swab32(0xff);
	}
	if (!init[thr_id]){
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], maxResults*sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], maxResults*sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}
	memcpy(endiandata, pdata, 180);

	decred_cpu_setBlock_52(thr_id, endiandata, &pdata[32]);
	h_resNonce[thr_id][0] = 1;

	do {
		if (h_resNonce[thr_id][0])
			cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

		// GPU HASH
		decred_gpu_hash_nonce <<<grid, block>>> (throughput, (*pnonce), d_resNonce[thr_id]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0])
		{
			cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], (h_resNonce[thr_id][0]+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);

			for(uint32_t i=1; i <= h_resNonce[thr_id][0]; i++)
			{
				uint32_t vhash64[8];
				be32enc(&endiandata[DCR_NONCE_OFT32], h_resNonce[thr_id][i]);
				decred_hash(vhash64, endiandata);
				if (vhash64[6] <= ptarget[6] && fulltest(vhash64, ptarget))
				{
					int rc = 1;
					work_set_target_ratio(work, vhash64);
					*hashes_done = (*pnonce) - first_nonce + throughput;
					work->nonces[0] = swab32(h_resNonce[thr_id][i]);
					// search for another nonce
					for(uint32_t j=i+1; j <= h_resNonce[thr_id][0]; j++)
					{
						be32enc(&endiandata[DCR_NONCE_OFT32], h_resNonce[thr_id][j]);
						decred_hash(vhash64, endiandata);
						if (vhash64[6] <= ptarget[6] && fulltest(vhash64, ptarget)){
							work->nonces[1] = swab32(h_resNonce[thr_id][j]);
							if(!opt_quiet)
								gpulog(LOG_NOTICE, thr_id, "second nonce found %u / %08x - %u / %08x", i, work->nonces[0], j, work->nonces[1]);
							if(bn_hash_target_ratio(vhash64, ptarget) > work->shareratio) {
								work_set_target_ratio(work, vhash64);
								xchg(work->nonces[1], work->nonces[0]);
							}
							rc = 2;
							break;
						}
					}
					*pnonce = work->nonces[0];
					return rc;
				}
			}
		}
		*pnonce += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + (*pnonce));

	*hashes_done = (*pnonce) - first_nonce;
	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return 0;
}

// cleanup
extern "C" void free_decred(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
