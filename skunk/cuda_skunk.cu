/**
 * skein + cube + fugue merged kernel, based on krnlx work
 *
 * based on alexis78 sib kernels, final touch by tpruvot
 */

#include <miner.h>
#include <cuda_vectors.h>
#include "skunk/skein_header.h"
#include <cuda_vector_uint2x4.h>

#define TPB 512

/* ************************ */
static __constant__ uint2 c_buffer[120]; // padded message (80 bytes + 72 bytes midstate + align)

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }

__device__ __forceinline__
static void rrounds(uint32_t *x){
	#pragma unroll 2
	for (int r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0], 7);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1], 7);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2], 7);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3], 7);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4], 7);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5], 7);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6], 7);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7], 7);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8], 7);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7);x[27] = x[27] + x[11];x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7);x[29] = x[29] + x[13];x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7);x[31] = x[31] + x[15];x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[ 0], x[ 8]);x[ 0] ^= x[16];x[ 8] ^= x[24];SWAP(x[ 1], x[ 9]);x[ 1] ^= x[17];x[ 9] ^= x[25];
		SWAP(x[ 2], x[10]);x[ 2] ^= x[18];x[10] ^= x[26];SWAP(x[ 3], x[11]);x[ 3] ^= x[19];x[11] ^= x[27];
		SWAP(x[ 4], x[12]);x[ 4] ^= x[20];x[12] ^= x[28];SWAP(x[ 5], x[13]);x[ 5] ^= x[21];x[13] ^= x[29];
		SWAP(x[ 6], x[14]);x[ 6] ^= x[22];x[14] ^= x[30];SWAP(x[ 7], x[15]);x[ 7] ^= x[23];x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]);
		SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[ 0]; x[ 0] = ROTL32(x[ 0],11);x[17] = x[17] + x[ 1];x[ 1] = ROTL32(x[ 1],11);
		x[18] = x[18] + x[ 2]; x[ 2] = ROTL32(x[ 2],11);x[19] = x[19] + x[ 3];x[ 3] = ROTL32(x[ 3],11);
		x[20] = x[20] + x[ 4]; x[ 4] = ROTL32(x[ 4],11);x[21] = x[21] + x[ 5];x[ 5] = ROTL32(x[ 5],11);
		x[22] = x[22] + x[ 6]; x[ 6] = ROTL32(x[ 6],11);x[23] = x[23] + x[ 7];x[ 7] = ROTL32(x[ 7],11);
		x[24] = x[24] + x[ 8]; x[ 8] = ROTL32(x[ 8],11);x[25] = x[25] + x[ 9];x[ 9] = ROTL32(x[ 9],11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10],11);x[27] = x[27] + x[11];x[11] = ROTL32(x[11],11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12],11);x[29] = x[29] + x[13];x[13] = ROTL32(x[13],11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14],11);x[31] = x[31] + x[15];x[15] = ROTL32(x[15],11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[ 0], x[ 4]); x[ 0] ^= x[16]; x[ 4] ^= x[20]; SWAP(x[ 1], x[ 5]); x[ 1] ^= x[17]; x[ 5] ^= x[21];
		SWAP(x[ 2], x[ 6]); x[ 2] ^= x[18]; x[ 6] ^= x[22]; SWAP(x[ 3], x[ 7]); x[ 3] ^= x[19]; x[ 7] ^= x[23];
		SWAP(x[ 8], x[12]); x[ 8] ^= x[24]; x[12] ^= x[28]; SWAP(x[ 9], x[13]); x[ 9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]);
		SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}

// fugue
static __constant__ const uint32_t c_S[16] = {
	0x8807a57e, 0xe616af75, 0xc5d3e4db, 0xac9ab027,
	0xd915f117, 0xb6eecc54, 0x06e8020b, 0x4a92efd1,
	0xaac6e2c9, 0xddb21398, 0xcae65838, 0x437f203f,
	0x25ea78e7, 0x951fddd6, 0xda6ed11d, 0xe13e3567
};

static __device__ uint32_t mixtab0[256] = {
	0x63633297, 0x7c7c6feb, 0x77775ec7, 0x7b7b7af7, 0xf2f2e8e5, 0x6b6b0ab7,	0x6f6f16a7, 0xc5c56d39,
	0x303090c0, 0x01010704, 0x67672e87, 0x2b2bd1ac, 0xfefeccd5, 0xd7d71371, 0xabab7c9a, 0x767659c3,
	0xcaca4005, 0x8282a33e, 0xc9c94909, 0x7d7d68ef, 0xfafad0c5, 0x5959947f, 0x4747ce07, 0xf0f0e6ed,
	0xadad6e82, 0xd4d41a7d, 0xa2a243be, 0xafaf608a, 0x9c9cf946, 0xa4a451a6, 0x727245d3, 0xc0c0762d,
	0xb7b728ea, 0xfdfdc5d9, 0x9393d47a, 0x2626f298, 0x363682d8, 0x3f3fbdfc, 0xf7f7f3f1, 0xcccc521d,
	0x34348cd0, 0xa5a556a2, 0xe5e58db9, 0xf1f1e1e9, 0x71714cdf, 0xd8d83e4d, 0x313197c4, 0x15156b54,
	0x04041c10, 0xc7c76331, 0x2323e98c, 0xc3c37f21, 0x18184860, 0x9696cf6e, 0x05051b14, 0x9a9aeb5e,
	0x0707151c, 0x12127e48, 0x8080ad36, 0xe2e298a5, 0xebeba781, 0x2727f59c, 0xb2b233fe, 0x757550cf,
	0x09093f24, 0x8383a43a, 0x2c2cc4b0, 0x1a1a4668, 0x1b1b416c, 0x6e6e11a3, 0x5a5a9d73, 0xa0a04db6,
	0x5252a553, 0x3b3ba1ec, 0xd6d61475, 0xb3b334fa, 0x2929dfa4, 0xe3e39fa1, 0x2f2fcdbc, 0x8484b126,
	0x5353a257, 0xd1d10169, 0x00000000, 0xededb599, 0x2020e080, 0xfcfcc2dd, 0xb1b13af2, 0x5b5b9a77,
	0x6a6a0db3, 0xcbcb4701, 0xbebe17ce, 0x3939afe4, 0x4a4aed33, 0x4c4cff2b, 0x5858937b, 0xcfcf5b11,
	0xd0d0066d, 0xefefbb91, 0xaaaa7b9e, 0xfbfbd7c1, 0x4343d217, 0x4d4df82f, 0x333399cc, 0x8585b622,
	0x4545c00f, 0xf9f9d9c9, 0x02020e08, 0x7f7f66e7, 0x5050ab5b, 0x3c3cb4f0, 0x9f9ff04a, 0xa8a87596,
	0x5151ac5f, 0xa3a344ba, 0x4040db1b, 0x8f8f800a, 0x9292d37e, 0x9d9dfe42, 0x3838a8e0, 0xf5f5fdf9,
	0xbcbc19c6, 0xb6b62fee, 0xdada3045, 0x2121e784, 0x10107040, 0xffffcbd1, 0xf3f3efe1, 0xd2d20865,
	0xcdcd5519, 0x0c0c2430, 0x1313794c, 0xececb29d, 0x5f5f8667, 0x9797c86a, 0x4444c70b, 0x1717655c,
	0xc4c46a3d, 0xa7a758aa, 0x7e7e61e3, 0x3d3db3f4, 0x6464278b, 0x5d5d886f, 0x19194f64, 0x737342d7,
	0x60603b9b, 0x8181aa32, 0x4f4ff627, 0xdcdc225d, 0x2222ee88, 0x2a2ad6a8, 0x9090dd76, 0x88889516,
	0x4646c903, 0xeeeebc95, 0xb8b805d6, 0x14146c50, 0xdede2c55, 0x5e5e8163, 0x0b0b312c, 0xdbdb3741,
	0xe0e096ad, 0x32329ec8, 0x3a3aa6e8, 0x0a0a3628, 0x4949e43f, 0x06061218, 0x2424fc90, 0x5c5c8f6b,
	0xc2c27825, 0xd3d30f61, 0xacac6986, 0x62623593, 0x9191da72, 0x9595c662, 0xe4e48abd, 0x797974ff,
	0xe7e783b1, 0xc8c84e0d, 0x373785dc, 0x6d6d18af, 0x8d8d8e02, 0xd5d51d79, 0x4e4ef123, 0xa9a97292,
	0x6c6c1fab, 0x5656b943, 0xf4f4fafd, 0xeaeaa085, 0x6565208f, 0x7a7a7df3, 0xaeae678e, 0x08083820,
	0xbaba0bde, 0x787873fb, 0x2525fb94, 0x2e2ecab8, 0x1c1c5470, 0xa6a65fae, 0xb4b421e6, 0xc6c66435,
	0xe8e8ae8d, 0xdddd2559, 0x747457cb, 0x1f1f5d7c, 0x4b4bea37, 0xbdbd1ec2, 0x8b8b9c1a, 0x8a8a9b1e,
	0x70704bdb, 0x3e3ebaf8, 0xb5b526e2, 0x66662983, 0x4848e33b, 0x0303090c, 0xf6f6f4f5, 0x0e0e2a38,
	0x61613c9f, 0x35358bd4, 0x5757be47, 0xb9b902d2, 0x8686bf2e, 0xc1c17129, 0x1d1d5374, 0x9e9ef74e,
	0xe1e191a9, 0xf8f8decd, 0x9898e556, 0x11117744, 0x696904bf, 0xd9d93949, 0x8e8e870e, 0x9494c166,
	0x9b9bec5a, 0x1e1e5a78, 0x8787b82a, 0xe9e9a989, 0xcece5c15, 0x5555b04f, 0x2828d8a0, 0xdfdf2b51,
	0x8c8c8906, 0xa1a14ab2, 0x89899212, 0x0d0d2334, 0xbfbf10ca, 0xe6e684b5, 0x4242d513, 0x686803bb,
	0x4141dc1f, 0x9999e252, 0x2d2dc3b4, 0x0f0f2d3c, 0xb0b03df6, 0x5454b74b, 0xbbbb0cda, 0x16166258
};

__device__ __forceinline__
uint32_t ROL8X(const uint32_t a){
	return __byte_perm(a, 0, 0x2103);
}
__device__ __forceinline__
uint32_t ROL16X(const uint32_t a){
	return __byte_perm(a, 0, 0x1032);
}
__device__ __forceinline__
uint32_t ROR8X(const uint32_t a){
	return __byte_perm(a, 0, 0x0321);
}

#define mixtab0(x) shared[0][x]
#define mixtab1(x) shared[1][x]
#define mixtab2(x) shared[2][x]
#define mixtab3(x) shared[3][x]

#define TIX4(q, x00, x01, x04, x07, x08, x22, x24, x27, x30) { \
		x22 ^= x00; \
		x00 = (q); \
		x08 ^= (q); \
		x01 ^= x24; \
		x04 ^= x27; \
		x07 ^= x30; \
	}

#define CMIX36(x00, x01, x02, x04, x05, x06, x18, x19, x20) { \
		x00 ^= x04; \
		x01 ^= x05; \
		x02 ^= x06; \
		x18 ^= x04; \
		x19 ^= x05; \
		x20 ^= x06; \
	}

__device__ __forceinline__
static void SMIX(const uint32_t shared[4][256], uint32_t &x0,uint32_t &x1,uint32_t &x2,uint32_t &x3){
	uint32_t c0 = mixtab0(__byte_perm(x0,0,0x4443));
	uint32_t r1 = mixtab1(__byte_perm(x0,0,0x4442));
	uint32_t r2 = mixtab2(__byte_perm(x0,0,0x4441));
	uint32_t r3 = mixtab3(__byte_perm(x0,0,0x4440));
	c0 = c0 ^ r1 ^ r2 ^ r3;
	uint32_t r0 = mixtab0(__byte_perm(x1,0,0x4443));
	uint32_t c1 = r0 ^ mixtab1(__byte_perm(x1,0,0x4442));
	uint32_t tmp = mixtab2(__byte_perm(x1,0,0x4441));
	c1 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x1,0,0x4440));
	c1 ^= tmp;
	r3 ^= tmp;
	uint32_t c2 = mixtab0(__byte_perm(x2,0,0x4443));
	r0 ^= c2;
	tmp = mixtab1(__byte_perm(x2,0,0x4442));
	c2 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x2,0,0x4441));
	c2 ^= tmp;
	tmp = mixtab3(__byte_perm(x2,0,0x4440));
	c2 ^= tmp;
	r3 ^= tmp;
	uint32_t c3 = mixtab0(__byte_perm(x3,0,0x4443));
	r0 ^= c3;
	tmp = mixtab1(__byte_perm(x3,0,0x4442));
	c3 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x3,0,0x4441));
	c3 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x3,0,0x4440));
	c3 ^= tmp;
	x0 = ((c0 ^ (r0 << 0)) & 0xFF000000) | ((c1 ^ (r1 << 0)) & 0x00FF0000) | ((c2 ^ (r2 << 0)) & 0x0000FF00) | ((c3 ^ (r3 << 0)) & 0x000000FF);
	x1 = ((c1 ^ (r0 << 8)) & 0xFF000000) | ((c2 ^ (r1 << 8)) & 0x00FF0000) | ((c3 ^ (r2 << 8)) & 0x0000FF00) | ((c0 ^ (r3 >>24)) & 0x000000FF);
	x2 = ((c2 ^ (r0 <<16)) & 0xFF000000) | ((c3 ^ (r1 <<16)) & 0x00FF0000) | ((c0 ^ (r2 >>16)) & 0x0000FF00) | ((c1 ^ (r3 >>16)) & 0x000000FF);
	x3 = ((c3 ^ (r0 <<24)) & 0xFF000000) | ((c0 ^ (r1 >> 8)) & 0x00FF0000) | ((c1 ^ (r2 >> 8)) & 0x0000FF00) | ((c2 ^ (r3 >> 8)) & 0x000000FF);
}

__device__
static void SMIX_LDG(const uint32_t shared[4][256], uint32_t &x0,uint32_t &x1,uint32_t &x2,uint32_t &x3){
	uint32_t c0 = __ldg(&mixtab0[__byte_perm(x0,0,0x4443)]);
	uint32_t r1 = mixtab1(__byte_perm(x0,0,0x4442));
	uint32_t r2 = mixtab2(__byte_perm(x0,0,0x4441));
	uint32_t r3 = mixtab3(__byte_perm(x0,0,0x4440));
	c0 = c0 ^ r1 ^ r2 ^ r3;
	uint32_t r0 = __ldg(&mixtab0[__byte_perm(x1,0,0x4443)]);
	uint32_t c1 = r0 ^ mixtab1(__byte_perm(x1,0,0x4442));
	uint32_t tmp = mixtab2(__byte_perm(x1,0,0x4441));
	c1 ^= tmp;
	r2 ^= tmp;
	tmp = mixtab3(__byte_perm(x1,0,0x4440));
	c1 ^= tmp;
	r3 ^= tmp;
	uint32_t c2 = __ldg(&mixtab0[__byte_perm(x2,0,0x4443)]);
	r0 ^= c2;
	tmp = mixtab1(__byte_perm(x2,0,0x4442));
	c2 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x2,0,0x4441));
	c2 ^= tmp;
	tmp = mixtab3(__byte_perm(x2,0,0x4440));
	c2 ^= tmp;
	r3 ^= tmp;
	uint32_t c3 = __ldg(&mixtab0[__byte_perm(x3,0,0x4443)]);
	r0 ^= c3;
	tmp = mixtab1(__byte_perm(x3,0,0x4442));
	c3 ^= tmp;
	r1 ^= tmp;
	tmp = mixtab2(__byte_perm(x3,0,0x4441));
	c3 ^= tmp;
	r2 ^= tmp;
	tmp = ROL8X(__ldg(&mixtab0[__byte_perm(x3,0,0x4440)]));
	c3 ^= tmp;
	x0 = ((c0 ^ (r0 << 0)) & 0xFF000000) | ((c1 ^ (r1 << 0)) & 0x00FF0000) | ((c2 ^ (r2 << 0)) & 0x0000FF00) | ((c3 ^ (r3 << 0)) & 0x000000FF);
	x1 = ((c1 ^ (r0 << 8)) & 0xFF000000) | ((c2 ^ (r1 << 8)) & 0x00FF0000) | ((c3 ^ (r2 << 8)) & 0x0000FF00) | ((c0 ^ (r3 >>24)) & 0x000000FF);
	x2 = ((c2 ^ (r0 <<16)) & 0xFF000000) | ((c3 ^ (r1 <<16)) & 0x00FF0000) | ((c0 ^ (r2 >>16)) & 0x0000FF00) | ((c1 ^ (r3 >>16)) & 0x000000FF);
	x3 = ((c3 ^ (r0 <<24)) & 0xFF000000) | ((c0 ^ (r1 >> 8)) & 0x00FF0000) | ((c1 ^ (r2 >> 8)) & 0x0000FF00) | ((c2 ^ (r3 >> 8)) & 0x000000FF);
}

#define mROR3 { \
	B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[32]; S[34] = S[31]; S[33] = S[30]; S[32] = S[29]; S[31] = S[28]; S[30] = S[27]; S[29] = S[26]; S[28] = S[25]; S[27] = S[24]; \
	S[26] = S[23]; S[25] = S[22]; S[24] = S[21]; S[23] = S[20]; S[22] = S[19]; S[21] = S[18]; S[20] = S[17]; S[19] = S[16]; S[18] = S[15]; \
	S[17] = S[14]; S[16] = S[13]; S[15] = S[12]; S[14] = S[11]; S[13] = S[10]; S[12] = S[ 9]; S[11] = S[ 8]; S[10] = S[ 7]; S[ 9] = S[ 6]; \
	S[ 8] = S[ 5]; S[ 7] = S[ 4]; S[ 6] = S[ 3]; S[ 5] = S[ 2]; S[ 4] = S[ 1]; S[ 3] = S[ 0]; S[ 2] = B[ 8]; S[ 1] = B[ 7]; S[ 0] = B[ 6]; \
	}

#define mROR8 { \
	B[ 1] = S[28], B[ 2] = S[29], B[ 3] = S[30], B[ 4] = S[31], B[ 5] = S[32], B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[27]; S[34] = S[26]; S[33] = S[25]; S[32] = S[24]; S[31] = S[23]; S[30] = S[22]; S[29] = S[21]; S[28] = S[20]; S[27] = S[19]; \
	S[26] = S[18]; S[25] = S[17]; S[24] = S[16]; S[23] = S[15]; S[22] = S[14]; S[21] = S[13]; S[20] = S[12]; S[19] = S[11]; S[18] = S[10]; \
	S[17] = S[ 9]; S[16] = S[ 8]; S[15] = S[ 7]; S[14] = S[ 6]; S[13] = S[ 5]; S[12] = S[ 4]; S[11] = S[ 3]; S[10] = S[ 2]; S[ 9] = S[ 1]; \
	S[ 8] = S[ 0]; S[ 7] = B[ 8]; S[ 6] = B[ 7]; S[ 5] = B[ 6]; S[ 4] = B[ 5]; S[ 3] = B[ 4]; S[ 2] = B[ 3]; S[ 1] = B[ 2]; S[ 0] = B[ 1]; \
	}

#define mROR9 { \
	B[ 0] = S[27], B[ 1] = S[28], B[ 2] = S[29], B[ 3] = S[30], B[ 4] = S[31], B[ 5] = S[32], B[ 6] = S[33], B[ 7] = S[34], B[ 8] = S[35]; \
	S[35] = S[26]; S[34] = S[25]; S[33] = S[24]; S[32] = S[23]; S[31] = S[22]; S[30] = S[21]; S[29] = S[20]; S[28] = S[19]; S[27] = S[18]; \
	S[26] = S[17]; S[25] = S[16]; S[24] = S[15]; S[23] = S[14]; S[22] = S[13]; S[21] = S[12]; S[20] = S[11]; S[19] = S[10]; S[18] = S[ 9]; \
	S[17] = S[ 8]; S[16] = S[ 7]; S[15] = S[ 6]; S[14] = S[ 5]; S[13] = S[ 4]; S[12] = S[ 3]; S[11] = S[ 2]; S[10] = S[ 1]; S[ 9] = S[ 0]; \
	S[ 8] = B[ 8]; S[ 7] = B[ 7]; S[ 6] = B[ 6]; S[ 5] = B[ 5]; S[ 4] = B[ 4]; S[ 3] = B[ 3]; S[ 2] = B[ 2]; S[ 1] = B[ 1]; S[ 0] = B[ 0]; \
	}

#define FUGUE512_3(x, y, z) { \
    TIX4(x, S[ 0], S[ 1], S[ 4], S[ 7], S[ 8], S[22], S[24], S[27], S[30]); \
    CMIX36(S[33], S[34], S[35], S[ 1], S[ 2], S[ 3], S[15], S[16], S[17]); \
    SMIX_LDG(shared, S[33], S[34], S[35], S[ 0]); \
    CMIX36(S[30], S[31], S[32], S[34], S[35], S[ 0], S[12], S[13], S[14]); \
    SMIX_LDG(shared, S[30], S[31], S[32], S[33]); \
    CMIX36(S[27], S[28], S[29], S[31], S[32], S[33], S[ 9], S[10], S[11]); \
    SMIX(shared, S[27], S[28], S[29], S[30]); \
    CMIX36(S[24], S[25], S[26], S[28], S[29], S[30], S[ 6], S[ 7], S[ 8]); \
    SMIX_LDG(shared, S[24], S[25], S[26], S[27]); \
    \
    TIX4(y, S[24], S[25], S[28], S[31], S[32], S[10], S[12], S[15], S[18]); \
    CMIX36(S[21], S[22], S[23], S[25], S[26], S[27], S[ 3], S[ 4], S[ 5]); \
    SMIX(shared, S[21], S[22], S[23], S[24]); \
    CMIX36(S[18], S[19], S[20], S[22], S[23], S[24], S[ 0], S[ 1], S[ 2]); \
    SMIX_LDG(shared, S[18], S[19], S[20], S[21]); \
    CMIX36(S[15], S[16], S[17], S[19], S[20], S[21], S[33], S[34], S[35]); \
    SMIX_LDG(shared, S[15], S[16], S[17], S[18]); \
    CMIX36(S[12], S[13], S[14], S[16], S[17], S[18], S[30], S[31], S[32]); \
    SMIX(shared, S[12], S[13], S[14], S[15]); \
    \
    TIX4(z, S[12], S[13], S[16], S[19], S[20], S[34], S[ 0], S[ 3], S[ 6]); \
    CMIX36(S[ 9], S[10], S[11], S[13], S[14], S[15], S[27], S[28], S[29]); \
    SMIX_LDG(shared, S[ 9], S[10], S[11], S[12]); \
    CMIX36(S[ 6], S[ 7], S[ 8], S[10], S[11], S[12], S[24], S[25], S[26]); \
    SMIX_LDG(shared, S[ 6], S[ 7], S[ 8], S[ 9]); \
    CMIX36(S[ 3], S[ 4], S[ 5], S[ 7], S[ 8], S[ 9], S[21], S[22], S[23]); \
    SMIX_LDG(shared, S[ 3], S[ 4], S[ 5], S[ 6]); \
    CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]); \
    SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]); \
	}

__global__
__launch_bounds__(TPB, 2)
void skunk_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ uint32_t shared[4][256];

	if(threadIdx.x<256) {
		const uint32_t tmp = __ldg(&mixtab0[threadIdx.x]);
		shared[0][threadIdx.x] = tmp;
		shared[1][threadIdx.x] = ROR8X(tmp);
		shared[2][threadIdx.x] = ROL16X(tmp);
		shared[3][threadIdx.x] = ROL8X(tmp);
	}
	__syncthreads();

	if (thread < threads)
	{
		// Skein
		uint2 h[9];
		uint2 t0, t1, t2;

		uint32_t nonce = cuda_swab32(startNounce + thread);
		uint2 nonce2 = make_uint2(c_buffer[0].x, nonce);

		uint2 p[8];
		p[1] = nonce2;

		h[0] = c_buffer[ 1];
		h[1] = c_buffer[ 2];
		h[2] = c_buffer[ 3];
		h[3] = c_buffer[ 4];
		h[4] = c_buffer[ 5];
		h[5] = c_buffer[ 6];
		h[6] = c_buffer[ 7];
		h[7] = c_buffer[ 8];
		h[8] = c_buffer[ 9];

		t0 = vectorize(0x50ull);
		t1 = vectorize(0xB000000000000000ull);
		t2 = t0^t1;

		p[ 1]=nonce2 + h[1];	p[ 0]= c_buffer[10] + p[ 1];
		p[ 2]=c_buffer[11];
		p[ 3]=c_buffer[12];
		p[ 4]=c_buffer[13];
		p[ 5]=c_buffer[14];
		p[ 6]=c_buffer[15];
		p[ 7]=c_buffer[16];

//		TFBIGMIX8e();
		p[1] = ROL2(p[1], 46) ^ p[0];
		p[2] += p[1];
		p[0] += p[3];
		p[1] = ROL2(p[1], 33) ^ p[2];
		p[3] = c_buffer[17] ^ p[0];
		p[4] += p[1];
		p[6] += p[3];
		p[0] += p[5];
		p[2] += p[7];
		p[1] = ROL2(p[1], 17) ^ p[4];
		p[3] = ROL2(p[3], 49) ^ p[6];
		p[5] = c_buffer[18] ^ p[0];
		p[7] = c_buffer[19] ^ p[2];
		p[6] += p[1];
		p[0] += p[7];
		p[2] += p[5];
		p[4] += p[3];
		p[1] = ROL2(p[1], 44) ^ p[6];
		p[7] = ROL2(p[7], 9) ^ p[0];
		p[5] = ROL2(p[5], 54) ^ p[2];
		p[3] = ROR8(p[3]) ^ p[4];

		p[ 0]+=h[1];	p[ 1]+=h[2];	p[ 2]+=h[3];	p[ 3]+=h[4];	p[ 4]+=h[5];	p[ 5]+=c_buffer[20];	p[ 7]+=c_buffer[21];	p[ 6]+=c_buffer[22];
		TFBIGMIX8o();
		p[ 0]+=h[2];	p[ 1]+=h[3];	p[ 2]+=h[4];	p[ 3]+=h[5];	p[ 4]+=h[6];	p[ 5]+=c_buffer[22];	p[ 7]+=c_buffer[23];	p[ 6]+=c_buffer[24];
		TFBIGMIX8e();
		p[ 0]+=h[3];	p[ 1]+=h[4];	p[ 2]+=h[5];	p[ 3]+=h[6];	p[ 4]+=h[7];	p[ 5]+=c_buffer[24];	p[ 7]+=c_buffer[25];	p[ 6]+=c_buffer[26];
		TFBIGMIX8o();
		p[ 0]+=h[4];	p[ 1]+=h[5];	p[ 2]+=h[6];	p[ 3]+=h[7];	p[ 4]+=h[8];	p[ 5]+=c_buffer[26];	p[ 7]+=c_buffer[27];	p[ 6]+=c_buffer[28];
		TFBIGMIX8e();
		p[ 0]+=h[5];	p[ 1]+=h[6];	p[ 2]+=h[7];	p[ 3]+=h[8];	p[ 4]+=h[0];	p[ 5]+=c_buffer[28];	p[ 7]+=c_buffer[29];	p[ 6]+=c_buffer[30];
		TFBIGMIX8o();
		p[ 0]+=h[6];	p[ 1]+=h[7];	p[ 2]+=h[8];	p[ 3]+=h[0];	p[ 4]+=h[1];	p[ 5]+=c_buffer[30];	p[ 7]+=c_buffer[31];	p[ 6]+=c_buffer[32];
		TFBIGMIX8e();
		p[ 0]+=h[7];	p[ 1]+=h[8];	p[ 2]+=h[0];	p[ 3]+=h[1];	p[ 4]+=h[2];	p[ 5]+=c_buffer[32];	p[ 7]+=c_buffer[33];	p[ 6]+=c_buffer[34];
		TFBIGMIX8o();
		p[ 0]+=h[8];	p[ 1]+=h[0];	p[ 2]+=h[1];	p[ 3]+=h[2];	p[ 4]+=h[3];	p[ 5]+=c_buffer[34];	p[ 7]+=c_buffer[35];	p[ 6]+=c_buffer[36];
		TFBIGMIX8e();
		p[ 0]+=h[0];	p[ 1]+=h[1];	p[ 2]+=h[2];	p[ 3]+=h[3];	p[ 4]+=h[4];	p[ 5]+=c_buffer[36];	p[ 7]+=c_buffer[37];	p[ 6]+=c_buffer[38];
		TFBIGMIX8o();
		p[ 0]+=h[1];	p[ 1]+=h[2];	p[ 2]+=h[3];	p[ 3]+=h[4];	p[ 4]+=h[5];	p[ 5]+=c_buffer[38];	p[ 7]+=c_buffer[39];	p[ 6]+=c_buffer[40];
		TFBIGMIX8e();
		p[ 0]+=h[2];	p[ 1]+=h[3];	p[ 2]+=h[4];	p[ 3]+=h[5];	p[ 4]+=h[6];	p[ 5]+=c_buffer[40];	p[ 7]+=c_buffer[41];	p[ 6]+=c_buffer[42];
		TFBIGMIX8o();
		p[ 0]+=h[3];	p[ 1]+=h[4];	p[ 2]+=h[5];	p[ 3]+=h[6];	p[ 4]+=h[7];	p[ 5]+=c_buffer[42];	p[ 7]+=c_buffer[43];	p[ 6]+=c_buffer[44];
		TFBIGMIX8e();
		p[ 0]+=h[4];	p[ 1]+=h[5];	p[ 2]+=h[6];	p[ 3]+=h[7];	p[ 4]+=h[8];	p[ 5]+=c_buffer[44];	p[ 7]+=c_buffer[45];	p[ 6]+=c_buffer[46];
		TFBIGMIX8o();
		p[ 0]+=h[5];	p[ 1]+=h[6];	p[ 2]+=h[7];	p[ 3]+=h[8];	p[ 4]+=h[0];	p[ 5]+=c_buffer[46];	p[ 7]+=c_buffer[47];	p[ 6]+=c_buffer[48];
		TFBIGMIX8e();
		p[ 0]+=h[6];	p[ 1]+=h[7];	p[ 2]+=h[8];	p[ 3]+=h[0];	p[ 4]+=h[1];	p[ 5]+=c_buffer[48];	p[ 7]+=c_buffer[49];	p[ 6]+=c_buffer[50];
		TFBIGMIX8o();
		p[ 0]+=h[7];	p[ 1]+=h[8];	p[ 2]+=h[0];	p[ 3]+=h[1];	p[ 4]+=h[2];	p[ 5]+=c_buffer[50];	p[ 7]+=c_buffer[51];	p[ 6]+=c_buffer[52];
		TFBIGMIX8e();
		p[ 0]+=h[8];	p[ 1]+=h[0];	p[ 2]+=h[1];	p[ 3]+=h[2];	p[ 4]+=h[3];	p[ 5]+=c_buffer[52];	p[ 7]+=c_buffer[53];	p[ 6]+=c_buffer[54];
		TFBIGMIX8o();
		p[ 0]+=h[0];	p[ 1]+=h[1];	p[ 2]+=h[2];	p[ 3]+=h[3];	p[ 4]+=h[4];	p[ 5]+=c_buffer[54];	p[ 7]+=c_buffer[55];	p[ 6]+=c_buffer[56];

		p[0]^= c_buffer[57];
		p[1]^= nonce2;

		t0 = vectorize(8); // extra
		t1 = vectorize(0xFF00000000000000ull); // etype
		t2 = t0^t1;

		h[0] = p[ 0];
		h[1] = p[ 1];
		h[2] = p[ 2];
		h[3] = p[ 3];
		h[4] = p[ 4];
		h[5] = p[ 5];
		h[6] = p[ 6];
		h[7] = p[ 7];

		h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		p[ 0] = p[ 1] = p[ 2] = p[ 3] = p[ 4] =p[ 5] =p[ 6] = p[ 7] = vectorize(0);

		#define h0 h[0]
		#define h1 h[1]
		#define h2 h[2]
		#define h3 h[3]
		#define h4 h[4]
		#define h5 h[5]
		#define h6 h[6]
		#define h7 h[7]
		#define h8 h[8]

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		// cubehash512
		uint32_t x[32] = {
			0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
			0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
			0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
			0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
			0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
			0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
			0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
			0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
		};

//		*(uint2x4*)&x[ 0] ^= *((uint2x4*)&p[0]);
		#pragma unroll 4
		for(int i=0;i<4;i++){
			x[i*2] ^= p[i].x;
			x[i*2+1] ^= p[i].y;
		}
		rrounds(x);

//		*(uint2x4*)&x[ 0] ^= *((uint2x4*)&p[4]);
		#pragma unroll 4
		for(int i=0;i<4;i++){
			x[i*2] ^= p[i+4].x;
			x[i*2+1] ^= p[i+4].y;
		}
		rrounds(x);

		// Padding Block
		x[ 0] ^= 0x80;
		rrounds(x);

//		Final(x, (BitSequence*)Hash);
		x[31] ^= 1;

		/* "the state is then transformed invertibly through 10r identical rounds" */
		#pragma unroll 10
		for (int i = 0;i < 10;++i)
			rrounds(x);

		// fugue512
		uint32_t Hash[16];
		#pragma unroll 16
		for(int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(x[i]);

		uint32_t S[36];
		uint32_t B[ 9];

		S[ 0] = S[ 1] = S[ 2] = S[ 3] = S[ 4] = S[ 5] = S[ 6] = S[ 7] = S[ 8] = S[ 9] = S[10] = S[11] = S[12] = S[13] = S[14] = S[15] = S[16] = S[17] = S[18] = S[19] = 0;
		*(uint2x4*)&S[20] = *(uint2x4*)&c_S[ 0];
		*(uint2x4*)&S[28] = *(uint2x4*)&c_S[ 8];

		FUGUE512_3(Hash[0x0], Hash[0x1], Hash[0x2]);
		FUGUE512_3(Hash[0x3], Hash[0x4], Hash[0x5]);
		FUGUE512_3(Hash[0x6], Hash[0x7], Hash[0x8]);
		FUGUE512_3(Hash[0x9], Hash[0xA], Hash[0xB]);
		FUGUE512_3(Hash[0xC], Hash[0xD], Hash[0xE]);
		FUGUE512_3(Hash[0xF], 0U, 512U);

		//#pragma unroll 16
		for (uint32_t i = 0; i < 32; i+=2){
			mROR3;
			CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]);
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			mROR3;
			CMIX36(S[ 0], S[ 1], S[ 2], S[ 4], S[ 5], S[ 6], S[18], S[19], S[20]);
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		}
		//#pragma unroll 13
		for (uint32_t i = 0; i < 13; i ++) {
			S[ 4] ^= S[ 0];	S[ 9] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[27] ^= S[ 0];
			mROR9;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
			S[ 4] ^= S[ 0];	S[10] ^= S[ 0];	S[19] ^= S[ 0];	S[28] ^= S[ 0];
			mROR8;
			SMIX_LDG(shared, S[ 0], S[ 1], S[ 2], S[ 3]);
		}
		S[ 4] ^= S[ 0];	S[ 9] ^= S[ 0];	S[18] ^= S[ 0];	S[27] ^= S[ 0];

		S[ 0] = cuda_swab32(S[ 1]);	S[ 1] = cuda_swab32(S[ 2]);
		S[ 2] = cuda_swab32(S[ 3]);	S[ 3] = cuda_swab32(S[ 4]);
		S[ 4] = cuda_swab32(S[ 9]);	S[ 5] = cuda_swab32(S[10]);
		S[ 6] = cuda_swab32(S[11]);	S[ 7] = cuda_swab32(S[12]);
		S[ 8] = cuda_swab32(S[18]);	S[ 9] = cuda_swab32(S[19]);
		S[10] = cuda_swab32(S[20]);	S[11] = cuda_swab32(S[21]);
		S[12] = cuda_swab32(S[27]);	S[13] = cuda_swab32(S[28]);
		S[14] = cuda_swab32(S[29]);	S[15] = cuda_swab32(S[30]);

		uint64_t *outpHash = &output64[thread<<3];
		*(uint2x4*)&outpHash[ 0] = *(uint2x4*)&S[ 0];
		*(uint2x4*)&outpHash[ 4] = *(uint2x4*)&S[ 8];
	}
}

__host__
void skunk_cuda_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
	const dim3 grid((threads + TPB - 1) / TPB);
	const dim3 block(TPB);

	uint64_t *d_hash = (uint64_t*) g_hash;
	skunk_gpu_hash_80 <<< grid, block >>> (threads, startNounce, d_hash);

	MyStreamSynchronize(NULL, 1, thr_id);
}

__host__
void skunk_setBlock_80(int thr_id, void *pdata)
{
	uint64_t message[20];
	memcpy(&message[0], pdata, 80);

	uint64_t p[8];
	uint64_t h[9];
	uint64_t t0, t1, t2;

	h[0] = 0x4903ADFF749C51CEull;
	h[1] = 0x0D95DE399746DF03ull;
	h[2] = 0x8FD1934127C79BCEull;
	h[3] = 0x9A255629FF352CB1ull;
	h[4] = 0x5DB62599DF6CA7B0ull;
	h[5] = 0xEABE394CA9D5C3F4ull;
	h[6] = 0x991112C71A75B523ull;
	h[7] = 0xAE18A40B660FCC33ull;
	// h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h[8] = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	memcpy(&p[0], &message[0], 64);

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	message[10] = message[0] ^ p[0];
	message[11] = message[1] ^ p[1];
	message[12] = message[2] ^ p[2];
	message[13] = message[3] ^ p[3];
	message[14] = message[4] ^ p[4];
	message[15] = message[5] ^ p[5];
	message[16] = message[6] ^ p[6];
	message[17] = message[7] ^ p[7];
	message[18] = t2;

	uint64_t buffer[128];

//	buffer[ 0] = message[ 8];
	buffer[ 0] = message[ 9];
	h[0] = buffer[ 1] = message[10];
	h[1] = buffer[ 2] = message[11];
	h[2] = buffer[ 3] = message[12];
	h[3] = buffer[ 4] = message[13];
	h[4] = buffer[ 5] = message[14];
	h[5] = buffer[ 6] = message[15];
	h[6] = buffer[ 7] = message[16];
	h[7] = buffer[ 8] = message[17];
	h[8] = buffer[ 9] = h[0]^h[1]^h[2]^h[3]^h[4]^h[5]^h[6]^h[7]^0x1BD11BDAA9FC1A22ULL;

	t0 = 0x50ull;
	t1 = 0xB000000000000000ull;
	t2 = t0^t1;

	p[0] = message[ 8] + h[0];
	p[2] = h[2]; p[3] = h[3]; p[4] = h[4];
	p[5] = h[5] + t0;
	p[6] = h[6] + t1;
	p[7] = h[7];

	p[2] += p[3];
	p[4] += p[5]; p[6] += p[7];

	p[3] = ROTL64(p[3], 36) ^ p[2];
	p[5] = ROTL64(p[5], 19) ^ p[4];
	p[7] = ROTL64(p[7], 37) ^ p[6];
	p[4] += p[7]; p[6] += p[5];

	p[7] = ROTL64(p[7], 27) ^ p[4];
	p[5] = ROTL64(p[5], 14) ^ p[6];

	buffer[10] = p[ 0];
	buffer[11] = p[ 2];
	buffer[12] = p[ 3];
	buffer[13] = p[ 4];
	buffer[14] = p[ 5];
	buffer[15] = p[ 6];
	buffer[16] = p[ 7];
	buffer[17] = ROTL64(p[3], 42);
	buffer[18] = ROTL64(p[5], 36);
	buffer[19] = ROTL64(p[7], 39);

	buffer[20] = h[6]+t1;
	buffer[21] = h[8]+1;
	buffer[22] = h[7]+t2;
	buffer[23] = h[0]+2;
	buffer[24] = h[8]+t0;
	buffer[25] = h[1]+3;
	buffer[26] = h[0]+t1;
	buffer[27] = h[2]+4;
	buffer[28] = h[1]+t2;
	buffer[29] = h[3]+5;
	buffer[30] = h[2]+t0;
	buffer[31] = h[4]+6;
	buffer[32] = h[3]+t1;
	buffer[33] = h[5]+7;
	buffer[34] = h[4]+t2;
	buffer[35] = h[6]+8;
	buffer[36] = h[5]+t0;
	buffer[37] = h[7]+9;
	buffer[38] = h[6]+t1;
	buffer[39] = h[8]+10;
	buffer[40] = h[7]+t2;
	buffer[41] = h[0]+11;
	buffer[42] = h[8]+t0;
	buffer[43] = h[1]+12;
	buffer[44] = h[0]+t1;
	buffer[45] = h[2]+13;
	buffer[46] = h[1]+t2;
	buffer[47] = h[3]+14;
	buffer[48] = h[2]+t0;
	buffer[49] = h[4]+15;
	buffer[50] = h[3]+t1;
	buffer[51] = h[5]+16;
	buffer[52] = h[4]+t2;
	buffer[53] = h[6]+17;
	buffer[54] = h[5]+t0;
	buffer[55] = h[7]+18;
	buffer[56] = h[6]+t1;

	buffer[57] = message[ 8];

	cudaMemcpyToSymbol(c_buffer, buffer, sizeof(c_buffer), 0, cudaMemcpyHostToDevice);
	CUDA_LOG_ERROR();
}

__host__
void skunk_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
}

