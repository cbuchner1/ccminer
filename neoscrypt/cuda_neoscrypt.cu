#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "cuda_vectors.h" /* NOT COMPATIBLE WITH SM 3.0 !!! */

static uint32_t *d_buffer[MAX_GPUS];
static uint32_t *d_NNonce[MAX_GPUS];
__constant__ uint4* W;
__constant__ uint32_t pTarget[8];
__constant__ uint32_t key_init[16];
__constant__ uint32_t input_init[16];
__constant__ uint32_t c_data[80];

/// constants ///

static const __constant__ uint8 BLAKE2S_IV_Vec = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint8 BLAKE2S_IV_Vechost = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

static __constant__ uint32_t BLAKE2S_SIGMA[10][16];

#define FASTKDF_BUFFER_SIZE  256U

// Blake2S

#define BLAKE2S_BLOCK_SIZE    64U
#define BLAKE2S_OUT_SIZE      32U
#define BLAKE2S_KEY_SIZE      32U

#if __CUDA_ARCH__ >= 500
#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	idx = BLAKE2S_SIGMA[idx0][idx1+1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}
#else
#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = rotateL(d^a, 16); \
	c += d; b = rotateR(b^c, 12); \
	idx = BLAKE2S_SIGMA[idx0][idx1+1]; a += key[idx]; \
	a += b; d = rotateR(d^a, 8); \
	c += d; b = rotateR(b^c, 7); \
}
#endif

#if __CUDA_ARCH__ >= 500
#define BLAKE_G_PRE(idx0, idx1, a, b, c, d, key) { \
	a += key[idx0]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key[idx1]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}
#else
#define BLAKE_G_PRE(idx0, idx1, a, b, c, d, key) { \
	a += key[idx0]; \
	a += b; d = rotateL(d^a, 16); \
	c += d; b = rotateR(b^c, 12); \
	a += key[idx1]; \
	a += b; d = rotateR(d^a, 8); \
	c += d; b = rotateR(b^c, 7); \
}
#endif

#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA_host[idx0][idx1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,16); \
	c += d; b = ROTR32(b^c, 12); \
	idx = BLAKE2S_SIGMA_host[idx0][idx1+1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,8); \
	c += d; b = ROTR32(b^c, 7); \
}

static __forceinline__ __device__
void Blake2S(uint32_t * inout, const uint32_t * TheKey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

#if 0
	for (int x = 0; x < 10; ++x)
	{
		BLAKE_G(x, 0x0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
		BLAKE_G(x, 0x2, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
		BLAKE_G(x, 0x4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
		BLAKE_G(x, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
		BLAKE_G(x, 0x8, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
		BLAKE_G(x, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
		BLAKE_G(x, 0xC, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
		BLAKE_G(x, 0xE, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	}
#else
	// { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0x0, 0x1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x2, 0x3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x4, 0x5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0x6, 0x7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x8, 0x9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xA, 0xB, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0xC, 0xD, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xE, 0xF, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(0xE, 0xA, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x4, 0x8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x9, 0xF, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0xD, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x1, 0xC, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x0, 0x2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0xB, 0x7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x5, 0x3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(0xB, 0x8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0xC, 0x0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x5, 0x2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0xF, 0xD, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xA, 0xE, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x3, 0x6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x7, 0x1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x9, 0x4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(0x7, 0x9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x3, 0x1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xD, 0xC, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0xB, 0xE, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x2, 0x6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x5, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x4, 0x0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xF, 0x8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE(0x9, 0x0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x5, 0x7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x2, 0x4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0xA, 0xF, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xE, 0x1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xB, 0xC, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x6, 0x8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x3, 0xD, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE(0x2, 0xC, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x6, 0xA, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x0, 0xB, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0x8, 0x3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x4, 0xD, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x7, 0x5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0xF, 0xE, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x1, 0x9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE(0xC, 0x5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x1, 0xF, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xE, 0xD, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0x4, 0xA, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x0, 0x7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x6, 0x3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x9, 0x2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x8, 0xB, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE(0xD, 0xB, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x7, 0xE, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xC, 0x1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0x3, 0x9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x5, 0x0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xF, 0x4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x8, 0x6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x2, 0xA, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE(0x6, 0xF, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0xE, 0x9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xB, 0x3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0x0, 0x8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xC, 0x2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xD, 0x7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x1, 0x4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xA, 0x5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	// { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE(0xA, 0x2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x8, 0x4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(0x7, 0x6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(0x1, 0x5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0xF, 0xB, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0x9, 0xE, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(0x3, 0xC, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(0xD, 0x0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
#endif

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

#if 0
	for (int x = 0; x < 10; ++x)
	{
		BLAKE_G(x, 0x0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_G(x, 0x2, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_G(x, 0x4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_G(x, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_G(x, 0x8, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_G(x, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_G(x, 0xC, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_G(x, 0xE, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}
#else
	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0x0, 0x1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(0x2, 0x3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(0x4, 0x5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(0x6, 0x7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(0x8, 0x9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0xA, 0xB, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(0xC, 0xD, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(0xE, 0xF, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	// { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(0xE, 0xA, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(0x4, 0x8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(0x9, 0xF, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(0xD, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(0x1, 0xC, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0x0, 0x2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(0xB, 0x7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(0x5, 0x3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	// { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(0xB, 0x8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(0xC, 0x0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(0x5, 0x2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(0xF, 0xD, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(0xA, 0xE, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0x3, 0x6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(0x7, 0x1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(0x9, 0x4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	// { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(0x7, 0x9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(0x3, 0x1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(0xD, 0xC, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(0xB, 0xE, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(0x2, 0x6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0x5, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(0x4, 0x0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(0xF, 0x8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	for (int x = 4; x < 10; ++x)
	{
		BLAKE_G(x, 0x0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_G(x, 0x2, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_G(x, 0x4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_G(x, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_G(x, 0x8, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_G(x, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_G(x, 0xC, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_G(x, 0xE, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}
#endif

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0]=V.lo;
}

static __forceinline__ __host__
void Blake2Shost(uint32_t * inout, const uint32_t * inkey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x2, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x8, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0xC, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0xE, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x2, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x8, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0xA, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0xC, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0xE, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}

static __forceinline__ __device__
void fastkdf256(const uint32_t* password, uint8_t* output)
{
	uint8_t bufidx = 0;
	uchar4 bufhelper;
	uint8_t A[320],B[288];

	((uintx64*)A)[0] = ((uintx64*)password)[0];
	((uint816 *)A)[4] =  ((uint816 *)password)[0];

	((uintx64*)B)[0] = ((uintx64*)password)[0];
	((uint48 *)B)[8] = ((uint48 *)password)[0];

	uint32_t input[BLAKE2S_BLOCK_SIZE/4]; uint32_t key[BLAKE2S_BLOCK_SIZE / 4] = { 0 };

	((uint816*)input)[0] = ((uint816*)input_init)[0];
	((uint48*)key)[0] = ((uint48*)key_init)[0];

	for (int i = 0; i < 32; ++i)
	{
		bufhelper = ((uchar4*)input)[0];
		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
			bufhelper += ((uchar4*)input)[x];
		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;

		int qbuf = bufidx/4;
		int rbuf = bufidx&3;
		int bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R(shifted, ((uint8*)input)[0], bitbuf);

		for (int k = 0; k < 9; ++k) {
			((uint32_t *)B)[k + qbuf] ^= ((uint32_t *)shifted)[k];
		}

		if (bufidx < BLAKE2S_KEY_SIZE)                          {((uint8*)B)[8] = ((uint8*)B)[0];}
		else if (bufidx > FASTKDF_BUFFER_SIZE-BLAKE2S_OUT_SIZE) {((uint8*)B)[0] = ((uint8*)B)[8];}

		if (i<31) {
			for (int k = 0; k <BLAKE2S_BLOCK_SIZE / 4; k++) {
				((uchar4*)(input))[k] = make_uchar4(
					(A + bufidx)[4 * k], (A + bufidx)[4 * k + 1],
					(A + bufidx)[4 * k + 2], (A + bufidx)[4 * k + 3]
				);
			}

			for (int k = 0; k <BLAKE2S_KEY_SIZE / 4; k++) {
				((uchar4*)(key))[k] = make_uchar4(
					(B + bufidx)[4 * k], (B + bufidx)[4 * k + 1],
					(B + bufidx)[4 * k + 2], (B + bufidx)[4 * k + 3]
				);
			}

			Blake2S((uint32_t*)input, key);
		}
	}

	int left = FASTKDF_BUFFER_SIZE - bufidx;
	int qleft =left/4;
	int rleft =left&3;
	for (int k = 0; k < qleft; ++k) {
		((uchar4*)output)[k] = make_uchar4(
			(B + bufidx)[4 * k], (B + bufidx)[4 * k + 1],
			(B + bufidx)[4 * k + 2], (B + bufidx)[4 * k + 3]
		) ^ ((uchar4*)A)[k];
	}
	for (int i = 4*qleft; i < 4*qleft+rleft; ++i) {
		output[i] = (B + bufidx)[i] ^ A[i];
	}
	for (int i = qleft*4+rleft; i < (qleft+1)*4; ++i) {
		((uint8_t *)output)[i] = ((uint8_t *)B)[i - left] ^ ((uint8_t *)A)[i];
	}
	for (int i = qleft+1; i < FASTKDF_BUFFER_SIZE/4; ++i) {
		((uchar4 *)output)[i] = make_uchar4(B[4*i - left],B[4*i+1-left],
		 B[4*i+2-left],B[4*i+3-left]) ^ ((uchar4 *)A)[i];
	}
}

static __forceinline__ __device__
void fastkdf32(const uint32_t * password, const uint32_t * salt, uint32_t * output)
{
	uint8_t bufidx = 0;
	uchar4 bufhelper;

	uint8_t A[320];
	uint8_t B[288];

	// Initialize the password buffer
	((uintx64*)A)[0] = ((uintx64*)password)[0];
	((uint816*)A)[4] = ((uint816*)password)[0];
	((uintx64*)B)[0] = ((uintx64*)salt)[0];
	((uintx64*)B)[1] = ((uintx64*)salt)[0];

	uint32_t input[BLAKE2S_BLOCK_SIZE/4];
	uint32_t key[BLAKE2S_BLOCK_SIZE/4] = { 0 };

	((uint816*)input)[0] = ((uint816*)password)[0];
	((uint48*)key)[0] = ((uint48*)salt)[0];

	for (int i = 0; i < 32; ++i)
	{
		Blake2S((uint32_t*)input, key);

		bufidx = 0;
		bufhelper = ((uchar4*)input)[0];

		for (int x = 1; x < BLAKE2S_OUT_SIZE / 4; ++x)
			bufhelper += ((uchar4*)input)[x];

		bufidx = bufhelper.x + bufhelper.y + bufhelper.z + bufhelper.w;
		int qbuf = bufidx / 4;
		int rbuf = bufidx & 3;
		int bitbuf = rbuf << 3;
		uint32_t shifted[9];

		shift256R(shifted, ((uint8*)input)[0], bitbuf);

		for (int k = 0; k < 9; ++k) {
			((uint32_t *)B)[k + qbuf] ^= ((uint32_t *)shifted)[k];
		}

		if (i<31) {
			if (bufidx < BLAKE2S_KEY_SIZE)                            {((uint8*)B)[8] = ((uint8*)B)[0];}
			else if (bufidx > FASTKDF_BUFFER_SIZE - BLAKE2S_OUT_SIZE) {((uint8*)B)[0] = ((uint8*)B)[8];}

			for (uint8_t k = 0; k < BLAKE2S_BLOCK_SIZE/4; k++) {
				((uchar4*)(input))[k] = make_uchar4(
					(A + bufidx)[4 * k], (A + bufidx)[4 * k + 1],
					(A + bufidx)[4 * k + 2], (A + bufidx)[4 * k + 3]
				);
			}
			for (uint8_t k = 0; k < BLAKE2S_KEY_SIZE / 4; k++) {
				((uchar4*)(key))[k] = make_uchar4(
					(B + bufidx)[4 * k], (B + bufidx)[4 * k + 1],
					(B + bufidx)[4 * k + 2], (B + bufidx)[4 * k + 3]
				);
			}
		}
	}

	uchar4 unfucked[1];
	unfucked[0] = make_uchar4(B[28 + bufidx], B[29 + bufidx],B[30 + bufidx], B[31 + bufidx]);
	((uint32_t*)output)[7] = ((uint32_t*)unfucked)[0] ^ ((uint32_t*)A)[7];
}


#define SALSA(a,b,c,d) { \
    t =a+d; b^=rotateL(t,  7); \
    t =b+a; c^=rotateL(t,  9); \
    t =c+b; d^=rotateL(t, 13); \
    t =d+c; a^=rotateL(t, 18); \
}

#define SALSA_CORE(state) { \
    SALSA(state.s0,state.s4,state.s8,state.sc); \
    SALSA(state.s5,state.s9,state.sd,state.s1); \
    SALSA(state.sa,state.se,state.s2,state.s6); \
    SALSA(state.sf,state.s3,state.s7,state.sb); \
    SALSA(state.s0,state.s1,state.s2,state.s3); \
    SALSA(state.s5,state.s6,state.s7,state.s4); \
    SALSA(state.sa,state.sb,state.s8,state.s9); \
    SALSA(state.sf,state.sc,state.sd,state.se); \
}

#if __CUDA_ARCH__ >=500
#define CHACHA_STEP(a,b,c,d) { \
    a += b; d = __byte_perm(d^a,0,0x1032); \
    c += d; b = rotateL(b^c, 12); \
    a += b; d = __byte_perm(d^a,0,0x2103); \
    c += d; b = rotateL(b^c, 7); \
}
#else
#define CHACHA_STEP(a,b,c,d) { \
    a += b; d = rotateL(d^a,16); \
    c += d; b = rotateL(b^c, 12); \
    a += b; d = rotateL(d^a,8); \
    c += d; b = rotateL(b^c, 7); \
}
#endif

#define CHACHA_CORE_PARALLEL(state) { \
    CHACHA_STEP(state.lo.s0, state.lo.s4, state.hi.s0, state.hi.s4); \
    CHACHA_STEP(state.lo.s1, state.lo.s5, state.hi.s1, state.hi.s5); \
    CHACHA_STEP(state.lo.s2, state.lo.s6, state.hi.s2, state.hi.s6); \
    CHACHA_STEP(state.lo.s3, state.lo.s7, state.hi.s3, state.hi.s7); \
    CHACHA_STEP(state.lo.s0, state.lo.s5, state.hi.s2, state.hi.s7); \
    CHACHA_STEP(state.lo.s1, state.lo.s6, state.hi.s3, state.hi.s4); \
    CHACHA_STEP(state.lo.s2, state.lo.s7, state.hi.s0, state.hi.s5); \
    CHACHA_STEP(state.lo.s3, state.lo.s4, state.hi.s1, state.hi.s6); \
}


static __forceinline__ __device__ uint16 salsa_small_scalar_rnd(const uint16 &X)
{
	uint16 state = X;
	uint32_t t;

	for (int i = 0; i < 10; ++i) { SALSA_CORE(state);}

	return(X + state);
}

static __device__ __forceinline__ uint16 chacha_small_parallel_rnd(const uint16 &X)
{
	uint16 st = X;

	for (int i = 0; i < 10; ++i) {CHACHA_CORE_PARALLEL(st);}
	return(X + st);
}

static __device__ __forceinline__ void neoscrypt_chacha(uint16 *XV)
{
	XV[0] ^= XV[3];
	uint16 temp;

	XV[0] = chacha_small_parallel_rnd(XV[0]); XV[1] ^= XV[0];
	 temp = chacha_small_parallel_rnd(XV[1]); XV[2] ^= temp;
	XV[1] = chacha_small_parallel_rnd(XV[2]); XV[3] ^= XV[1];
	XV[3] = chacha_small_parallel_rnd(XV[3]);
	XV[2] = temp;
}

static __device__ __forceinline__ void neoscrypt_salsa(uint16 *XV)
{
	XV[0] ^= XV[3];
	uint16 temp;

	XV[0] = salsa_small_scalar_rnd(XV[0]); XV[1] ^= XV[0];
	 temp = salsa_small_scalar_rnd(XV[1]); XV[2] ^= temp;
	XV[1] = salsa_small_scalar_rnd(XV[2]); XV[3] ^= XV[1];
	XV[3] = salsa_small_scalar_rnd(XV[3]);
	XV[2] = temp;
}


#define SHIFT 130

__global__ __launch_bounds__(128, 1)
void neoscrypt_gpu_hash_k0(uint32_t threads, uint32_t startNonce, int stratum)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		uint32_t data[80];
		uint16 X[4];
		uint32_t shift = thread * SHIFT * 16;
		const uint32_t nonce = startNonce + thread;

		for (int i = 0; i<20; i++) {
			((uint4*)data)[i] = ((uint4 *)c_data)[i];
		}  //ld.local.v4
		data[19] = (stratum) ? cuda_swab32(nonce) : nonce; //freaking morons !!!
		data[39] = data[19];
		data[59] = data[19];

		fastkdf256(data, (uint8_t*)X);

		((uintx64 *)(W + shift))[0] = ((uintx64 *)X)[0];
//		((ulonglong16 *)(W + shift))[0] = ((ulonglong16 *)X)[0];
	}
}

__global__ __launch_bounds__(128, 1)
void neoscrypt_gpu_hash_k01(uint32_t threads, uint32_t startNonce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		uint16 X[4];
		uint32_t shift = thread * SHIFT * 16;
		((uintx64 *)X)[0]= ldg256(&(W + shift)[0]);

		//#pragma unroll
		for (int i = 0; i < 128; ++i)
		{
			neoscrypt_chacha(X);
			((ulonglong16 *)(W + shift))[i+1] = ((ulonglong16 *)X)[0];
//			((uintx64 *)(W + shift))[i + 1] = ((uintx64 *)X)[0];
		}
	}
}

__global__ __launch_bounds__(128, 1)
void neoscrypt_gpu_hash_k2(uint32_t threads, uint32_t startNonce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		uint16 X[4];
		uint32_t shift = thread * SHIFT * 16;
		((uintx64 *)X)[0] = ldg256(&(W + shift)[2048]);

		for (int t = 0; t < 128; t++)
		{
			int idx = X[3].lo.s0 & 0x7F;
			((uintx64 *)X)[0] ^= ldg256(&(W + shift)[idx << 4]);
			neoscrypt_chacha(X);

		}
		((uintx64 *)(W + shift))[129] = ((uintx64*)X)[0];  // best checked

	}
}

__global__ __launch_bounds__(128, 1)
void neoscrypt_gpu_hash_k3(uint32_t threads, uint32_t startNonce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		uint32_t shift = thread * SHIFT * 16;
		uint16 Z[4];

		((uintx64*)Z)[0] = ldg256(&(W + shift)[0]);

		//#pragma unroll
		for (int i = 0; i < 128; ++i) {
			neoscrypt_salsa(Z);
			((ulonglong16 *)(W + shift))[i+1] = ((ulonglong16 *)Z)[0];
//			((uintx64 *)(W + shift))[i + 1] = ((uintx64 *)Z)[0];
		}
	}
}

__global__ __launch_bounds__(128, 1)
void neoscrypt_gpu_hash_k4(uint32_t threads, uint32_t startNonce, uint32_t *nonceRes, int stratum)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		uint32_t shift = thread * SHIFT * 16;
		uint16 Z[4];
		uint32_t outbuf[8];
		uint32_t data[80];

		for (int i=0; i<20; i++) {
			((uint4*)data)[i] = ((uint4 *)c_data)[i];
		}

		data[19] = (stratum) ? cuda_swab32(nonce) : nonce;
		data[39] = data[19];
		data[59] = data[19];
		((uintx64 *)Z)[0] = ldg256(&(W + shift)[2048]);
		for (int t = 0; t < 128; t++)
		{
			int idx = Z[3].lo.s0 & 0x7F;
			((uintx64 *)Z)[0] ^= ldg256(&(W + shift)[idx << 4]);
			neoscrypt_salsa(Z);
		}
		((uintx64 *)Z)[0] ^= ldg256(&(W + shift)[2064]);
		fastkdf32(data, (uint32_t*)Z, outbuf);
#if __CUDA_ARCH__ < 320
		// workaround required when using SM 3.0 shift256R() func (tested on SM 5.0)
		if (thread == 0)
			printf("", outbuf[7]);
#endif
		if (outbuf[7] <= pTarget[7]) {
			atomicMin(nonceRes, nonce); // init val is UINT32_MAX
		}
	}
}

__host__
void neoscrypt_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
	cudaMalloc(&d_NNonce[thr_id], sizeof(uint32_t));
	CUDA_SAFE_CALL(cudaMalloc(&d_buffer[thr_id], (size_t) 256 * SHIFT * threads));
	cudaMemcpyToSymbol(W, &d_buffer[thr_id], sizeof(uint4*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(BLAKE2S_SIGMA, BLAKE2S_SIGMA_host, sizeof(BLAKE2S_SIGMA_host), 0, cudaMemcpyHostToDevice);
}

__host__
void neoscrypt_cpu_free(int thr_id)
{
	cudaFree(d_NNonce[thr_id]);
	cudaFree(d_buffer[thr_id]);
}

__host__
uint32_t neoscrypt_cpu_hash_k4(int thr_id, uint32_t threads, uint32_t startNounce, int have_stratum, int order)
{
	uint32_t result[MAX_GPUS];
	memset(result, 0xff, sizeof(result));
	cudaMemset(d_NNonce[thr_id], 0xff, sizeof(uint32_t));

	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	neoscrypt_gpu_hash_k0  <<< grid, block >>>(threads, startNounce, have_stratum);
	neoscrypt_gpu_hash_k01 <<< grid, block >>>(threads, startNounce);
	neoscrypt_gpu_hash_k2  <<< grid, block >>>(threads, startNounce);
	neoscrypt_gpu_hash_k3  <<< grid, block >>>(threads, startNounce);
	neoscrypt_gpu_hash_k4  <<< grid, block >>>(threads, startNounce, d_NNonce[thr_id], have_stratum);

	MyStreamSynchronize(NULL, order, thr_id);
	cudaMemcpy(&result[thr_id], d_NNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	return result[thr_id];
}

__host__
void neoscrypt_setBlockTarget(uint32_t* pdata, const void *target)
{
	unsigned char PaddedMessage[80*4]; //bring balance to the force
	uint32_t input[16], key[16] = { 0 };

	memcpy(PaddedMessage,     pdata, 80);
	memcpy(PaddedMessage + 80, pdata, 80);
	memcpy(PaddedMessage + 160, pdata, 80);
	memcpy(PaddedMessage + 240, pdata, 80);

	((uint16*)input)[0] = ((uint16*)pdata)[0];
	((uint8*)key)[0] = ((uint8*)pdata)[0];

	Blake2Shost(input,key);

	cudaMemcpyToSymbol(pTarget, target, 32, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(input_init, input, sizeof(input), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(key_init, key, sizeof(key), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_data, PaddedMessage, 80*4, 0, cudaMemcpyHostToDevice);
}

