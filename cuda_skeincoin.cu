/* Merged skein512 80 + sha256 64 (in a single kernel) for SM 5+
 * based on sp and klaus work, adapted by tpruvot to keep skein2 compat
 */

#include <stdint.h>
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

/* try 1024 for 970+ */
#define TPB 512

static __constant__ uint64_t c_message16[2];
static __constant__ uint2 precalcvalues[9];

static uint32_t *d_found[MAX_GPUS];

static __device__ __forceinline__ uint2 vectorizelow(uint32_t v) {
	uint2 result;
	result.x = v;
	result.y = 0;
	return result;
}

static __device__ __forceinline__ uint2 vectorizehigh(uint32_t v) {
	uint2 result;
	result.x = 0;
	result.y = v;
	return result;
}

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_KINIT_UI2(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
	k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
		^ vectorize(SPH_C64(0x1BD11BDAA9FC1A22)); \
	t2 = t0 ^ t1; \
}

#define TFBIG_ADDKEY_UI2(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + vectorize(s)); \
}

#define TFBIG_MIX_UI2(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROL2(x1, rc) ^ x0; \
}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX_UI2(w0, w1, rc0); \
	TFBIG_MIX_UI2(w2, w3, rc1); \
	TFBIG_MIX_UI2(w4, w5, rc2); \
	TFBIG_MIX_UI2(w6, w7, rc3); \
}

#define TFBIG_4e_UI2(s)  { \
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4o_UI2(s)  { \
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

/* precalc */

#define TFBIG_ADDKEY_PRE(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
	w0 = (w0 + SKBI(k, s, 0)); \
	w1 = (w1 + SKBI(k, s, 1)); \
	w2 = (w2 + SKBI(k, s, 2)); \
	w3 = (w3 + SKBI(k, s, 3)); \
	w4 = (w4 + SKBI(k, s, 4)); \
	w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
	w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
	w7 = (w7 + SKBI(k, s, 7) + (s)); \
}

#define TFBIG_MIX_PRE(x0, x1, rc) { \
	x0 = x0 + x1; \
	x1 = ROTL64(x1, rc) ^ x0; \
}

#define TFBIG_MIX8_PRE(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
	TFBIG_MIX_PRE(w0, w1, rc0); \
	TFBIG_MIX_PRE(w2, w3, rc1); \
	TFBIG_MIX_PRE(w4, w5, rc2); \
	TFBIG_MIX_PRE(w6, w7, rc3); \
}

#define TFBIG_4e_PRE(s)  { \
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
	TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
	TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
	TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
}

#define TFBIG_4o_PRE(s)  { \
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
	TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
	TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
	TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
	TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
}

/* Elementary defines for SHA256 */

#define SWAB32(x)     cuda_swab32(x)

#define R(x, n)       ((x) >> (n))
#define Ch(x, y, z)   ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)  ((x & (y | z)) | (y & z))
#define S0(x)         (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)         (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)         (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)         (ROTR32(x,17) ^ ROTR32(x, 19) ^ R(x, 10))

static __device__ __constant__ uint32_t sha256_hashTable[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// precomputed table
static __constant__ uint32_t sha256_endingTable[64] = {
	0xc28a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf374,
	0x649b69c1, 0xf0fe4786, 0x0fe1edc6, 0x240cf254, 0x4fe9346f, 0x6cc984be, 0x61b9411e, 0x16f988fa,
	0xf2c65152, 0xa88e5a6d, 0xb019fc65, 0xb9d99ec7, 0x9a1231c3, 0xe70eeaa0, 0xfdb1232b, 0xc7353eb0,
	0x3069bad5, 0xcb976d5f, 0x5a0f118f, 0xdc1eeefd, 0x0a35b689, 0xde0b7a04, 0x58f4ca9d, 0xe15d5b16,
	0x007f3e86, 0x37088980, 0xa507ea32, 0x6fab9537, 0x17406110, 0x0d8cd6f1, 0xcdaa3b6d, 0xc0bbbe37,
	0x83613bda, 0xdb48a363, 0x0b02e931, 0x6fd15ca7, 0x521afaca, 0x31338431, 0x6ed41a95, 0x6d437890,
	0xc39c91f2, 0x9eccabbd, 0xb5c9a0e6, 0x532fb63c, 0xd2c741c6, 0x07237ea3, 0xa4954b68, 0x4c191d76
};

static __constant__ uint32_t sha256_constantTable[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__global__ __launch_bounds__(TPB)
void skeincoin_gpu_hash_50(uint32_t threads, uint32_t startNounce, uint32_t* d_found, uint64_t target64, int swap)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;
		uint2 p[8];

		h0 = precalcvalues[0];
		h1 = precalcvalues[1];
		h2 = precalcvalues[2];
		h3 = precalcvalues[3];
		h4 = precalcvalues[4];
		h5 = precalcvalues[5];
		h6 = precalcvalues[6];
		h7 = precalcvalues[7];
		t2 = precalcvalues[8];

		const uint32_t nonce = startNounce + thread;
		const uint2 nonce2 = make_uint2(_LODWORD(c_message16[1]), swap ? cuda_swab32(nonce) : nonce);

		// skein_big_close -> etype = 0x160, ptr = 16, bcount = 1, extra = 16
		p[0] = vectorize(c_message16[0]);
		p[1] = nonce2;

		#pragma unroll
		for (int i = 2; i < 8; i++)
			p[i] = make_uint2(0, 0);

		t0 = vectorizelow(0x50ull); // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = vectorizehigh(0xB0000000ul); // (bcount >> 58) + ((sph_u64)(etype) << 55);
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
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


		t0 = vectorizelow(8); // extra
		t1 = vectorizehigh(0xFF000000ul); // etype

		h0 = vectorize(c_message16[0]) ^ p[0];
		h1 = nonce2 ^ p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		h8 = h0 ^ h1 ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		t2 = vectorize(0xFF00000000000008ull);

		// p[8] = { 0 };
		#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = make_uint2(0, 0);

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

		uint32_t *message = (uint32_t *)p;

		uint32_t regs[8];
		uint32_t hash[8];

		// Init with Hash-Table
		#pragma unroll 8
		for (int k = 0; k < 8; k++) {
			hash[k] = regs[k] = sha256_hashTable[k];
		}

		uint32_t W1[16];
		uint32_t W2[16];

		#pragma unroll 16
		for (int k = 0; k<16; k++)
			W1[k] = SWAB32(message[k]);

		// Progress W1
		#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int k = 6; k >= 0; k--) regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		// Progress W2...W3

		////// PART 1
		#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];
		#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

		#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Round function
		#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 16] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		////// PART 2
		#pragma unroll 2
		for (int j = 0; j<2; j++)
			W1[j] = s1(W2[14 + j]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

		#pragma unroll 5
		for (int j = 2; j<7; j++)
			W1[j] = s1(W1[j - 2]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

		#pragma unroll 8
		for (int j = 7; j<15; j++)
			W1[j] = s1(W1[j - 2]) + W1[j - 7] + s0(W2[1 + j]) + W2[j];

		W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

		// Round function
		#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 32] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		////// PART 3
		#pragma unroll 2
		for (int j = 0; j<2; j++)
			W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

		#pragma unroll 5
		for (int j = 2; j<7; j++)
			W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

		#pragma unroll 8
		for (int j = 7; j<15; j++)
			W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Round function
		#pragma unroll 16
		for (int j = 0; j<16; j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 48] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

		#pragma unroll 8
		for (int k = 0; k<8; k++)
			hash[k] += regs[k];

		/////
		///// Second Pass (ending)
		/////
		#pragma unroll 8
		for (int k = 0; k<8; k++)
			regs[k] = hash[k];

		// Progress W1
		uint32_t T1, T2;
		#pragma unroll 1
		for (int j = 0; j<56; j++)//62
		{
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6])+sha256_endingTable[56];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		regs[7] = T1 + T2;
		regs[3] += T1;

		T1 = regs[6] + S1(regs[3]) + Ch(regs[3], regs[4], regs[5]) + sha256_endingTable[57];
		T2 = S0(regs[7]) + Maj(regs[7], regs[0], regs[1]);
		regs[6] = T1 + T2;
		regs[2] += T1;
		//************
		regs[1] += regs[5] + S1(regs[2]) + Ch(regs[2], regs[3], regs[4]) + sha256_endingTable[58];
		regs[0] += regs[4] + S1(regs[1]) + Ch(regs[1], regs[2], regs[3]) + sha256_endingTable[59];
		regs[7] += regs[3] + S1(regs[0]) + Ch(regs[0], regs[1], regs[2]) + sha256_endingTable[60];
		regs[6] += regs[2] + S1(regs[7]) + Ch(regs[7], regs[0], regs[1]) + sha256_endingTable[61];

		uint64_t test = SWAB32(hash[7] + regs[7]);
		test <<= 32;
		test|= SWAB32(hash[6] + regs[6]);
		if (test <= target64)
		{
			uint32_t tmp = atomicExch(&(d_found[0]), startNounce + thread);
			if (tmp != UINT32_MAX)
				d_found[1] = tmp;
		}
	}
}

__host__
static void precalc(uint64_t* message)
{
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	//h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h8 = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	uint64_t p[8];
	for (int i = 0; i<8; i++)
		p[i] = message[i];

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

	uint64_t buffer[9];
	buffer[0] = message[0] ^ p[0];
	buffer[1] = message[1] ^ p[1];
	buffer[2] = message[2] ^ p[2];
	buffer[3] = message[3] ^ p[3];
	buffer[4] = message[4] ^ p[4];
	buffer[5] = message[5] ^ p[5];
	buffer[6] = message[6] ^ p[6];
	buffer[7] = message[7] ^ p[7];
	buffer[8] = t2;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(precalcvalues, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice));
}

__host__
void skeincoin_init(int thr_id)
{
	cuda_get_arch(thr_id);
	CUDA_SAFE_CALL(cudaMalloc(&d_found[thr_id], 2 * sizeof(uint32_t)));
}

__host__
void skeincoin_free(int thr_id) {
	cudaFree(d_found[thr_id]);
}

__host__
void skeincoin_setBlock_80(int thr_id, void *pdata)
{
	uint64_t message[16];
	memcpy(&message[0], pdata, 80);

	cudaMemcpyToSymbol(c_message16, &message[8], 16, 0, cudaMemcpyHostToDevice);

	precalc(message);
}

__host__
uint32_t skeincoin_hash_sm5(int thr_id, uint32_t threads, uint32_t startNounce, int swap, uint64_t target64, uint32_t *secNonce)
{
	uint32_t h_found[2];
	uint32_t threadsperblock = TPB;
	dim3 block(threadsperblock);
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);

	memset(h_found, 0xff, sizeof(h_found));
	cudaMemset(d_found[thr_id], 0xff, 2 * sizeof(uint32_t));

	skeincoin_gpu_hash_50 <<< grid, block >>> (threads, startNounce, d_found[thr_id], target64, swap);

	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (h_found[1] && h_found[1] != UINT32_MAX && h_found[1] != h_found[0])
		*secNonce = h_found[1];
	return h_found[0];
}
