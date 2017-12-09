#define SP_KERNEL

#ifdef SP_KERNEL
#include "cuda_skein512_sp.cuh"
#undef TFBIG_KINIT
#undef TFBIG_ADDKEY
#undef TFBIG_MIX
#else

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_helper.h"

#endif

static __constant__ uint64_t c_PaddedMessage80[20]; // padded message (80 bytes + 72 bytes midstate + align)

// Take a look at: https://www.schneier.com/skein1.3.pdf

#define SHL(x, n)			((x) << (n))
#define SHR(x, n)			((x) >> (n))

#if __CUDA_ARCH__ > 300
__device__
uint64_t skein_rotl64(const uint64_t x, const int offset)
{
	uint64_t res;
	asm("{\n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		".reg .pred p;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"setp.lt.u32 p, %2, 32;\n\t"
		"@!p mov.b64 %0, {vl,vh};\n\t"
		"@p  mov.b64 %0, {vh,vl};\n\t"
	"}"
		: "=l"(res) : "l"(x) , "r"(offset)
	);
	return res;
}
#undef ROTL64
#define ROTL64 skein_rotl64
#endif

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

#define TFBIG_KINIT(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7)) \
			^ SPH_C64(0x1BD11BDAA9FC1A22); \
		t2 = t0 ^ t1; \
	}

#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + (uint64_t)s); \
	}

#define TFBIG_MIX(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROTL64(x1, rc) ^ x0; \
	}

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
	}

#define TFBIG_4e(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
	}

#define TFBIG_4o(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
	}

/* uint64_t midstate for skein 80 */

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

#define TFBIG_4e_PRE(s) { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
	}

#define TFBIG_4o_PRE(s) { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
	}

/* uint2 variant for SM3.2+ */

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


__global__
void quark_skein512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t * const __restrict__ g_hash, uint32_t *g_nonceVector)
{
#if !defined(SP_KERNEL) || __CUDA_ARCH__ < 500
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;

		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		uint32_t hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[hashPosition * 8U];

		// Initialisierung
		h0 = vectorize(0x4903ADFF749C51CEull);
		h1 = vectorize(0x0D95DE399746DF03ull);
		h2 = vectorize(0x8FD1934127C79BCEull);
		h3 = vectorize(0x9A255629FF352CB1ull);
		h4 = vectorize(0x5DB62599DF6CA7B0ull);
		h5 = vectorize(0xEABE394CA9D5C3F4ull);
		h6 = vectorize(0x991112C71A75B523ull);
		h7 = vectorize(0xAE18A40B660FCC33ull);

		uint2 p[8];
		// 1st Round -> etype = 480, ptr = 64, bcount = 0, data = msg
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			p[i] = vectorize(inpHash[i]);

		t0 = vectorize(64); // ptr
		// t1 = vectorize(480ull << 55); // etype
		t1 = vectorize(0xf000000000000000ULL);

//#if CUDA_VERSION >= 7000
		// doesnt really affect x11 perfs.
		__threadfence();
//#endif
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

		h0 = vectorize(inpHash[0]) ^ p[0];
		h1 = vectorize(inpHash[1]) ^ p[1];
		h2 = vectorize(inpHash[2]) ^ p[2];
		h3 = vectorize(inpHash[3]) ^ p[3];
		h4 = vectorize(inpHash[4]) ^ p[4];
		h5 = vectorize(inpHash[5]) ^ p[5];
		h6 = vectorize(inpHash[6]) ^ p[6];
		h7 = vectorize(inpHash[7]) ^ p[7];

		// 2. Runde -> etype = 510, ptr = 8, bcount = 0, data = 0
		#pragma unroll 8
		for(int i=0; i<8; i++)
			p[i] = vectorize(0);

		t0 = vectorize(8); // ptr
		//t1 = vectorize(510ull << 55); // etype
		t1 = vectorize(0xff00000000000000ULL);

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

		// output
		uint64_t *outpHash = &g_hash[hashPosition * 8U];
		#pragma unroll 8
		for(int i=0; i<8; i++)
			outpHash[i] = devectorize(p[i]);
	}
#endif /* SM < 5.0 */
}

__global__
void quark_skein512_gpu_hash_64_sm3(uint32_t threads, uint32_t startNounce, uint64_t * const __restrict__ g_hash, uint32_t *g_nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint64_t p[8];
		uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint64_t t0, t1, t2;

		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		uint32_t hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[hashPosition * 8];

		// Init
		h0 = 0x4903ADFF749C51CEull;
		h1 = 0x0D95DE399746DF03ull;
		h2 = 0x8FD1934127C79BCEull;
		h3 = 0x9A255629FF352CB1ull;
		h4 = 0x5DB62599DF6CA7B0ull;
		h5 = 0xEABE394CA9D5C3F4ull;
		h6 = 0x991112C71A75B523ull;
		h7 = 0xAE18A40B660FCC33ull;

		// 1. Runde -> etype = 480, ptr = 64, bcount = 0, data = msg
		#pragma unroll 8
		for(int i=0; i<8; i++)
			p[i] = inpHash[i];

		t0 = 64; // ptr
		// t1 = 480ull << 55; // etype
		t1 = 0xf000000000000000ULL;

		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		h0 = inpHash[0] ^ p[0];
		h1 = inpHash[1] ^ p[1];
		h2 = inpHash[2] ^ p[2];
		h3 = inpHash[3] ^ p[3];
		h4 = inpHash[4] ^ p[4];
		h5 = inpHash[5] ^ p[5];
		h6 = inpHash[6] ^ p[6];
		h7 = inpHash[7] ^ p[7];

		// 2. Runde -> etype = 510, ptr = 8, bcount = 0, data = 0
		#pragma unroll 8
		for(int i=0; i<8; i++)
			p[i] = 0ull;

		t0 = 8; // ptr
		t1 = 510ull << 55; // etype
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		// output
		uint64_t *outpHash = &g_hash[hashPosition * 8];

		#pragma unroll 8
		for(int i=0; i<8; i++)
			outpHash[i] = p[i];
	}
}

__global__ __launch_bounds__(128,5)
void skein512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64, int swap)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;

		h0 = vectorize(c_PaddedMessage80[10]);
		h1 = vectorize(c_PaddedMessage80[11]);
		h2 = vectorize(c_PaddedMessage80[12]);
		h3 = vectorize(c_PaddedMessage80[13]);
		h4 = vectorize(c_PaddedMessage80[14]);
		h5 = vectorize(c_PaddedMessage80[15]);
		h6 = vectorize(c_PaddedMessage80[16]);
		h7 = vectorize(c_PaddedMessage80[17]);

		t2 = vectorize(c_PaddedMessage80[18]);

		uint32_t nonce = swap ? cuda_swab32(startNounce + thread) : startNounce + thread;
		uint2 nonce2 = make_uint2(_LODWORD(c_PaddedMessage80[9]), nonce);

		uint2 p[8];
		p[0] = vectorize(c_PaddedMessage80[8]);
		p[1] = nonce2;

		#pragma unroll
		for (int i = 2; i < 8; i++)
			p[i] = vectorize(0ull);

		t0 = vectorize(0x50ull);
		t1 = vectorize(0xB000000000000000ull);
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

		uint64_t *outpHash = &output64[thread * 8];
		outpHash[0] = c_PaddedMessage80[8] ^ devectorize(p[0]);
		outpHash[1] = devectorize(nonce2 ^ p[1]);
		#pragma unroll
		for(int i=2; i<8; i++)
			outpHash[i] = devectorize(p[i]);
	}
}

__global__
void skein512_gpu_hash_80_sm3(uint32_t threads, uint32_t startNounce, uint64_t *output64, int swap)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint64_t t0, t1, t2;

		// Init
		h0 = 0x4903ADFF749C51CEull;
		h1 = 0x0D95DE399746DF03ull;
		h2 = 0x8FD1934127C79BCEull;
		h3 = 0x9A255629FF352CB1ull;
		h4 = 0x5DB62599DF6CA7B0ull;
		h5 = 0xEABE394CA9D5C3F4ull;
		h6 = 0x991112C71A75B523ull;
		h7 = 0xAE18A40B660FCC33ull;

		t0 = 64; // ptr
		//t1 = vectorize(0xE0ull << 55); // etype
		t1 = 0x7000000000000000ull;
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);

		uint64_t p[8];
		#pragma unroll 8
		for (int i = 0; i<8; i++)
			p[i] = c_PaddedMessage80[i];

		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		h0 = c_PaddedMessage80[0] ^ p[0];
		h1 = c_PaddedMessage80[1] ^ p[1];
		h2 = c_PaddedMessage80[2] ^ p[2];
		h3 = c_PaddedMessage80[3] ^ p[3];
		h4 = c_PaddedMessage80[4] ^ p[4];
		h5 = c_PaddedMessage80[5] ^ p[5];
		h6 = c_PaddedMessage80[6] ^ p[6];
		h7 = c_PaddedMessage80[7] ^ p[7];

		uint32_t nonce = swap ? cuda_swab32(startNounce + thread) : startNounce + thread;
		uint64_t nonce64 = MAKE_ULONGLONG(_LODWORD(c_PaddedMessage80[9]), nonce);

		// skein_big_close -> etype = 0x160, ptr = 16, bcount = 1, extra = 16
		p[0] = c_PaddedMessage80[8];
		p[1] = nonce64;

		#pragma unroll
		for (int i = 2; i < 8; i++)
			p[i] = 0ull;

		t0 = 0x50ull; // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = 0xB000000000000000ull; // (bcount >> 58) + ((sph_u64)(etype) << 55);

		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		// skein_big_close 2nd loop -> etype = 0x1fe, ptr = 8, bcount = 0
		// output
		uint64_t *outpHash = &output64[thread * 8];
		outpHash[0] = c_PaddedMessage80[8] ^ p[0];
		outpHash[1] = nonce64 ^ p[1];
		#pragma unroll
		for(int i=2; i<8; i++)
			outpHash[i] = p[i];
	}
}

__global__ __launch_bounds__(128,6)
void skein512_gpu_hash_close(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 t0 = vectorize(8); // extra
		uint2 t1 = vectorize(0xFF00000000000000ull); // etype
		uint2 t2 = vectorize(0xB000000000000050ull);

		uint64_t *state = &g_hash[thread * 8];
		uint2 h0 = vectorize(state[0]);
		uint2 h1 = vectorize(state[1]);
		uint2 h2 = vectorize(state[2]);
		uint2 h3 = vectorize(state[3]);
		uint2 h4 = vectorize(state[4]);
		uint2 h5 = vectorize(state[5]);
		uint2 h6 = vectorize(state[6]);
		uint2 h7 = vectorize(state[7]);
		uint2 h8;
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);

		uint2 p[8] = { 0 };

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

		uint64_t *outpHash = state;
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			outpHash[i] = devectorize(p[i]);
	}
}

__global__ __launch_bounds__(128,6)
void skein512_gpu_hash_close_sm3(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t t0 = 8ull; // extra
		uint64_t t1 = 0xFF00000000000000ull; // etype
		uint64_t t2 = 0xB000000000000050ull;

		uint64_t *state = &g_hash[thread * 8];

		uint64_t h0 = state[0];
		uint64_t h1 = state[1];
		uint64_t h2 = state[2];
		uint64_t h3 = state[3];
		uint64_t h4 = state[4];
		uint64_t h5 = state[5];
		uint64_t h6 = state[6];
		uint64_t h7 = state[7];
		uint64_t h8;
		TFBIG_KINIT(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);

		uint64_t p[8] = { 0 };

		TFBIG_4e(0);
		TFBIG_4o(1);
		TFBIG_4e(2);
		TFBIG_4o(3);
		TFBIG_4e(4);
		TFBIG_4o(5);
		TFBIG_4e(6);
		TFBIG_4o(7);
		TFBIG_4e(8);
		TFBIG_4o(9);
		TFBIG_4e(10);
		TFBIG_4o(11);
		TFBIG_4e(12);
		TFBIG_4o(13);
		TFBIG_4e(14);
		TFBIG_4o(15);
		TFBIG_4e(16);
		TFBIG_4o(17);
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint64_t *outpHash = state;
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
			outpHash[i] = p[i];
	}
}

__host__
void quark_skein512_cpu_init(int thr_id, uint32_t threads)
{
	// store the binary SM version
	cuda_get_arch(thr_id);
}

__host__
void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	int dev_id = device_map[thr_id];

	// uint2 uint64 variants for SM 3.2+
#ifdef SP_KERNEL
	if (device_sm[dev_id] >= 500 && cuda_arch[dev_id] >= 500)
		quark_skein512_cpu_hash_64(threads, startNounce, d_nonceVector, d_hash); /* sp.cuh */
	else
#endif
	if (device_sm[dev_id] > 300 && cuda_arch[dev_id] > 300)
		quark_skein512_gpu_hash_64 <<<grid, block>>> (threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	else
		quark_skein512_gpu_hash_64_sm3 <<<grid, block>>> (threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	MyStreamSynchronize(NULL, order, thr_id);
}

/* skein / skein2 */

__host__
static void skein512_precalc_80(uint64_t* message)
{
	uint64_t p[8];
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
	// h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h8 = 0xcab2076d98173ec4ULL;

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
}

__host__
void skein512_cpu_setBlock_80(void *pdata)
{
	uint64_t message[20];
	memcpy(&message[0], pdata, 80);
	skein512_precalc_80(message);
	cudaMemcpyToSymbol(c_PaddedMessage80, message, sizeof(message), 0, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaGetLastError());
}

__host__
void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *g_hash, int swap)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	int dev_id = device_map[thr_id];
	uint64_t *d_hash = (uint64_t*) g_hash;

	if (device_sm[dev_id] > 300 && cuda_arch[dev_id] > 300) {
		// hash function is cut in 2 parts to reduce kernel size
		skein512_gpu_hash_80 <<< grid, block >>> (threads, startNounce, d_hash, swap);
		skein512_gpu_hash_close <<< grid, block >>> (threads, startNounce, d_hash);
	} else {
		// variant without uint2 variables
		skein512_gpu_hash_80_sm3 <<< grid, block >>> (threads, startNounce, d_hash, swap);
		skein512_gpu_hash_close_sm3 <<< grid, block >>> (threads, startNounce, d_hash);
	}
}
