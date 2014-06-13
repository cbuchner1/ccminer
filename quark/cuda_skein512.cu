#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

// Folgende Definitionen später durch header ersetzen
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define SPH_C64(x)    ((uint64_t)(x ## ULL))

// aus cpu-miner.c
extern "C" extern int device_map[8];
// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// Take a look at: https://www.schneier.com/skein1.3.pdf

#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTL64(const uint64_t value, const int offset) {
    uint2 result;
    if(offset >= 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    } else {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    }
    return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif
#define SHL(x, n)			((x) << (n))
#define SHR(x, n)			((x) >> (n))

// Zum testen Hostcode...
/*	Hier erstmal die Tabelle mit den Konstanten für die Mix-Funktion. Kann später vll.
	mal direkt in den Code eingesetzt werden
*/

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

static __constant__ uint64_t d_constMem[8];
static uint64_t h_constMem[8] = {
	SPH_C64(0x4903ADFF749C51CE),
	SPH_C64(0x0D95DE399746DF03),
	SPH_C64(0x8FD1934127C79BCE),
	SPH_C64(0x9A255629FF352CB1),
	SPH_C64(0x5DB62599DF6CA7B0),
	SPH_C64(0xEABE394CA9D5C3F4),
	SPH_C64(0x991112C71A75B523),
	SPH_C64(0xAE18A40B660FCC33) };

__global__ void quark_skein512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint64_t p[8];
		uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint64_t t0, t1, t2;

		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint64_t *inpHash = &g_hash[8 * hashPosition];

		// Initialisierung
		h0 = d_constMem[0];
		h1 = d_constMem[1];
		h2 = d_constMem[2];
		h3 = d_constMem[3];
		h4 = d_constMem[4];
		h5 = d_constMem[5];
		h6 = d_constMem[6];
		h7 = d_constMem[7];

		// 1. Runde -> etype = 480, ptr = 64, bcount = 0, data = msg		
#pragma unroll 8
		for(int i=0;i<8;i++)
			p[i] = inpHash[i];

		t0 = 64; // ptr
		t1 = 480ull << 55; // etype
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
		for(int i=0;i<8;i++)
			p[i] = 0;

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

		// fertig
		uint64_t *outpHash = &g_hash[8 * hashPosition];

#pragma unroll 8
		for(int i=0;i<8;i++)
			outpHash[i] = p[i];
	}
}

// Setup-Funktionen
__host__ void quark_skein512_cpu_init(int thr_id, int threads)
{
	// nix zu tun ;-)
	cudaMemcpyToSymbol( d_constMem,
                        h_constMem,
                        sizeof(h_constMem),
                        0, cudaMemcpyHostToDevice);
}

__host__ void quark_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	quark_skein512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);
}
