/**
 * Blake-256 Cuda Kernel (Tested on SM 5/5.2)
 *
 * Tanguy Pruvot / SP - Jan 2016
 */

#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

/* threads per block */
#define TPB 512

/* hash by cpu with blake 256 */
extern "C" void blake256hash(void *output, const void *input, int8_t rounds = 14)
{
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(rounds);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

__constant__ uint32_t _ALIGN(32) d_data[12];

/* 8 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* max count of found nonces in one call */
#define NBN 2
static __thread uint32_t extra_results[NBN] = { UINT32_MAX };

#define GSPREC(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ c_u256[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ c_u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
	}

__device__ __forceinline__
void blake256_compress_14(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t /*_ALIGN(8)*/ m[16];
	uint32_t v[16];

	m[0] = block[0];
	m[1] = block[1];
	m[2] = block[2];
	m[3] = block[3];

	const uint32_t c_u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	const uint32_t c_Padding[12] = {
		0x80000000UL, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640,
	};

	#pragma unroll
	for (uint32_t i = 0; i < 12; i++) {
		m[i+4] = c_Padding[i];
	}

	//#pragma unroll 8
	for(uint32_t i = 0; i < 8; i++)
		v[i] = h[i];

	v[ 8] = c_u256[0];
	v[ 9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC,0,1);
	GSPREC(1, 5, 0x9, 0xD,2,3);
	GSPREC(2, 6, 0xA, 0xE, 4,5);
	GSPREC(3, 7, 0xB, 0xF, 6,7);
	GSPREC(0, 5, 0xA, 0xF, 8,9);
	GSPREC(1, 6, 0xB, 0xC, 10,11);
	GSPREC(2, 7, 0x8, 0xD, 12,13);
	GSPREC(3, 4, 0x9, 0xE, 14,15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);
	//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	GSPREC(0, 4, 0x8, 0xC, 9, 0);
	GSPREC(1, 5, 0x9, 0xD, 5, 7);
	GSPREC(2, 6, 0xA, 0xE, 2, 4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6, 8);
	GSPREC(3, 4, 0x9, 0xE, 3, 13);
	//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15, 14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);
	//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14, 13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0, 7);
	GSPREC(1, 6, 0xB, 0xC, 6, 3);
	GSPREC(2, 7, 0x8, 0xD, 9, 2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);
	//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	GSPREC(0, 4, 0x8, 0xC, 13, 11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3, 9);
	GSPREC(0, 5, 0xA, 0xF, 5, 0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8, 6);
	GSPREC(3, 4, 0x9, 0xE, 2, 10);
	//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	GSPREC(0, 4, 0x8, 0xC, 6, 15);
	GSPREC(1, 5, 0x9, 0xD, 14, 9);
	GSPREC(2, 6, 0xA, 0xE, 11, 3);
	GSPREC(3, 7, 0xB, 0xF, 0, 8);
	GSPREC(0, 5, 0xA, 0xF, 12, 2);
	GSPREC(1, 6, 0xB, 0xC, 13, 7);
	GSPREC(2, 7, 0x8, 0xD, 1, 4);
	GSPREC(3, 4, 0x9, 0xE, 10, 5);
	//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	GSPREC(0, 4, 0x8, 0xC, 10, 2);
	GSPREC(1, 5, 0x9, 0xD, 8, 4);
	GSPREC(2, 6, 0xA, 0xE, 7, 6);
	GSPREC(3, 7, 0xB, 0xF, 1, 5);
	GSPREC(0, 5, 0xA, 0xF, 15, 11);
	GSPREC(1, 6, 0xB, 0xC, 9, 14);
	GSPREC(2, 7, 0x8, 0xD, 3, 12);
	GSPREC(3, 4, 0x9, 0xE, 13, 0);
	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC, 0, 1);
	GSPREC(1, 5, 0x9, 0xD, 2, 3);
	GSPREC(2, 6, 0xA, 0xE, 4, 5);
	GSPREC(3, 7, 0xB, 0xF, 6, 7);
	GSPREC(0, 5, 0xA, 0xF, 8, 9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);

	// only compute h6 & 7
	h[6U] ^= v[6U] ^ v[14U];
	h[7U] ^= v[7U] ^ v[15U];
}

/* ############################################################################################################################### */
/* Precalculated 1st 64-bytes block (midstate) method */

__global__ __launch_bounds__(1024,1)
void blake256_gpu_hash_16(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t _ALIGN(16) h[8];

		#pragma unroll
		for(int i=0; i < 8; i++) {
			h[i] = d_data[i];
		}

		// ------ Close: Bytes 64 to 80 ------

		uint32_t _ALIGN(16) ending[4];
		ending[0] = d_data[8];
		ending[1] = d_data[9];
		ending[2] = d_data[10];
		ending[3] = nonce; /* our tested value */

		blake256_compress_14(h, ending, 640);

		if (h[7] == 0 && cuda_swab32(h[6]) <= highTarget) {
#if NBN == 2
			if (resNonce[0] != UINT32_MAX)
				resNonce[1] = nonce;
			else
				resNonce[0] = nonce;
#else
			resNonce[0] = nonce;
#endif
		}
	}
}

__global__
#if __CUDA_ARCH__ >= 500
__launch_bounds__(512, 3) /* 40 regs */
#endif
void blake256_gpu_hash_16_8(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t h[8];
		const uint32_t nonce = startNonce + thread;

		#pragma unroll
		for (int i = 0; i < 8; i++) {
			h[i] = d_data[i];
		}

		// ------ Close: Bytes 64 to 80 ------

		uint32_t m[16] = {
			d_data[8], d_data[9], d_data[10], nonce,
			0x80000000UL, 0, 0, 0,
			0, 0, 0, 0,
			0, 1, 0, 640,
		};

		const uint32_t c_u256[16] = {
			0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
			0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
			0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
			0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
		};

		uint32_t v[16];

		#pragma unroll
		for (uint32_t i = 0; i < 8; i++)
			v[i] = h[i];

		v[8]  = c_u256[0];
		v[9]  = c_u256[1];
		v[10] = c_u256[2];
		v[11] = c_u256[3];

		v[12] = c_u256[4] ^ 640U;
		v[13] = c_u256[5] ^ 640U;
		v[14] = c_u256[6];
		v[15] = c_u256[7];

		//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		GSPREC(0, 4, 0x8, 0xC, 0, 1);
		GSPREC(1, 5, 0x9, 0xD, 2, 3);
		GSPREC(2, 6, 0xA, 0xE, 4, 5);
		GSPREC(3, 7, 0xB, 0xF, 6, 7);
		GSPREC(0, 5, 0xA, 0xF, 8, 9);
		GSPREC(1, 6, 0xB, 0xC, 10, 11);
		GSPREC(2, 7, 0x8, 0xD, 12, 13);
		GSPREC(3, 4, 0x9, 0xE, 14, 15);
		//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 0x8, 0xC, 14, 10);
		GSPREC(1, 5, 0x9, 0xD, 4, 8);
		GSPREC(2, 6, 0xA, 0xE, 9, 15);
		GSPREC(3, 7, 0xB, 0xF, 13, 6);
		GSPREC(0, 5, 0xA, 0xF, 1, 12);
		GSPREC(1, 6, 0xB, 0xC, 0, 2);
		GSPREC(2, 7, 0x8, 0xD, 11, 7);
		GSPREC(3, 4, 0x9, 0xE, 5, 3);
		//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 0x8, 0xC, 11, 8);
		GSPREC(1, 5, 0x9, 0xD, 12, 0);
		GSPREC(2, 6, 0xA, 0xE, 5, 2);
		GSPREC(3, 7, 0xB, 0xF, 15, 13);
		GSPREC(0, 5, 0xA, 0xF, 10, 14);
		GSPREC(1, 6, 0xB, 0xC, 3, 6);
		GSPREC(2, 7, 0x8, 0xD, 7, 1);
		GSPREC(3, 4, 0x9, 0xE, 9, 4);
		//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 0x8, 0xC, 7, 9);
		GSPREC(1, 5, 0x9, 0xD, 3, 1);
		GSPREC(2, 6, 0xA, 0xE, 13, 12);
		GSPREC(3, 7, 0xB, 0xF, 11, 14);
		GSPREC(0, 5, 0xA, 0xF, 2, 6);
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(2, 7, 0x8, 0xD, 4, 0);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);
		//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 0x8, 0xC, 9, 0);
		GSPREC(1, 5, 0x9, 0xD, 5, 7);
		GSPREC(2, 6, 0xA, 0xE, 2, 4);
		GSPREC(3, 7, 0xB, 0xF, 10, 15);
		GSPREC(0, 5, 0xA, 0xF, 14, 1);
		GSPREC(1, 6, 0xB, 0xC, 11, 12);
		GSPREC(2, 7, 0x8, 0xD, 6, 8);
		GSPREC(3, 4, 0x9, 0xE, 3, 13);
		//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 0x8, 0xC, 2, 12);
		GSPREC(1, 5, 0x9, 0xD, 6, 10);
		GSPREC(2, 6, 0xA, 0xE, 0, 11);
		GSPREC(3, 7, 0xB, 0xF, 8, 3);
		GSPREC(0, 5, 0xA, 0xF, 4, 13);
		GSPREC(1, 6, 0xB, 0xC, 7, 5);
		GSPREC(2, 7, 0x8, 0xD, 15, 14);
		GSPREC(3, 4, 0x9, 0xE, 1, 9);
		//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 0x8, 0xC, 12, 5);
		GSPREC(1, 5, 0x9, 0xD, 1, 15);
		GSPREC(2, 6, 0xA, 0xE, 14, 13);
		GSPREC(3, 7, 0xB, 0xF, 4, 10);
		GSPREC(0, 5, 0xA, 0xF, 0, 7);
		GSPREC(1, 6, 0xB, 0xC, 6, 3);
		GSPREC(2, 7, 0x8, 0xD, 9, 2);
		GSPREC(3, 4, 0x9, 0xE, 8, 11);
		//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 0x8, 0xC, 13, 11);
		GSPREC(1, 5, 0x9, 0xD, 7, 14);
		GSPREC(2, 6, 0xA, 0xE, 12, 1);
		GSPREC(3, 7, 0xB, 0xF, 3, 9);
		GSPREC(0, 5, 0xA, 0xF, 5, 0);
		GSPREC(1, 6, 0xB, 0xC, 15, 4);
		GSPREC(2, 7, 0x8, 0xD, 8, 6);
		//GSPREC(3, 4, 0x9, 0xE, 2, 10);
		//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },

		// only compute h6 & 7
		//h[6] ^= v[6] ^ v[14];
		//h[7] ^= v[7] ^ v[15];

		if ((h[7]^v[7]^v[15]) == 0) // h7
		{
			GSPREC(3, 4, 0x9, 0xE, 2, 10);
			if (cuda_swab32(h[6]^v[6]^v[14]) <= highTarget) {
#if NBN == 2
				if (resNonce[0] != UINT32_MAX)
					resNonce[1] = nonce;
				else
					resNonce[0] = nonce;
#else
				resNonce[0] = nonce;
#endif
			}
		}
	}
}

__host__
static uint32_t blake256_cpu_hash_16(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint64_t highTarget,
	const int8_t rounds)
{
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	if (rounds == 8)
		blake256_gpu_hash_16_8 <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id], highTarget);
	else
		blake256_gpu_hash_16  <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id], highTarget);

	if (cudaSuccess == cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = h_resNonce[thr_id][0];
		for (int n=0; n < (NBN-1); n++)
			extra_results[n] = h_resNonce[thr_id][n+1];
	}
	return result;
}

__host__
static void blake256mid(uint32_t *output, const uint32_t *input, int8_t rounds = 14)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(rounds);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 64);

	memcpy(output, (void*)ctx.H, 32);
}

__host__
void blake256_cpu_setBlock_16(uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget)
{
	uint32_t _ALIGN(64) data[11];
	memcpy(data, midstate, 32);
	data[8] = penddata[0];
	data[9] = penddata[1];
	data[10]= penddata[2];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, data, 32 + 12, 0, cudaMemcpyHostToDevice));
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int8_t blakerounds=14)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t _ALIGN(64) midstate[8];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	uint64_t targetHigh = ((uint64_t*)ptarget)[3];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 30 : 26;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	int rc = 0;

	if (opt_benchmark) {
		targetHigh = 0x1ULL << 32;
		ptarget[6] = swab32(0xff);
	}

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256mid(midstate, endiandata, blakerounds);
	blake256_cpu_setBlock_16(&pdata[16], midstate, ptarget);

	do {
		// GPU HASH (second block only, first is midstate)
		work->nonces[0] = blake256_cpu_hash_16(thr_id, throughput, pdata[19], targetHigh, blakerounds);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhashcpu[8];
			const uint32_t Htarg = ptarget[6];

			for (int k=16; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			be32enc(&endiandata[19], work->nonces[0]);
			blake256hash(vhashcpu, endiandata, blakerounds);

			if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhashcpu);
#if NBN > 1
				if (extra_results[0] != UINT32_MAX) {
					work->nonces[1] = extra_results[0];
					be32enc(&endiandata[19], work->nonces[1]);
					blake256hash(vhashcpu, endiandata, blakerounds);
					if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget)) {
						if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio[0]) {
							work_set_target_ratio(work, vhashcpu);
							xchg(work->nonces[0], work->nonces[1]);
						} else {
							bn_set_target_ratio(work, vhashcpu, 1);
						}
						work->valid_nonces = 2;
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
					extra_results[0] = UINT32_MAX;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
#endif
				return work->valid_nonces;
			}
			else if (vhashcpu[6] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + pdata[19]);

	*hashes_done = pdata[19] - first_nonce;

	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return rc;
}

// cleanup
extern "C" void free_blake256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

