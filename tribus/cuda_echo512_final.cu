/**
 *  Based on Provos Alexis work - 2016 FOR SM 5+
 *
 *  final touch by tpruvot for tribus - 09 2017
 */
#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>
#include <cuda_vectors.h>

#define INTENSIVE_GMF
#include "tribus/cuda_echo512_aes.cuh"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#define atomicExch(p,y) (*p) = y
#endif

__device__
static void echo_round(const uint32_t sharedMemory[4][256], uint32_t *W, uint32_t &k0)
{
	// Big Sub Words
	#pragma unroll 16
	for (int idx = 0; idx < 16; idx++)
		AES_2ROUND(sharedMemory,W[(idx<<2) + 0], W[(idx<<2) + 1], W[(idx<<2) + 2], W[(idx<<2) + 3], k0);

	// Shift Rows
	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		uint32_t t[4];
		/// 1, 5, 9, 13
		t[0] = W[i +  4];
		t[1] = W[i +  8];
		t[2] = W[i + 24];
		t[3] = W[i + 60];

		W[i +  4] = W[i + 20];
		W[i +  8] = W[i + 40];
		W[i + 24] = W[i + 56];
		W[i + 60] = W[i + 44];

		W[i + 20] = W[i + 36];
		W[i + 40] = t[1];
		W[i + 56] = t[2];
		W[i + 44] = W[i + 28];

		W[i + 28] = W[i + 12];
		W[i + 12] = t[3];
		W[i + 36] = W[i + 52];
		W[i + 52] = t[0];
	}
	// Mix Columns
	#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		#pragma unroll 4
		for (int idx = 0; idx < 64; idx += 16)
		{
			uint32_t a[4];
			a[0] = W[idx + i];
			a[1] = W[idx + i + 4];
			a[2] = W[idx + i + 8];
			a[3] = W[idx + i +12];

			uint32_t ab = a[0] ^ a[1];
			uint32_t bc = a[1] ^ a[2];
			uint32_t cd = a[2] ^ a[3];

			uint32_t t, t2, t3;
			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[idx + i] = bc ^ a[3] ^ abx;
			W[idx + i + 4] = a[0] ^ cd ^ bcx;
			W[idx + i + 8] = ab ^ a[3] ^ cdx;
			W[idx + i +12] = ab ^ a[2] ^ (abx ^ bcx ^ cdx);
		}
	}
}

__global__ __launch_bounds__(256, 3) /* will force 80 registers */
static void tribus_echo512_gpu_final(uint32_t threads, uint64_t *g_hash, uint32_t* resNonce, const uint64_t target)
{
	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init256(sharedMemory);

	const uint32_t P[48] = {
		0xe7e9f5f5, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0xa4213d7e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		//8-12
		0x01425eb8, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0x65978b09, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		//21-25
		0x2cb6b661, 0x6b23b3b3, 0xcf93a7cf, 0x9d9d3751,0x9ac2dea3, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		//34-38
		0x579f9f33, 0xfbfbfbfb, 0xfbfbfbfb, 0xefefd3c7,0xdbfde1dd, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,
		0x34514d9e, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0xb134347e, 0xea6f7e7e, 0xbd7731bd, 0x8a8a1968,
		0x14b8a457, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af,0x265f4382, 0xf5e7e9f5, 0xb3b36b23, 0xb3dbe7af
		//58-61
	};
	uint32_t k0;
	uint32_t h[16];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t *hash = (uint32_t*)&g_hash[thread<<3];

		*(uint2x4*)&h[0] = __ldg4((uint2x4*)&hash[0]);
		*(uint2x4*)&h[8] = __ldg4((uint2x4*)&hash[8]);

		uint64_t backup = *(uint64_t*)&h[6];

		k0 = 512 + 8;

		#pragma unroll 4
		for (uint32_t idx = 0; idx < 16; idx += 4)
			AES_2ROUND(sharedMemory,h[idx + 0], h[idx + 1], h[idx + 2], h[idx + 3], k0);

		k0 += 4;

		uint32_t W[64];

		#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++)
		{
			uint32_t a = P[i];
			uint32_t b = P[i + 4];
			uint32_t c = h[i + 8];
			uint32_t d = P[i + 8];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;

			uint32_t t =  ((a ^ b) & 0x80808080);
			uint32_t t2 = ((b ^ c) & 0x80808080);
			uint32_t t3 = ((c ^ d) & 0x80808080);

			uint32_t abx = ((t  >> 7) * 27U) ^ ((ab^t) << 1);
			uint32_t bcx = ((t2 >> 7) * 27U) ^ ((bc^t2) << 1);
			uint32_t cdx = ((t3 >> 7) * 27U) ^ ((cd^t3) << 1);

			W[0 + i] = bc ^ d ^ abx;
			W[4 + i] = a ^ cd ^ bcx;
			W[8 + i] = ab ^ d ^ cdx;
			W[12+ i] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[12 + i];
			b = h[i + 4];
			c = P[12 + i + 4];
			d = P[12 + i + 8];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[16 + i] = abx ^ bc ^ d;
			W[16 + i + 4] = bcx ^ a ^ cd;
			W[16 + i + 8] = cdx ^ ab ^ d;
			W[16 + i +12] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = h[i];
			b = P[24 + i];
			c = P[24 + i + 4];
			d = P[24 + i + 8];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[32 + i] = abx ^ bc ^ d;
			W[32 + i + 4] = bcx ^ a ^ cd;
			W[32 + i + 8] = cdx ^ ab ^ d;
			W[32 + i +12] = abx ^ bcx ^ cdx ^ ab ^ c;

			a = P[36 + i ];
			b = P[36 + i + 4];
			c = P[36 + i + 8];
			d = h[i + 12];

			ab = a ^ b;
			bc = b ^ c;
			cd = c ^ d;

			t  = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			abx = (t  >> 7) * 27U ^ ((ab^t) << 1);
			bcx = (t2 >> 7) * 27U ^ ((bc^t2) << 1);
			cdx = (t3 >> 7) * 27U ^ ((cd^t3) << 1);

			W[48 + i] = abx ^ bc ^ d;
			W[48 + i + 4] = bcx ^ a ^ cd;
			W[48 + i + 8] = cdx ^ ab ^ d;
			W[48 + i +12] = abx ^ bcx ^ cdx ^ ab ^ c;
		}

		for (int k = 1; k < 9; k++)
			echo_round(sharedMemory,W,k0);

		// Big Sub Words
		uint32_t y0, y1, y2, y3;
//		AES_2ROUND(sharedMemory,W[ 0], W[ 1], W[ 2], W[ 3], k0);
		aes_round(sharedMemory, W[ 0], W[ 1], W[ 2], W[ 3], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[ 0], W[ 1], W[ 2], W[ 3]);

		aes_round(sharedMemory, W[ 4], W[ 5], W[ 6], W[ 7], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[ 4], W[ 5], W[ 6], W[ 7]);
		aes_round(sharedMemory, W[ 8], W[ 9], W[10], W[11], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[ 8], W[ 9], W[10], W[11]);

		aes_round(sharedMemory, W[20], W[21], W[22], W[23], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[20], W[21], W[22], W[23]);
		aes_round(sharedMemory, W[28], W[29], W[30], W[31], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[28], W[29], W[30], W[31]);

		aes_round(sharedMemory, W[32], W[33], W[34], W[35], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[32], W[33], W[34], W[35]);
		aes_round(sharedMemory, W[40], W[41], W[42], W[43], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[40], W[41], W[42], W[43]);

		aes_round(sharedMemory, W[52], W[53], W[54], W[55], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[52], W[53], W[54], W[55]);
		aes_round(sharedMemory, W[60], W[61], W[62], W[63], k0, y0, y1, y2, y3);
		aes_round(sharedMemory, y0, y1, y2, y3, W[60], W[61], W[62], W[63]);

		uint32_t bc = W[22] ^ W[42];
		uint32_t t2 = (bc & 0x80808080);
		W[ 6] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		bc = W[23] ^ W[43];
		t2 = (bc & 0x80808080);
		W[ 7] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		bc = W[10] ^ W[54];
		t2 = (bc & 0x80808080);
		W[38] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		bc = W[11] ^ W[55];
		t2 = (bc & 0x80808080);
		W[39] = (t2 >> 7) * 27U ^ ((bc^t2) << 1);

		uint64_t check = backup ^ *(uint64_t*)&W[2] ^ *(uint64_t*)&W[6] ^ *(uint64_t*)&W[10] ^ *(uint64_t*)&W[30]
			^ *(uint64_t*)&W[34] ^ *(uint64_t*)&W[38] ^ *(uint64_t*)&W[42] ^ *(uint64_t*)&W[62];

		if(check <= target){
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}
	}
}

__host__
void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t *d_resNonce, const uint64_t target)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	tribus_echo512_gpu_final <<<grid, block>>> (threads, (uint64_t*)d_hash, d_resNonce, target);
}
