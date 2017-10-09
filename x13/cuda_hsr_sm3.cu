#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <miner.h>

#define  F(x, y, z) (((x) ^ (y) ^ (z)))
#define FF(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG(x, y, z) ((z)  ^ ((x) & ((y) ^ (z))))

#define P0(x) x ^ ROTL32(x,  9) ^ ROTL32(x, 17)
#define P1(x) x ^ ROTL32(x, 15) ^ ROTL32(x, 23)

static __forceinline__ __device__
void sm3_compress2(uint32_t digest[8], const uint32_t pblock[16])
{
	uint32_t tt1, tt2, i, t, ss1, ss2, x, y;
	uint32_t w[68];
	uint32_t a = digest[0];
	uint32_t b = digest[1];
	uint32_t c = digest[2];
	uint32_t d = digest[3];
	uint32_t e = digest[4];
	uint32_t f = digest[5];
	uint32_t g = digest[6];
	uint32_t h = digest[7];

	#pragma unroll
	for (i = 0; i<16; i++) {
		w[i] = cuda_swab32(pblock[i]);
	}

	for (i = 16; i<68; i++) {
		x = ROTL32(w[i - 3], 15);
		y = ROTL32(w[i - 13], 7);

		x ^= w[i - 16];
		x ^= w[i - 9];
		y ^= w[i - 6];

		w[i] = P1(x) ^ y;
	}

	for (i = 0; i<64; i++) {

		t = (i < 16) ? 0x79cc4519 : 0x7a879d8a;

		ss2 = ROTL32(a, 12);
		ss1 = ROTL32(ss2 + e + ROTL32(t, i), 7);
		ss2 ^= ss1;

		tt1 = d + ss2 + (w[i] ^ w[i + 4]);
		tt2 = h + ss1 + w[i];

		if (i < 16) {
			tt1 += F(a, b, c);
			tt2 += F(e, f, g);
		}
		else {
			tt1 += FF(a, b, c);
			tt2 += GG(e, f, g);
		}
		d = c;
		c = ROTL32(b, 9);
		b = a;
		a = tt1;
		h = g;
		g = ROTL32(f, 19);
		f = e;
		e = P0(tt2);
	}

	digest[0] ^= a;
	digest[1] ^= b;
	digest[2] ^= c;
	digest[3] ^= d;
	digest[4] ^= e;
	digest[5] ^= f;
	digest[6] ^= g;
	digest[7] ^= h;
}

/***************************************************/
// GPU Hash Function
__global__
void sm3_gpu_hash_64(const uint32_t threads, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		const size_t hashPosition = thread;

		uint32_t digest[8];
		digest[0] = 0x7380166F;
		digest[1] = 0x4914B2B9;
		digest[2] = 0x172442D7;
		digest[3] = 0xDA8A0600;
		digest[4] = 0xA96F30BC;
		digest[5] = 0x163138AA;
		digest[6] = 0xE38DEE4D;
		digest[7] = 0xB0FB0E4E;

		uint32_t *pHash = &g_hash[hashPosition << 4];
		sm3_compress2(digest, pHash);

		uint32_t block[16];
		block[0] = 0x80;

		#pragma unroll
		for (int i = 1; i < 14; i++)
			block[i] = 0;

		// count
		block[14] = cuda_swab32(1 >> 23);
		block[15] = cuda_swab32((1 << 9) + (0 << 3));

		sm3_compress2(digest, block);

		for (int i = 0; i < 8; i++)
			pHash[i] = cuda_swab32(digest[i]);

		for (int i = 8; i < 16; i++)
			pHash[i] = 0;
	}
}

__host__
void sm3_cuda_hash_64(int thr_id, uint32_t threads, uint32_t *g_hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	sm3_gpu_hash_64 <<<grid, block>>>(threads, g_hash);
	//MyStreamSynchronize(NULL, order, thr_id);
}
