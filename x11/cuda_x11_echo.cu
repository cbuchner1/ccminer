#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

#include "cuda_x11_aes.cu"

__device__ __forceinline__ void AES_2ROUND(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3,
	uint32_t &k0, uint32_t &k1, uint32_t &k2, uint32_t &k3)
{
	uint32_t y0, y1, y2, y3;
	
	aes_round(sharedMemory,
		x0, x1, x2, x3,
		k0,
		y0, y1, y2, y3);

	aes_round(sharedMemory,
		y0, y1, y2, y3,
		x0, x1, x2, x3);

	// hier werden wir ein carry brauchen (oder auch nicht)
	k0++;
}

__device__ __forceinline__ void cuda_echo_round(
	const uint32_t *sharedMemory,
	uint32_t &k0, uint32_t &k1, uint32_t &k2, uint32_t &k3,
	uint32_t *W, int round)
{
	// W hat 16*4 als Abmaße

	// Big Sub Words
#pragma unroll 16
	for(int i=0;i<16;i++)
	{
		int idx = i<<2; // *4
		AES_2ROUND(sharedMemory,
			W[idx+0], W[idx+1], W[idx+2], W[idx+3],
			k0, k1, k2, k3);
	}

	// Shift Rows
#pragma unroll 4
	for(int i=0;i<4;i++)
	{
		uint32_t t;

		/// 1, 5, 9, 13
		t = W[4 + i];
		W[4 + i] = W[20 + i];
		W[20 + i] = W[36 + i];
		W[36 + i] = W[52 + i];
		W[52 + i] = t;

		// 2, 6, 10, 14
		t = W[8 + i];
		W[8 + i] = W[40 + i];
		W[40 + i] = t;
		t = W[24 + i];
		W[24 + i] = W[56 + i];
		W[56 + i] = t;

		// 15, 11, 7, 3
		t = W[60 + i];
		W[60 + i] = W[44 + i];
		W[44 + i] = W[28 + i];
		W[28 + i] = W[12 + i];
		W[12 + i] = t;
	}

	// Mix Columns
	#pragma unroll
	for (int i = 0; i<4; i++) // Schleife über je 2*uint32_t
	{
		#pragma unroll 64
		for (int idx = 0; idx<64; idx += 16) // Schleife über die elemnte
		{

			uint32_t a = W[idx + i];
			uint32_t b = W[idx + i + 4];
			uint32_t c = W[idx + i + 8];
			uint32_t d = W[idx + i + 12];

			uint32_t ab = a ^ b;
			uint32_t bc = b ^ c;
			uint32_t cd = c ^ d;

			uint32_t t, t2, t3;
			t = (ab & 0x80808080);
			t2 = (bc & 0x80808080);
			t3 = (cd & 0x80808080);

			uint32_t abx = (t >> 7) * 27 ^ ((ab^t) << 1);
			uint32_t bcx = (t2 >> 7) * 27 ^ ((bc^t2) << 1);
			uint32_t cdx = (t3 >> 7) * 27 ^ ((cd^t3) << 1);

			W[idx + i] = abx ^ bc ^ d;
			W[idx + i + 4] = bcx ^ a ^ cd;
			W[idx + i + 8] = cdx ^ ab ^ d;
			W[idx + i + 12] = abx ^ bcx ^ cdx ^ ab ^ c;
		}
	}

}

__global__ void x11_echo512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__shared__ uint32_t sharedMemory[1024];

	aes_gpu_init(sharedMemory);

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		uint32_t W[64];
		uint32_t k0 = 512, k1 = 0, k2 = 0, k3 = 0; // K0 = bitlen
		/* Initialisierung */
#pragma unroll 8
		for(int i=0;i<32;i+=4)
		{
			W[i + 0] = 512;
			W[i + 1] = 0;
			W[i + 2] = 0;
			W[i + 3] = 0;
		}

		// kopiere 32-byte großen hash
#pragma unroll 16
		for(int i=0;i<16;i++)
			W[i+32] = Hash[i];
		W[48] = 0x80; // fest
#pragma unroll 10
		for(int i=49;i<59;i++)
			W[i] = 0;
		W[59] = 0x02000000; // fest
		W[60] = k0; // bitlen
		W[61] = k1;
		W[62] = k2;
		W[63] = k3;
		
		for(int i=0;i<10;i++)
		{
			cuda_echo_round(sharedMemory, k0, k1, k2, k3, W, i);
		}

#pragma unroll 8
		for(int i=0;i<32;i+=4)
		{
			W[i  ] ^= W[32 + i    ] ^ 512;
			W[i+1] ^= W[32 + i + 1];
			W[i+2] ^= W[32 + i + 2];
			W[i+3] ^= W[32 + i + 3];
		}

#pragma unroll 16
		for(int i=0;i<16;i++)
			W[i] ^= Hash[i];

		// tsiv: I feel	iffy about removing	this, but it seems to break	the	full hash
		// fortunately for X11 the flipped bit lands outside the first 32 bytes	used as	the	final X11 hash
		// try chaining	more algos after echo (X13)	and	boom
		//W[8] ^= 0x10;

		W[27] ^= 0x02000000;
		W[28] ^= k0;

#pragma unroll 16
		for(int i=0;i<16;i++)
			Hash[i] = W[i];
    }
}

// Setup-Funktionen
__host__ void x11_echo512_cpu_init(int thr_id, int threads)
{
	aes_cpu_init();
}

__host__ void x11_echo512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

//    fprintf(stderr, "threads=%d, %d blocks, %d threads per block, %d bytes shared\n", threads, grid.x, block.x, shared_size);

    x11_echo512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    MyStreamSynchronize(NULL, order, thr_id);
}
