#include <stdio.h>

#include "cuda_helper.h"

// globaler Speicher für unsere Ergebnisse
static uint32_t *d_hashoutput[MAX_GPUS];
extern uint32_t *d_hash2output[MAX_GPUS];
extern uint32_t *d_hash3output[MAX_GPUS];
extern uint32_t *d_hash4output[MAX_GPUS];
extern uint32_t *d_hash5output[MAX_GPUS];

extern uint32_t *heavy_nonceVector[MAX_GPUS];

/* Combines top 64-bits from each hash into a single hash */
__device__
static void combine_hashes(uint32_t *out, uint32_t *hash1, uint32_t *hash2, uint32_t *hash3, uint32_t *hash4)
{
	uint32_t lout[8]; // Combining in Registern machen

#pragma unroll 8
	for (int i=0; i < 8; ++i)
		lout[i] = 0;

	// das Makro setzt jeweils 4 Bits aus vier verschiedenen Hashes zu einem Nibble zusammen
#define MIX(bits, mask, i) \
	lout[(255 - (bits+3))/32] <<= 4; \
	if ((hash1[i] & mask) != 0) lout[(255 - (bits+0))/32] |= 8; \
	if ((hash2[i] & mask) != 0) lout[(255 - (bits+1))/32] |= 4; \
	if ((hash3[i] & mask) != 0) lout[(255 - (bits+2))/32] |= 2; \
	if ((hash4[i] & mask) != 0) lout[(255 - (bits+3))/32] |= 1; \

	/* Transpose first 64 bits of each hash into out */
	MIX(  0, 0x80000000, 7);
	MIX(  4, 0x40000000, 7);
	MIX(  8, 0x20000000, 7);
	MIX( 12, 0x10000000, 7);
	MIX( 16, 0x08000000, 7);
	MIX( 20, 0x04000000, 7);
	MIX( 24, 0x02000000, 7);
	MIX( 28, 0x01000000, 7);
	MIX( 32, 0x00800000, 7);
	MIX( 36, 0x00400000, 7);
	MIX( 40, 0x00200000, 7);
	MIX( 44, 0x00100000, 7);
	MIX( 48, 0x00080000, 7);
	MIX( 52, 0x00040000, 7);
	MIX( 56, 0x00020000, 7);
	MIX( 60, 0x00010000, 7);
	MIX( 64, 0x00008000, 7);
	MIX( 68, 0x00004000, 7);
	MIX( 72, 0x00002000, 7);
	MIX( 76, 0x00001000, 7);
	MIX( 80, 0x00000800, 7);
	MIX( 84, 0x00000400, 7);
	MIX( 88, 0x00000200, 7);
	MIX( 92, 0x00000100, 7);
	MIX( 96, 0x00000080, 7);
	MIX(100, 0x00000040, 7);
	MIX(104, 0x00000020, 7);
	MIX(108, 0x00000010, 7);
	MIX(112, 0x00000008, 7);
	MIX(116, 0x00000004, 7);
	MIX(120, 0x00000002, 7);
	MIX(124, 0x00000001, 7);

	MIX(128, 0x80000000, 6);
	MIX(132, 0x40000000, 6);
	MIX(136, 0x20000000, 6);
	MIX(140, 0x10000000, 6);
	MIX(144, 0x08000000, 6);
	MIX(148, 0x04000000, 6);
	MIX(152, 0x02000000, 6);
	MIX(156, 0x01000000, 6);
	MIX(160, 0x00800000, 6);
	MIX(164, 0x00400000, 6);
	MIX(168, 0x00200000, 6);
	MIX(172, 0x00100000, 6);
	MIX(176, 0x00080000, 6);
	MIX(180, 0x00040000, 6);
	MIX(184, 0x00020000, 6);
	MIX(188, 0x00010000, 6);
	MIX(192, 0x00008000, 6);
	MIX(196, 0x00004000, 6);
	MIX(200, 0x00002000, 6);
	MIX(204, 0x00001000, 6);
	MIX(208, 0x00000800, 6);
	MIX(212, 0x00000400, 6);
	MIX(216, 0x00000200, 6);
	MIX(220, 0x00000100, 6);
	MIX(224, 0x00000080, 6);
	MIX(228, 0x00000040, 6);
	MIX(232, 0x00000020, 6);
	MIX(236, 0x00000010, 6);
	MIX(240, 0x00000008, 6);
	MIX(244, 0x00000004, 6);
	MIX(248, 0x00000002, 6);
	MIX(252, 0x00000001, 6);

#pragma unroll 8
	for (int i=0; i < 8; ++i)
		out[i] = lout[i];
}

__global__
void combine_gpu_hash(uint32_t threads, uint32_t startNounce, uint32_t *out, uint32_t *hash2, uint32_t *hash3, uint32_t *hash4, uint32_t *hash5, uint32_t *nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = nonceVector[thread];
		uint32_t hashPosition = nounce - startNounce;
		// Die Aufgabe der combine-funktion besteht aus zwei Teilen.
		// 1) Komprimiere die hashes zu einem kleinen Array
		// 2) Errechne dort den combines-value

		// Die Kompression wird dadurch verwirklicht, dass im out-array weiterhin mit "thread" indiziert
		// wird. Die anderen Werte werden mit der nonce indiziert

		combine_hashes(&out[8 * thread], &hash2[8 * hashPosition], &hash3[16 * hashPosition], &hash4[16 * hashPosition], &hash5[16 * hashPosition]);
	}
}

__host__
void combine_cpu_init(int thr_id, uint32_t threads)
{
	// Speicher für alle Ergebnisse belegen
	CUDA_SAFE_CALL(cudaMalloc(&d_hashoutput[thr_id], (size_t) 32 * threads));
}

__host__
void combine_cpu_free(int thr_id)
{
	cudaFree(d_hashoutput[thr_id]);
}

__host__
void combine_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *hash)
{
	// diese Kopien sind optional, da die Hashes jetzt bereits auf der GPU liegen sollten

	const uint32_t threadsperblock = 128;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	combine_gpu_hash <<<grid, block>>> (threads, startNounce, d_hashoutput[thr_id], d_hash2output[thr_id], d_hash3output[thr_id], d_hash4output[thr_id], d_hash5output[thr_id], heavy_nonceVector[thr_id]);

	// da die Hash Auswertung noch auf der CPU erfolgt, müssen die Ergebnisse auf jeden Fall zum Host kopiert werden
	CUDA_SAFE_CALL(cudaMemcpy(hash, d_hashoutput[thr_id], 8 * sizeof(uint32_t) * threads, cudaMemcpyDeviceToHost));
}
