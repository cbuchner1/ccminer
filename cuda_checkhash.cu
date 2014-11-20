#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// Hash Target gegen das wir testen sollen
__constant__ uint32_t pTarget[8];

static uint32_t *d_resNounce[8];
static uint32_t *h_resNounce[8];

__global__
void cuda_check_gpu_hash_64(int threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		uint32_t hashPosition = (nounce - startNounce) << 4;
		uint32_t *inpHash = &g_hash[hashPosition];
		uint32_t hash[8];

		#pragma unroll 8
		for (int i=0; i < 8; i++)
			hash[i] = inpHash[i];

		for (int i = 7; i >= 0; i--) {
			if (hash[i] > pTarget[i]) {
				return;
			}
			if (hash[i] <= pTarget[i]) {
				break;
			}
		}
		if (resNounce[0] > nounce)
			resNounce[0] = nounce;
	}
}

// Setup-Funktionen
__host__
void cuda_check_cpu_init(int thr_id, int threads)
{
    CUDA_CALL_OR_RET(cudaMallocHost(&h_resNounce[thr_id], 1*sizeof(uint32_t)));
    CUDA_CALL_OR_RET(cudaMalloc(&d_resNounce[thr_id], 1*sizeof(uint32_t)));
}

// Target Difficulty
__host__
void cuda_check_cpu_setTarget(const void *ptarget)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

__host__
uint32_t cuda_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order)
{
	uint32_t result = 0xffffffff;
	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_gpu_hash_64 <<<grid, block>>> (threads, startNounce, d_nonceVector, d_inputHash, d_resNounce[thr_id]);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);

	// Ergebnis zum Host kopieren (in page locked memory, damits schneller geht)
	cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// cudaMemcpy() ist asynchron!
	cudaThreadSynchronize();
	result = *h_resNounce[thr_id];

	return result;
}

__global__
void cuda_check_gpu_hash_fast(int threads, uint32_t startNounce, uint32_t *hashEnd, uint32_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		/* only test the last 2 dwords, ok for most algos */
		int hashPos = thread << 4;
		uint32_t *inpHash = &hashEnd[hashPos];

		if (inpHash[7] <= pTarget[7] && inpHash[6] <= pTarget[6]) {
			uint32_t nounce = (startNounce + thread);
			if (resNounce[0] > nounce)
				resNounce[0] = nounce;
		}
	}
}

__host__
uint32_t cuda_check_hash_fast(int thr_id, int threads, uint32_t startNounce, uint32_t *d_inputHash, int order)
{
	uint32_t result = 0xffffffff;
	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_gpu_hash_fast <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNounce[thr_id]);

	// MyStreamSynchronize(NULL, order, thr_id);
	cudaThreadSynchronize();

	cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// cudaMemcpy() was asynchron ?
	// cudaThreadSynchronize();
	result = *h_resNounce[thr_id];

	return result;
}
