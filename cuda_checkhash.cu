/**
 * This code compares final hash against target
 */
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

__constant__ uint32_t pTarget[8];

static uint32_t *d_resNounce[8];
static uint32_t *h_resNounce[8];

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

/* --------------------------------------------------------------------------------------------- */

__device__ __forceinline__
static bool hashbelowtarget(const uint32_t *const __restrict__ hash, const uint32_t *const __restrict__ target)
{
	if (hash[7] > target[7])
		return false;
	if (hash[7] < target[7])
		return true;
	if (hash[6] > target[6])
		return false;
	if (hash[6] < target[6])
		return true;

	if (hash[5] > target[5])
		return false;
	if (hash[5] < target[5])
		return true;
	if (hash[4] > target[4])
		return false;
	if (hash[4] < target[4])
		return true;

	if (hash[3] > target[3])
		return false;
	if (hash[3] < target[3])
		return true;
	if (hash[2] > target[2])
		return false;
	if (hash[2] < target[2])
		return true;

	if (hash[1] > target[1])
		return false;
	if (hash[1] < target[1])
		return true;
	if (hash[0] > target[0])
		return false;

	return true;
}

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_64(int threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// shl 4 = *16 x 4 (uint32) = 64 bytes
		uint32_t *inpHash = &hash[thread << 4];

		if (hashbelowtarget(inpHash, pTarget)) {
			uint32_t nounce = (startNounce + thread);
			resNounce[0] = nounce;
		}
	}
}

__host__
uint32_t cuda_check_hash(int thr_id, int threads, uint32_t startNounce, uint32_t *d_inputHash)
{
	uint32_t result = 0xffffffff;
	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	const int threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	cuda_checkhash_64 <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNounce[thr_id]);

	cudaThreadSynchronize();

	cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_resNounce[thr_id];

	return result;
}

/* --------------------------------------------------------------------------------------------- */

__global__
void cuda_check_hash_branch_64(int threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = g_nonceVector[thread];
		uint32_t hashPosition = (nounce - startNounce) << 4;
		uint32_t *inpHash = &g_hash[hashPosition];
		//uint32_t hash[8];

		//#pragma unroll 8
		//for (int i=0; i < 8; i++)
		//	hash[i] = inpHash[i];

		for (int i = 7; i >= 0; i--) {
			if (inpHash[i] > pTarget[i]) {
				return;
			}
			if (inpHash[i] < pTarget[i]) {
				break;
			}
		}
		if (resNounce[0] > nounce)
			resNounce[0] = nounce;
	}
}

__host__
uint32_t cuda_check_hash_branch(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order)
{
	uint32_t result = 0xffffffff;
	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_hash_branch_64 <<<grid, block>>> (threads, startNounce, d_nonceVector, d_inputHash, d_resNounce[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	result = *h_resNounce[thr_id];

	return result;
}