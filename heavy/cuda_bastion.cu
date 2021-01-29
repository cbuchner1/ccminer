#include <stdio.h>

#include "cuda_helper.h"

static uint32_t *d_offsets1[MAX_GPUS] = { 0 };
static uint32_t *d_offsets2[MAX_GPUS] = { 0 };

static uint32_t *d_brcount1[MAX_GPUS] = { 0 };
static uint32_t *d_brcount2[MAX_GPUS] = { 0 };

__global__ __launch_bounds__(128, 6)
void bastion_filter2_gpu(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_hash1, uint32_t* d_hash2, uint32_t* d_br_ofts1, uint32_t* d_count1, uint32_t* d_br_ofts2, uint32_t* d_count2)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U; // 64U / sizeof(uint32_t);
		uint4 *psrc = (uint4*) (&d_hash[offset]);
		uint4 *pdst;
		d_br_ofts1[thread] = 0;
		d_br_ofts2[thread] = 0;
		if (((uint8_t*)psrc)[0] & 0x8) {
			// uint4 = 4x uint32_t = 16 bytes
			uint32_t oft = atomicAdd(d_count1, 1U) * 16U;
			d_br_ofts1[thread] = oft + 16U;
			pdst = (uint4*) (&d_hash1[oft]);
		} else {
			uint32_t oft = atomicAdd(d_count2, 1U) * 16U;
			d_br_ofts2[thread] = oft + 16U;
			pdst = (uint4*) (&d_hash2[oft]);
		}
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__global__ __launch_bounds__(128, 6)
void bastion_merge2_gpu(const uint32_t threads, uint32_t* d_hash, uint32_t* d_hash1, uint32_t* d_hash2, uint32_t* d_br_ofts1, uint32_t* d_br_ofts2)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U;
		uint4 *pdst = (uint4*) (&d_hash[offset]);
		uint4 *psrc;
		if (d_br_ofts1[thread]) {
			const uint32_t oft = d_br_ofts1[thread] - 16U;
			psrc = (uint4*) (&d_hash1[oft]);
		} else {
			const uint32_t oft = d_br_ofts2[thread] - 16U;
			psrc = (uint4*) (&d_hash2[oft]);
		}
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}


__host__
void bastion_init(const int thr_id, const uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_offsets1[thr_id], sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&d_offsets2[thr_id], sizeof(uint32_t) * threads));
	CUDA_SAFE_CALL(cudaMalloc(&d_brcount1[thr_id], sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc(&d_brcount2[thr_id], sizeof(uint32_t)));
}

__host__
void bastion_free(const int thr_id)
{
	cudaFree(d_offsets1[thr_id]);
	cudaFree(d_offsets2[thr_id]);
	cudaFree(d_brcount1[thr_id]);
	cudaFree(d_brcount2[thr_id]);
}

__host__
uint32_t bastion_filter2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_hash1, uint32_t* d_hash2)
{
	uint32_t num = 0;
	cudaMemset(d_brcount1[thr_id], 0, 4);
	cudaMemset(d_brcount2[thr_id], 0, 4);
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	bastion_filter2_gpu <<<grid, block>>> (threads, inpHashes, d_hash1, d_hash2, d_offsets1[thr_id], d_brcount1[thr_id], d_offsets2[thr_id], d_brcount2[thr_id]);
	cudaMemcpy(&num, d_brcount1[thr_id], 4, cudaMemcpyDeviceToHost);
	return num;
}

__host__
void bastion_merge2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_hash1, uint32_t* d_hash2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// put back branch hashes to the common buffer d_hash
	bastion_merge2_gpu <<<grid, block>>> (threads, outpHashes, d_hash1, d_hash2, d_offsets1[thr_id], d_offsets2[thr_id]);
}
