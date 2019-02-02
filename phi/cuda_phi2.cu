#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

__global__ __launch_bounds__(128, 8)
void phi_filter_gpu(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_branch2, uint32_t* d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U; // 64U / sizeof(uint32_t);
		uint4 *psrc = (uint4*) (&d_hash[offset]);
		d_NonceBranch[thread] = ((uint8_t*)psrc)[0] & 1;
		if (d_NonceBranch[thread]) return;
		if (d_branch2) {
			uint4 *pdst = (uint4*)(&d_branch2[offset]);
			uint4 data;
			data = psrc[0]; pdst[0] = data;
			data = psrc[1]; pdst[1] = data;
			data = psrc[2]; pdst[2] = data;
			data = psrc[3]; pdst[3] = data;
		}
	}
}

__global__ __launch_bounds__(128, 8)
void phi_merge_gpu(const uint32_t threads, uint32_t* d_hash, uint32_t* d_branch2, uint32_t* const d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads && !d_NonceBranch[thread])
	{
		const uint32_t offset = thread * 16U;
		uint4 *psrc = (uint4*) (&d_branch2[offset]);
		uint4 *pdst = (uint4*) (&d_hash[offset]);
		uint4 data;
		data = psrc[0]; pdst[0] = data;
		data = psrc[1]; pdst[1] = data;
		data = psrc[2]; pdst[2] = data;
		data = psrc[3]; pdst[3] = data;
	}
}

__global__
void phi_final_compress_gpu(const uint32_t threads, uint32_t* d_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U;
		uint2 *psrc = (uint2*) (&d_hash[offset]);
		uint2 *pdst = (uint2*) (&d_hash[offset]);
		uint2 data;
		data = psrc[4]; pdst[0] ^= data;
		data = psrc[5]; pdst[1] ^= data;
		data = psrc[6]; pdst[2] ^= data;
		data = psrc[7]; pdst[3] ^= data;
	}
}

__host__
uint32_t phi_filter_cuda(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_br2, uint32_t* d_nonces)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// extract algo permution hashes to a second branch buffer
	phi_filter_gpu <<<grid, block>>> (threads, inpHashes, d_br2, d_nonces);
	return threads;
}

__host__
void phi_merge_cuda(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_br2, uint32_t* d_nonces)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// put back second branch hashes to the common buffer d_hash
	phi_merge_gpu <<<grid, block>>> (threads, outpHashes, d_br2, d_nonces);
}

__host__
void phi_final_compress_cuda(const int thr_id, const uint32_t threads, uint32_t *d_hashes)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	phi_final_compress_gpu <<<grid, block>>> (threads, d_hashes);
}
