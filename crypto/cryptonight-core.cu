#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "cryptonight.h"

#define LONG_SHL32 19 // 1<<19
#define LONG_SHL64 18 // 1<<18 (uint64_t* index)
#define LONG_LOOPS32 0x80000U
#define LONG_LOOPS64 0x40000U

#include "cn_aes.cuh"

__global__
//__launch_bounds__(128, 9) // 56 registers
void cryptonight_core_gpu_phase1(const uint32_t threads, uint64_t * long_state, uint64_t * const ctx_state, uint32_t * ctx_key1)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	const uint32_t sub = (threadIdx.x & 7) << 1; // 0 2 .. 14

	if(thread < threads)
	{
		const uint32_t long_oft = (thread << LONG_SHL64) + sub;

		const uint32_t* ctx_key = &ctx_key1[thread * 40U];
		uint4 keys[10];
		#pragma unroll 10 // load 160 bytes
		for (int i = 0; i < 10; i ++)
			keys[i] = AS_UINT4(&ctx_key[i*4]);

		uint4 text = AS_UINT4(&ctx_state[thread * 26U + sub + 8U]);

		for (uint32_t i = 0; i < LONG_LOOPS64; i += 16U) {
			cn_aes_pseudo_round_mut_uint4(sharedMemory, text, keys);
			AS_UINT4(&long_state[long_oft + i]) = text;
		}
	}
}

__device__ __forceinline__ ulonglong2 cuda_mul128(const uint64_t multiplier, const uint64_t multiplicand)
{
	ulonglong2 product;
	product.x = __umul64hi(multiplier, multiplicand);
	product.y = multiplier * multiplicand;
	return product;
}

static __forceinline__ __device__ void operator += (ulonglong2 &a, const ulonglong2 b) {
	a.x += b.x; a.y += b.y;
}

static __forceinline__ __device__ ulonglong2 operator ^ (const ulonglong2 &a, const ulonglong2 &b) {
	return make_ulonglong2(a.x ^ b.x, a.y ^ b.y);
}

#undef MUL_SUM_XOR_DST
__device__ __forceinline__ void MUL_SUM_XOR_DST(const uint64_t m, uint4 &a, void* far_dst)
{
	ulonglong2 d = AS_UL2(far_dst);
	ulonglong2 p = cuda_mul128(m, d.x);
	p += AS_UL2(&a);
	AS_UL2(&a) = p ^ d;
	AS_UL2(far_dst) = p;
}

__global__
#if __CUDA_ARCH__ >= 500
//__launch_bounds__(128,12) /* force 40 regs to allow -l ...x32 */
#endif
void cryptonight_core_gpu_phase2(const uint32_t threads, const uint32_t bfactor, const uint32_t partidx,
	uint64_t * d_long_state, uint32_t * d_ctx_a, uint32_t * d_ctx_b)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2U + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;

		void * ctx_a = (void*)(&d_ctx_a[thread << 2U]);
		void * ctx_b = (void*)(&d_ctx_b[thread << 2U]);
		uint4 A = AS_UINT4(ctx_a); // ld.global.u32.v4
		uint4 B = AS_UINT4(ctx_b);

		uint64_t * long_state = &d_long_state[thread << LONG_SHL64];
		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;

			uint32_t j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &C);
			AS_UINT4(&long_state[j]) = C ^ B; // st.global.u32.v4
			MUL_SUM_XOR_DST((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3]);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			AS_UINT4(&long_state[j]) = C ^ B;
			MUL_SUM_XOR_DST((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3]);
		}

		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

__global__
void cryptonight_core_gpu_phase3(const uint32_t threads, const uint64_t * long_state, uint64_t * ctx_state, uint32_t * __restrict__ ctx_key2)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3U;
	const uint32_t sub = (threadIdx.x & 7U) << 1U;

	if(thread < threads)
	{
		const uint32_t long_oft = (thread << LONG_SHL64) + sub;
		const uint32_t st_oft = (thread * 26U) + sub + 8U;

		uint4 key[10];
		const uint32_t* ctx_key = &ctx_key2[thread * 40U];
		#pragma unroll 10 // 160 bytes
		for (int i = 0; i < 10; i++)
			key[i] = AS_UINT4(&ctx_key[i*4U]);

		uint4 text = AS_UINT4(&ctx_state[st_oft]);

		for(uint32_t i = 0; i < LONG_LOOPS64; i += 16U)
		{
			uint4 st = AS_UINT4(&long_state[long_oft + i]);
			text = text ^ st;
			cn_aes_pseudo_round_mut_uint4(sharedMemory, text, key);
		}

		AS_UINT4(&ctx_state[st_oft]) = text;
	}
}

extern int device_bfactor[MAX_GPUS];

__host__
void cryptonight_core_cuda(int thr_id, int blocks, int threads, uint64_t *d_long_state, uint64_t *d_ctx_state,
	uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2)
{
	dim3 grid(blocks);
	dim3 block(threads);
	//dim3 block2(threads << 1);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	const uint32_t bfactor = (uint32_t) device_bfactor[thr_id];
	const uint32_t partcount = 1 << bfactor;
	const uint32_t throughput = (uint32_t) (blocks*threads);

	const int bsleep = bfactor ? 100 : 0;
	const int dev_id = device_map[thr_id];

	cryptonight_core_gpu_phase1 <<<grid, block8>>> (throughput, d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	if(partcount > 1) usleep(bsleep);

	for (uint32_t i = 0; i < partcount; i++)
	{
		dim3 b = device_sm[dev_id] >= 300 ? block4 : block;
		cryptonight_core_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		if(partcount > 1) usleep(bsleep);
	}

	cryptonight_core_gpu_phase3 <<<grid, block8>>> (throughput, d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
}
