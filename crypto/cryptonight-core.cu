#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"

#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#undef __shfl
#define __shfl(var, srcLane, width) __shfl_sync(0xFFFFFFFFu, var, srcLane, width)
#endif

#include "cryptonight.h"

#define LONG_SHL32 19 // 1<<19 (uint32_t* index)
#define LONG_SHL64 18 // 1<<18 (uint64_t* index)
#define LONG_LOOPS32 0x80000U

#ifndef _WIN32
#include <unistd.h>
#endif

#include "cn_aes.cuh"

__global__
void cryptonight_gpu_phase1(const uint32_t threads, uint32_t * __restrict__ d_long_state,
	uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1)
{
	__shared__ uint32_t sharedMemory[1024];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	if(thread < threads)
	{
		cn_aes_gpu_init(sharedMemory);
		__syncthreads();

		const int sub = (threadIdx.x & 7) << 2;
		uint32_t *longstate = &d_long_state[(thread << LONG_SHL32) + sub];

		uint32_t key[40], text[4];

		MEMCPY8(key, ctx_key1 + thread * 40, 20);
		MEMCPY8(text, ctx_state + thread * 50 + sub + 16, 2);

		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			cn_aes_pseudo_round_mut(sharedMemory, text, key);
			MEMCPY8(&longstate[i], text, 2);
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

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

__device__ __forceinline__ void MUL_SUM_XOR_DST_0(const uint64_t m, uint4 &a, void* far_dst)
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
void cryptonight_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
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
			MUL_SUM_XOR_DST_0((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3]);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			AS_UINT4(&long_state[j]) = C ^ B;
			MUL_SUM_XOR_DST_0((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3]);
		}

		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

#if 0
#if UINTPTR_MAX == UINT64_MAX
#define LPTR "l"
#else
#define LPTR "r"
#endif

__device__ __forceinline__ uint64_t loadGlobal64(uint64_t * const addr) {
	uint64_t x;
	asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(x) : LPTR (addr));
	return x;
}

__device__ __forceinline__ uint32_t loadGlobal32(uint32_t * const addr) {
	uint32_t x;
	asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : LPTR (addr));
	return x;
}

__device__ __forceinline__ void storeGlobal32(uint32_t* addr, uint32_t const & val) {
	asm volatile("st.global.cg.u32 [%0], %1;" : : LPTR (addr), "r"(val));
}

__device__ __forceinline__ uint32_t variant1_1(const uint32_t src)
{
	const uint32_t tmp = (src >> 24); // __byte_perm(src, 0, 0x7773);
	const uint32_t index = (((tmp >> 3) & 6u) | (tmp & 1u)) << 1;
	return (src & 0x00ffffffu) | ((tmp ^ ((0x75310u >> index) & 0x30u)) << 24);
}

__global__
void monero_phase2_messed(const uint32_t threads, const uint32_t bfactor, const uint32_t partidx,
	uint32_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
	uint64_t * __restrict__ d_tweak)
{
	__shared__ uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread >= threads)
		return;

	const uint32_t batchsize = ITER >> (2U + bfactor);
	const uint32_t start = partidx * batchsize;
	const uint32_t end = start + batchsize;

	const uint32_t subthr = threadIdx.x & 3;
	const uint32_t thrctx = thread << 2;
	uint32_t * ctx_a = &d_ctx_a[thrctx];
	uint32_t * ctx_b = &d_ctx_b[thrctx];
	uint32_t * long_state = &d_long_state[thread << LONG_SHL32];

	uint32_t d[2], t1[2], t2[2];
	uint32_t a = ctx_a[subthr];
	d[1] = ctx_b[subthr];

	uint32_t tweak[2];
	AS_UINT2(&tweak) = AS_UINT2(&d_tweak[thread]);

	for (uint32_t i = start; i < end; i++)
	{
		#pragma unroll 2
		for (int x = 0; x < 2; x++)
		{
			uint32_t j = ((__shfl(a, 0, 4) & 0x1FFFF0) >> 2) + subthr;

			const uint32_t x_0 = loadGlobal32<uint32_t>(long_state + j);
			const uint32_t x_1 = __shfl(x_0, subthr + 1, 4);
			const uint32_t x_2 = __shfl(x_0, subthr + 2, 4);
			const uint32_t x_3 = __shfl(x_0, subthr + 3, 4);

			// t_fn = aes shared mem read
			d[x] = a ^ 	t_fn0(x_0 & 0xff) ^
				t_fn1((x_1 >>  8) & 0xff) ^
				t_fn2((x_2 >> 16) & 0xff) ^
				t_fn3((x_3 >> 24));
			t1[0] = __shfl(d[x], 0, 4);

			uint32_t z = d[0] ^ d[1];
			if (subthr == 2) z = variant1_1(z);
			storeGlobal32(long_state + j, z);

			// -----------------------------------------------------------
			j = ((*t1 & 0x1FFFF0) >> 2) + subthr;

			uint32_t yy[2];
			AS_U64(yy) = loadGlobal64<uint64_t>(((uint64_t *)long_state) + (j >> 1));

			t1[1] = __shfl(d[x], 1, 4);

			uint32_t sub2 = (threadIdx.x & 2);
			t2[0] = __shfl(a, sub2, 4);
			t2[1] = __shfl(a, sub2 + 1U, 4);

			uint32_t zz[2];
			zz[0] = __shfl(yy[0], 0, 4);
			zz[1] = __shfl(yy[1], 0, 4);

			AS_U64(t2) += sub2 ? (AS_U64(t1) * AS_U64(zz)) : __umul64hi(AS_U64(t1), AS_U64(zz));

			uint32_t s1 = subthr & 1U;
			z = AS_U64(t2) >> (s1 * 32U); // hi or lo dword
			//z = __byte_perm(t2[0], t2[1], s1 ? 0x7654 : 0x3210);
			storeGlobal32(long_state + j, sub2 ? tweak[s1] ^ z : z);
			a = z ^ yy[s1];
		}
	}

	if (bfactor) {
		ctx_a[subthr] = a;
		ctx_b[subthr] = d[1];
	}
}
#endif

__device__ __forceinline__ void store_variant1(uint64_t* long_state, uint4 Z)
{
	const uint32_t tmp = (Z.z >> 24); // __byte_perm(src, 0, 0x7773);
	const uint32_t index = (((tmp >> 3) & 6u) | (tmp & 1u)) << 1;
	Z.z = (Z.z & 0x00ffffffu) | ((tmp ^ ((0x75310u >> index) & 0x30u)) << 24);
	AS_UINT4(long_state) = Z;
}

__device__ __forceinline__ void MUL_SUM_XOR_DST_1(const uint64_t m, uint4 &a, void* far_dst, uint64_t tweak)
{
	ulonglong2 d = AS_UL2(far_dst);
	ulonglong2 p = cuda_mul128(m, d.x);
	p += AS_UL2(&a);
	AS_UL2(&a) = p ^ d;
	p.y = p.y ^ tweak;
	AS_UL2(far_dst) = p;
}

__global__
void monero_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
	uint64_t * __restrict__ d_tweak)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;

		uint64_t tweak = d_tweak[thread];

		void * ctx_a = (void*)(&d_ctx_a[thread << 2]);
		void * ctx_b = (void*)(&d_ctx_b[thread << 2]);
		uint4 A = AS_UINT4(ctx_a); // ld.global.u32.v4
		uint4 B = AS_UINT4(ctx_b);

		uint64_t * long_state = &d_long_state[thread << LONG_SHL64];
		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;

			uint32_t j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &C);
			store_variant1(&long_state[j], C ^ B); // st.global.u32.v4
			MUL_SUM_XOR_DST_1((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3], tweak);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			store_variant1(&long_state[j], C ^ B); // st.global.u32.v4
			MUL_SUM_XOR_DST_1((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3], tweak);
		}

		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__global__
void cryptonight_gpu_phase3(const uint32_t threads, const uint32_t * __restrict__ d_long_state,
	uint32_t * __restrict__ d_ctx_state, const uint32_t * __restrict__ d_ctx_key2)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;

	if(thread < threads)
	{
		const int sub = (threadIdx.x & 7) << 2;
		const uint32_t *longstate = &d_long_state[(thread << LONG_SHL32) + sub];
		uint32_t key[40], text[4];
		MEMCPY8(key, d_ctx_key2 + thread * 40, 20);
		MEMCPY8(text, d_ctx_state + thread * 50 + sub + 16, 2);

		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			#pragma unroll
			for(int j = 0; j < 4; ++j)
				text[j] ^= longstate[i + j];

			cn_aes_pseudo_round_mut(sharedMemory, text, key);
		}

		MEMCPY8(d_ctx_state + thread * 50 + sub + 16, text, 2);
	}
}

// --------------------------------------------------------------------------------------------------------------

extern int device_bfactor[MAX_GPUS];

__host__
void cryptonight_core_cuda(int thr_id, int blocks, int threads, uint64_t *d_long_state, uint32_t *d_ctx_state,
	uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint64_t *d_ctx_tweak)
{
	dim3 grid(blocks);
	dim3 block(threads);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	const uint16_t bfactor = (uint16_t) device_bfactor[thr_id];
	const uint32_t partcount = 1U << bfactor;
	const uint32_t throughput = (uint32_t) (blocks*threads);

	const int bsleep = bfactor ? 100 : 0;
	const int dev_id = device_map[thr_id];

	cryptonight_gpu_phase1 <<<grid, block8>>> (throughput, (uint32_t*) d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	if(partcount > 1) usleep(bsleep);

	for (uint32_t i = 0; i < partcount; i++)
	{
		dim3 b = device_sm[dev_id] >= 300 ? block4 : block;
		if (variant == 0)
			cryptonight_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b);
		else
			monero_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b, d_ctx_tweak);
			//monero_phase2_messed <<<grid, b>>> (throughput, bfactor, i, (uint32_t*) d_long_state, d_ctx_a, d_ctx_b, d_ctx_tweak);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		if(partcount > 1) usleep(bsleep);
	}
	//cudaDeviceSynchronize();
	//exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	cryptonight_gpu_phase3 <<<grid, block8>>> (throughput, (uint32_t*) d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
}
