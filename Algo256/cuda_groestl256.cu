#include <memory.h>

#include "cuda_helper.h"

uint32_t *d_gnounce[8];
uint32_t *d_GNonce[8];

__constant__ uint32_t pTarget[8];

#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

#define C32e(x) \
	  ((SPH_C32(x) >> 24) \
	| ((SPH_C32(x) >>  8) & SPH_C32(0x0000FF00)) \
	| ((SPH_C32(x) <<  8) & SPH_C32(0x00FF0000)) \
	| ((SPH_C32(x) << 24) & SPH_C32(0xFF000000)))

#define PC32up(j, r)   ((uint32_t)((j) + (r)))
#define PC32dn(j, r)   0
#define QC32up(j, r)   0xFFFFFFFF
#define QC32dn(j, r)   (((uint32_t)(r) << 24) ^ SPH_T32(~((uint32_t)(j) << 24)))

#define B32_0(x)    __byte_perm(x, 0, 0x4440)
//((x) & 0xFF)
#define B32_1(x)    __byte_perm(x, 0, 0x4441)
//(((x) >> 8) & 0xFF)
#define B32_2(x)    __byte_perm(x, 0, 0x4442)
//(((x) >> 16) & 0xFF)
#define B32_3(x)    __byte_perm(x, 0, 0x4443)
//((x) >> 24)

#define MAXWELL_OR_FERMI 1
#if MAXWELL_OR_FERMI
	#define USE_SHARED 1
	// Maxwell and Fermi cards get the best speed with SHARED access it seems.
	#if USE_SHARED
	#define T0up(x) (*((uint32_t*)mixtabs + (    (x))))
	#define T0dn(x) (*((uint32_t*)mixtabs + (256+(x))))
	#define T1up(x) (*((uint32_t*)mixtabs + (512+(x))))
	#define T1dn(x) (*((uint32_t*)mixtabs + (768+(x))))
	#define T2up(x) (*((uint32_t*)mixtabs + (1024+(x))))
	#define T2dn(x) (*((uint32_t*)mixtabs + (1280+(x))))
	#define T3up(x) (*((uint32_t*)mixtabs + (1536+(x))))
	#define T3dn(x) (*((uint32_t*)mixtabs + (1792+(x))))
	#else
	#define T0up(x) tex1Dfetch(t0up2, x)
	#define T0dn(x) tex1Dfetch(t0dn2, x)
	#define T1up(x) tex1Dfetch(t1up2, x)
	#define T1dn(x) tex1Dfetch(t1dn2, x)
	#define T2up(x) tex1Dfetch(t2up2, x)
	#define T2dn(x) tex1Dfetch(t2dn2, x)
	#define T3up(x) tex1Dfetch(t3up2, x)
	#define T3dn(x) tex1Dfetch(t3dn2, x)
	#endif
#else
	#define USE_SHARED 1
	// a healthy mix between shared and textured access provides the highest speed on Compute 3.0 and 3.5!
	#define T0up(x) (*((uint32_t*)mixtabs + (    (x))))
	#define T0dn(x) tex1Dfetch(t0dn2, x)
	#define T1up(x) tex1Dfetch(t1up2, x)
	#define T1dn(x) (*((uint32_t*)mixtabs + (768+(x))))
	#define T2up(x) tex1Dfetch(t2up2, x)
	#define T2dn(x) (*((uint32_t*)mixtabs + (1280+(x))))
	#define T3up(x) (*((uint32_t*)mixtabs + (1536+(x))))
	#define T3dn(x) tex1Dfetch(t3dn2, x)
#endif

texture<unsigned int, 1, cudaReadModeElementType> t0up2;
texture<unsigned int, 1, cudaReadModeElementType> t0dn2;
texture<unsigned int, 1, cudaReadModeElementType> t1up2;
texture<unsigned int, 1, cudaReadModeElementType> t1dn2;
texture<unsigned int, 1, cudaReadModeElementType> t2up2;
texture<unsigned int, 1, cudaReadModeElementType> t2dn2;
texture<unsigned int, 1, cudaReadModeElementType> t3up2;
texture<unsigned int, 1, cudaReadModeElementType> t3dn2;

#define RSTT(d0, d1, a, b0, b1, b2, b3, b4, b5, b6, b7) do { \
	t[d0] = T0up(B32_0(a[b0])) \
		^ T1up(B32_1(a[b1])) \
		^ T2up(B32_2(a[b2])) \
		^ T3up(B32_3(a[b3])) \
		^ T0dn(B32_0(a[b4])) \
		^ T1dn(B32_1(a[b5])) \
		^ T2dn(B32_2(a[b6])) \
		^ T3dn(B32_3(a[b7])); \
	t[d1] = T0dn(B32_0(a[b0])) \
		^ T1dn(B32_1(a[b1])) \
		^ T2dn(B32_2(a[b2])) \
		^ T3dn(B32_3(a[b3])) \
		^ T0up(B32_0(a[b4])) \
		^ T1up(B32_1(a[b5])) \
		^ T2up(B32_2(a[b6])) \
		^ T3up(B32_3(a[b7])); \
	} while (0)


extern uint32_t T0up_cpu[];
extern uint32_t T0dn_cpu[];
extern uint32_t T1up_cpu[];
extern uint32_t T1dn_cpu[];
extern uint32_t T2up_cpu[];
extern uint32_t T2dn_cpu[];
extern uint32_t T3up_cpu[];
extern uint32_t T3dn_cpu[];

__device__ __forceinline__
void groestl256_perm_P(int thread,uint32_t *a, char *mixtabs)
{
	#pragma unroll 10
	for (int r = 0; r<10; r++)
	{
		uint32_t t[16];

		a[0x0] ^= PC32up(0x00, r);
		a[0x2] ^= PC32up(0x10, r);
		a[0x4] ^= PC32up(0x20, r);
		a[0x6] ^= PC32up(0x30, r);
		a[0x8] ^= PC32up(0x40, r);
		a[0xA] ^= PC32up(0x50, r);
		a[0xC] ^= PC32up(0x60, r);
		a[0xE] ^= PC32up(0x70, r);
		RSTT(0x0, 0x1, a, 0x0, 0x2, 0x4, 0x6, 0x9, 0xB, 0xD, 0xF);
		RSTT(0x2, 0x3, a, 0x2, 0x4, 0x6, 0x8, 0xB, 0xD, 0xF, 0x1);
		RSTT(0x4, 0x5, a, 0x4, 0x6, 0x8, 0xA, 0xD, 0xF, 0x1, 0x3);
		RSTT(0x6, 0x7, a, 0x6, 0x8, 0xA, 0xC, 0xF, 0x1, 0x3, 0x5);
		RSTT(0x8, 0x9, a, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7);
		RSTT(0xA, 0xB, a, 0xA, 0xC, 0xE, 0x0, 0x3, 0x5, 0x7, 0x9);
		RSTT(0xC, 0xD, a, 0xC, 0xE, 0x0, 0x2, 0x5, 0x7, 0x9, 0xB);
		RSTT(0xE, 0xF, a, 0xE, 0x0, 0x2, 0x4, 0x7, 0x9, 0xB, 0xD);

		#pragma unroll 16
		for (int k = 0; k<16; k++)
			a[k] = t[k];
	}
}

__device__ __forceinline__
void groestl256_perm_Q(int thread, uint32_t *a, char *mixtabs)
{
	#pragma unroll
	for (int r = 0; r<10; r++)
	{
		uint32_t t[16];

		a[0x0] ^= QC32up(0x00, r);
		a[0x1] ^= QC32dn(0x00, r);
		a[0x2] ^= QC32up(0x10, r);
		a[0x3] ^= QC32dn(0x10, r);
		a[0x4] ^= QC32up(0x20, r);
		a[0x5] ^= QC32dn(0x20, r);
		a[0x6] ^= QC32up(0x30, r);
		a[0x7] ^= QC32dn(0x30, r);
		a[0x8] ^= QC32up(0x40, r);
		a[0x9] ^= QC32dn(0x40, r);
		a[0xA] ^= QC32up(0x50, r);
		a[0xB] ^= QC32dn(0x50, r);
		a[0xC] ^= QC32up(0x60, r);
		a[0xD] ^= QC32dn(0x60, r);
		a[0xE] ^= QC32up(0x70, r);
		a[0xF] ^= QC32dn(0x70, r);
		RSTT(0x0, 0x1, a, 0x2, 0x6, 0xA, 0xE, 0x1, 0x5, 0x9, 0xD);
		RSTT(0x2, 0x3, a, 0x4, 0x8, 0xC, 0x0, 0x3, 0x7, 0xB, 0xF);
		RSTT(0x4, 0x5, a, 0x6, 0xA, 0xE, 0x2, 0x5, 0x9, 0xD, 0x1);
		RSTT(0x6, 0x7, a, 0x8, 0xC, 0x0, 0x4, 0x7, 0xB, 0xF, 0x3);
		RSTT(0x8, 0x9, a, 0xA, 0xE, 0x2, 0x6, 0x9, 0xD, 0x1, 0x5);
		RSTT(0xA, 0xB, a, 0xC, 0x0, 0x4, 0x8, 0xB, 0xF, 0x3, 0x7);
		RSTT(0xC, 0xD, a, 0xE, 0x2, 0x6, 0xA, 0xD, 0x1, 0x5, 0x9);
		RSTT(0xE, 0xF, a, 0x0, 0x4, 0x8, 0xC, 0xF, 0x3, 0x7, 0xB);

		#pragma unroll
		for (int k = 0; k<16; k++)
			a[k] = t[k];
	}
}

__global__ __launch_bounds__(256,1)
void groestl256_gpu_hash32(int threads, uint32_t startNounce, uint64_t *outputHash, uint32_t *nonceVector)
{
#if USE_SHARED
	extern __shared__ char mixtabs[];

	if (threadIdx.x < 256) {
		*((uint32_t*)mixtabs + (threadIdx.x)) = tex1Dfetch(t0up2, threadIdx.x);
		*((uint32_t*)mixtabs + (256 + threadIdx.x)) = tex1Dfetch(t0dn2, threadIdx.x);
		*((uint32_t*)mixtabs + (512 + threadIdx.x)) = tex1Dfetch(t1up2, threadIdx.x);
		*((uint32_t*)mixtabs + (768 + threadIdx.x)) = tex1Dfetch(t1dn2, threadIdx.x);
		*((uint32_t*)mixtabs + (1024 + threadIdx.x)) = tex1Dfetch(t2up2, threadIdx.x);
		*((uint32_t*)mixtabs + (1280 + threadIdx.x)) = tex1Dfetch(t2dn2, threadIdx.x);
		*((uint32_t*)mixtabs + (1536 + threadIdx.x)) = tex1Dfetch(t3up2, threadIdx.x);
		*((uint32_t*)mixtabs + (1792 + threadIdx.x)) = tex1Dfetch(t3dn2, threadIdx.x);
	}

	__syncthreads();
#endif

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// GROESTL
		uint32_t message[16];
		uint32_t state[16];

		#pragma unroll
		for (int k = 0; k<4; k++)
			LOHI(message[2*k], message[2*k+1], outputHash[k*threads+thread]);

		#pragma unroll
		for (int k = 9; k<15; k++)
			message[k] = 0;

		message[8] = 0x80;
		message[15] = 0x01000000;

		#pragma unroll 16
		for (int u = 0; u<16; u++)
			state[u] = message[u];

		state[15] ^= 0x10000;

		// Perm

#if USE_SHARED
		groestl256_perm_P(thread, state, mixtabs);
		state[15] ^= 0x10000;
		groestl256_perm_Q(thread, message, mixtabs);
#else
		groestl256_perm_P(thread, state, NULL);
		state[15] ^= 0x10000;
		groestl256_perm_P(thread, message, NULL);
#endif
		#pragma unroll 16
		for (int u = 0; u<16; u++) state[u] ^= message[u];
		#pragma unroll 16
		for (int u = 0; u<16; u++) message[u] = state[u];
#if USE_SHARED
		groestl256_perm_P(thread, message, mixtabs);
#else
		groestl256_perm_P(thread, message, NULL);
#endif
		state[14] ^= message[14];
		state[15] ^= message[15];

		uint32_t nonce = startNounce + thread;
		if (state[15] <= pTarget[7]) {
			nonceVector[0] = nonce;
		}
	}
}

#define texDef(texname, texmem, texsource, texsize) \
	unsigned int *texmem; \
	cudaMalloc(&texmem, texsize); \
	cudaMemcpy(texmem, texsource, texsize, cudaMemcpyHostToDevice); \
	texname.normalized = 0; \
	texname.filterMode = cudaFilterModePoint; \
	texname.addressMode[0] = cudaAddressModeClamp; \
	{ cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>(); \
	  cudaBindTexture(NULL, &texname, texmem, &channelDesc, texsize ); } \

__host__
void groestl256_cpu_init(int thr_id, int threads)
{

	// Texturen mit obigem Makro initialisieren
	texDef(t0up2, d_T0up, T0up_cpu, sizeof(uint32_t) * 256);
	texDef(t0dn2, d_T0dn, T0dn_cpu, sizeof(uint32_t) * 256);
	texDef(t1up2, d_T1up, T1up_cpu, sizeof(uint32_t) * 256);
	texDef(t1dn2, d_T1dn, T1dn_cpu, sizeof(uint32_t) * 256);
	texDef(t2up2, d_T2up, T2up_cpu, sizeof(uint32_t) * 256);
	texDef(t2dn2, d_T2dn, T2dn_cpu, sizeof(uint32_t) * 256);
	texDef(t3up2, d_T3up, T3up_cpu, sizeof(uint32_t) * 256);
	texDef(t3dn2, d_T3dn, T3dn_cpu, sizeof(uint32_t) * 256);

	cudaMalloc(&d_GNonce[thr_id], sizeof(uint32_t));
	cudaMallocHost(&d_gnounce[thr_id], 1*sizeof(uint32_t));
}

__host__
uint32_t groestl256_cpu_hash_32(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	uint32_t result = 0xffffffff;
	cudaMemset(d_GNonce[thr_id], 0xff, sizeof(uint32_t));
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

#if USE_SHARED
	size_t shared_size = 8 * 256 * sizeof(uint32_t);
#else
	size_t shared_size = 0;
#endif
	groestl256_gpu_hash32<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash, d_GNonce[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);
	cudaMemcpy(d_gnounce[thr_id], d_GNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	result = *d_gnounce[thr_id];

	return result;
}

__host__
void groestl256_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}