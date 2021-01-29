// SM 2.x variant (tpruvot)

#ifdef __INTELLISENSE__
//#define __CUDA_ARCH__ 210
#define __CUDACC__
#include <cuda_helper.h>
#include <cuda_texture_types.h>
#define __byte_perm(a,b,c) (a)
#define tex1Dfetch(t, n) (n)
#endif

#define USE_SHARED 1

static unsigned int *d_textures[MAX_GPUS][8];

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

#define T0up(x) (*((uint32_t*)mixtabs + (     (x))))
#define T0dn(x) (*((uint32_t*)mixtabs + ( 256+(x))))
#define T1up(x) (*((uint32_t*)mixtabs + ( 512+(x))))
#define T1dn(x) (*((uint32_t*)mixtabs + ( 768+(x))))
#define T2up(x) (*((uint32_t*)mixtabs + (1024+(x))))
#define T2dn(x) (*((uint32_t*)mixtabs + (1280+(x))))
#define T3up(x) (*((uint32_t*)mixtabs + (1536+(x))))
#define T3dn(x) (*((uint32_t*)mixtabs + (1792+(x))))

texture<unsigned int, 1, cudaReadModeElementType> t0up1;
texture<unsigned int, 1, cudaReadModeElementType> t0dn1;
texture<unsigned int, 1, cudaReadModeElementType> t1up1;
texture<unsigned int, 1, cudaReadModeElementType> t1dn1;
texture<unsigned int, 1, cudaReadModeElementType> t2up1;
texture<unsigned int, 1, cudaReadModeElementType> t2dn1;
texture<unsigned int, 1, cudaReadModeElementType> t3up1;
texture<unsigned int, 1, cudaReadModeElementType> t3dn1;

extern uint32_t T0up_cpu[];
extern uint32_t T0dn_cpu[];
extern uint32_t T1up_cpu[];
extern uint32_t T1dn_cpu[];
extern uint32_t T2up_cpu[];
extern uint32_t T2dn_cpu[];
extern uint32_t T3up_cpu[];
extern uint32_t T3dn_cpu[];

#if __CUDA_ARCH__ < 300 || defined(_DEBUG)

#if (!USE_SHARED)
#include "groestl_simple.cuh"
#endif

__device__ __forceinline__
void quark_groestl512_perm_P(uint32_t *a, char *mixtabs)
{
	#pragma unroll 1
	for(int r=0; r<14; r++)
	{
		uint32_t t[32];

		#pragma unroll 16
		for (int k=0; k<16; k++)
			a[(k*2)+0] ^= PC32up(k<< 4, r);

		#pragma unroll 16
		for(int k=0;k<32;k+=2) {
			uint32_t t0_0 = B32_0(a[(k    ) & 0x1f]), t9_0  = B32_0(a[(k +  9) & 0x1f]);
			uint32_t t2_1 = B32_1(a[(k + 2) & 0x1f]), t11_1 = B32_1(a[(k + 11) & 0x1f]);
			uint32_t t4_2 = B32_2(a[(k + 4) & 0x1f]), t13_2 = B32_2(a[(k + 13) & 0x1f]);
			uint32_t t6_3 = B32_3(a[(k + 6) & 0x1f]), t23_3 = B32_3(a[(k + 23) & 0x1f]);

			t[k + 0] =  T0up( t0_0 ) ^ T1up(  t2_1 ) ^ T2up(  t4_2 ) ^ T3up(  t6_3 ) ^
						T0dn( t9_0 ) ^ T1dn( t11_1 ) ^ T2dn( t13_2 ) ^ T3dn( t23_3 );

			t[k + 1] =  T0dn( t0_0 ) ^ T1dn(  t2_1 ) ^ T2dn(  t4_2 ) ^ T3dn(  t6_3 ) ^
						T0up( t9_0 ) ^ T1up( t11_1 ) ^ T2up( t13_2 ) ^ T3up( t23_3 );
		}

		#pragma unroll 32
		for(int k=0; k<32; k++)
			a[k] = t[k];
	}
}

__device__ __forceinline__
void quark_groestl512_perm_Q(uint32_t *a, char *mixtabs)
{
	#pragma unroll 1
	for(int r=0; r<14; r++)
	{
		uint32_t t[32];

		#pragma unroll 16
		for (int k=0; k<16; k++) {
			a[(k*2)+0] ^= QC32up(k << 4, r);
			a[(k*2)+1] ^= QC32dn(k << 4, r);
		}

		#pragma unroll 16
		for(int k=0;k<32;k+=2)
		{
			uint32_t t2_0  = B32_0(a[(k +  2) & 0x1f]), t1_0  = B32_0(a[(k +  1) & 0x1f]);
			uint32_t t6_1  = B32_1(a[(k +  6) & 0x1f]), t5_1  = B32_1(a[(k +  5) & 0x1f]);
			uint32_t t10_2 = B32_2(a[(k + 10) & 0x1f]), t9_2  = B32_2(a[(k +  9) & 0x1f]);
			uint32_t t22_3 = B32_3(a[(k + 22) & 0x1f]), t13_3 = B32_3(a[(k + 13) & 0x1f]);

			t[k + 0] =  T0up( t2_0 ) ^ T1up( t6_1 ) ^ T2up( t10_2 ) ^ T3up( t22_3 ) ^
						T0dn( t1_0 ) ^ T1dn( t5_1 ) ^ T2dn(  t9_2 ) ^ T3dn( t13_3 );

			t[k + 1] =  T0dn( t2_0 ) ^ T1dn( t6_1 ) ^ T2dn( t10_2 ) ^ T3dn( t22_3 ) ^
						T0up( t1_0 ) ^ T1up( t5_1 ) ^ T2up(  t9_2 ) ^ T3up( t13_3 );
		}
		#pragma unroll 32
		for(int k=0; k<32; k++)
			a[k] = t[k];
	}
}

#endif

__global__
void quark_groestl512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_hash, uint32_t *g_nonceVector)
{
#if __CUDA_ARCH__ < 300 || defined(_DEBUG)

#if USE_SHARED
	__shared__ char mixtabs[8 * 1024];
	if (threadIdx.x < 256) {
		*((uint32_t*)mixtabs + (     threadIdx.x)) = tex1Dfetch(t0up1, threadIdx.x);
		*((uint32_t*)mixtabs + ( 256+threadIdx.x)) = tex1Dfetch(t0dn1, threadIdx.x);
		*((uint32_t*)mixtabs + ( 512+threadIdx.x)) = tex1Dfetch(t1up1, threadIdx.x);
		*((uint32_t*)mixtabs + ( 768+threadIdx.x)) = tex1Dfetch(t1dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (1024+threadIdx.x)) = tex1Dfetch(t2up1, threadIdx.x);
		*((uint32_t*)mixtabs + (1280+threadIdx.x)) = tex1Dfetch(t2dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (1536+threadIdx.x)) = tex1Dfetch(t3up1, threadIdx.x);
		*((uint32_t*)mixtabs + (1792+threadIdx.x)) = tex1Dfetch(t3dn1, threadIdx.x);
	}
	__syncthreads();
#endif

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// GROESTL
		uint32_t message[32];
		uint32_t state[32];

		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		off_t hashPosition = nounce - startNounce;
		uint32_t *pHash = &g_hash[hashPosition * 16];

		#pragma unroll 4
		for (int i=0; i<16; i += 4)
			AS_UINT4(&message[i]) = AS_UINT4(&pHash[i]);

		message[16] = 0x80U;
		#pragma unroll 14
		for(int i=17; i<31; i++) message[i] = 0;
		message[31] = 0x01000000U;

		#pragma unroll 32
		for(int i=0; i<32; i++) state[i] = message[i];
		state[31] ^= 0x20000U;

		// Perm
#if USE_SHARED
		quark_groestl512_perm_P(state, mixtabs);
		state[31] ^= 0x20000U;
		quark_groestl512_perm_Q(message, mixtabs);
		#pragma unroll 32
		for(int i=0; i<32; i++) state[i] ^= message[i];
		#pragma unroll 16
		for(int i=16; i<32; i++) message[i] = state[i];
		quark_groestl512_perm_P(state, mixtabs);
#else
		tex_groestl512_perm_P(state);
		state[31] ^= 0x20000U;
		tex_groestl512_perm_Q(message);
		#pragma unroll 32
		for(int i=0; i<32; i++) state[i] ^= message[i];
		#pragma unroll 16
		for(int i=16; i<32; i++) message[i] = state[i];
		tex_groestl512_perm_P(state);
#endif

		#pragma unroll 16
		for(int i=16; i<32; i++) state[i] ^= message[i];

		uint4 *outpt = (uint4*)(pHash);
		uint4 *phash = (uint4*)(&state[16]);
		outpt[0] = phash[0];
		outpt[1] = phash[1];
		outpt[2] = phash[2];
		outpt[3] = phash[3];
	}
#endif
}

#define texDef(id, texname, texmem, texsource, texsize) { \
	unsigned int *texmem; \
	cudaMalloc(&texmem, texsize); \
	d_textures[thr_id][id] = texmem; \
	cudaMemcpy(texmem, texsource, texsize, cudaMemcpyHostToDevice); \
	texname.normalized = 0; \
	texname.filterMode = cudaFilterModePoint; \
	texname.addressMode[0] = cudaAddressModeClamp; \
	{ cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>(); \
	  cudaBindTexture(NULL, &texname, texmem, &channelDesc, texsize ); \
	} \
}

__host__
void quark_groestl512_sm20_init(int thr_id, uint32_t threads)
{
	// Texturen mit obigem Makro initialisieren
	texDef(0, t0up1, d_T0up, T0up_cpu, sizeof(uint32_t)*256);
	texDef(1, t0dn1, d_T0dn, T0dn_cpu, sizeof(uint32_t)*256);
	texDef(2, t1up1, d_T1up, T1up_cpu, sizeof(uint32_t)*256);
	texDef(3, t1dn1, d_T1dn, T1dn_cpu, sizeof(uint32_t)*256);
	texDef(4, t2up1, d_T2up, T2up_cpu, sizeof(uint32_t)*256);
	texDef(5, t2dn1, d_T2dn, T2dn_cpu, sizeof(uint32_t)*256);
	texDef(6, t3up1, d_T3up, T3up_cpu, sizeof(uint32_t)*256);
	texDef(7, t3dn1, d_T3dn, T3dn_cpu, sizeof(uint32_t)*256);
}

__host__
void quark_groestl512_sm20_free(int thr_id)
{
	if (!d_textures[thr_id][0]) return;
	for (int i=0; i<8; i++)
		cudaFree(d_textures[thr_id][i]);
	d_textures[thr_id][0] = NULL;
}

__host__
void quark_groestl512_sm20_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	int threadsperblock = 512;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	quark_groestl512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
}

__host__
void quark_doublegroestl512_sm20_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	int threadsperblock = 512;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	quark_groestl512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
	quark_groestl512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
}

// --------------------------------------------------------------------------------------------------------------------------------------------

#ifdef WANT_GROESTL80

// defined in groest512.cu
// __constant__ static uint32_t c_Message80[20];

__global__
//__launch_bounds__(256)
void groestl512_gpu_hash_80_sm2(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
#if __CUDA_ARCH__ < 300 || defined(_DEBUG)

#if USE_SHARED
	__shared__ char mixtabs[8 * 1024];
	if (threadIdx.x < 256) {
		*((uint32_t*)mixtabs + (     threadIdx.x)) = tex1Dfetch(t0up1, threadIdx.x);
		*((uint32_t*)mixtabs + ( 256+threadIdx.x)) = tex1Dfetch(t0dn1, threadIdx.x);
		*((uint32_t*)mixtabs + ( 512+threadIdx.x)) = tex1Dfetch(t1up1, threadIdx.x);
		*((uint32_t*)mixtabs + ( 768+threadIdx.x)) = tex1Dfetch(t1dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (1024+threadIdx.x)) = tex1Dfetch(t2up1, threadIdx.x);
		*((uint32_t*)mixtabs + (1280+threadIdx.x)) = tex1Dfetch(t2dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (1536+threadIdx.x)) = tex1Dfetch(t3up1, threadIdx.x);
		*((uint32_t*)mixtabs + (1792+threadIdx.x)) = tex1Dfetch(t3dn1, threadIdx.x);
	}
	__syncthreads();
#endif

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t message[32];

		#pragma unroll 5
		for (int i=0; i < 20; i += 4)
			AS_UINT4(&message[i]) = AS_UINT4(&c_Message80[i]);

		message[19] = cuda_swab32(startNounce + thread);
		message[20] = 0x80U; // end tag

		#pragma unroll
		for(int i=21; i<31; i++) message[i] = 0U;
		message[31] = 0x01000000U; // end block

		uint32_t state[32];
		#pragma unroll
		for(int i=0; i<32; i++) state[i] = message[i];
		state[31] ^= 0x00020000U; // "...00000201"

#if USE_SHARED
		quark_groestl512_perm_P(state, mixtabs);
		quark_groestl512_perm_Q(message, mixtabs);

		state[31] ^= 0x00020000U;
		#pragma unroll 32
		for(int i=0; i<32; i++) state[i] ^= message[i];

		#pragma unroll 16
		for(int i=16; i<32; i++) message[i] = state[i];

		quark_groestl512_perm_P(state, mixtabs);
#else
		tex_groestl512_perm_P(state);
		tex_groestl512_perm_Q(message);

		state[31] ^= 0x00020000U;
		#pragma unroll 32
		for(int i=0; i<32; i++) state[i] ^= message[i];

		#pragma unroll 16
		for(int i=16; i<32; i++) message[i] = state[i];

		tex_groestl512_perm_P(state);
#endif
		#pragma unroll 16
		for(int i=16; i<32; i++) state[i] ^= message[i];

		// uint4 = 4 x uint32_t = 16 bytes, x 4 => 64 bytes
		const off_t hashPosition = thread;

		uint4 *outpt = (uint4*) (&g_outhash[hashPosition << 4]);
		uint4 *phash = (uint4*) (&state[16]);
		outpt[0] = phash[0];
		outpt[1] = phash[1];
		outpt[2] = phash[2];
		outpt[3] = phash[3];
	}
#endif
}

#endif // WANT_GROESTL80
