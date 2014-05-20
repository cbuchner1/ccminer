// Auf Myriadcoin spezialisierte Version von Groestl

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

// it's unfortunate that this is a compile time constant.
#define MAXWELL_OR_FERMI 1

// aus cpu-miner.c
extern int device_map[8];

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// Folgende Definitionen später durch header ersetzen
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

// diese Struktur wird in der Init Funktion angefordert
static cudaDeviceProp props;

__constant__ uint32_t pTarget[8]; // Single GPU
extern uint32_t *d_resultNonce[8];

__constant__ uint32_t myriadgroestl_gpu_msg[32];

// muss expandiert werden
__constant__ uint32_t myr_sha256_gpu_constantTable[64];
__constant__ uint32_t myr_sha256_gpu_hashTable[8];

uint32_t myr_sha256_cpu_hashTable[] = { 
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
uint32_t myr_sha256_cpu_constantTable[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
    // Kepler (Compute 3.5)
    #define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif
#define R(x, n)			((x) >> (n))
#define Ch(x, y, z)		((x & (y ^ z)) ^ z)
#define Maj(x, y, z)	((x & (y | z)) | (y & z))
#define S0(x)			(ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)			(ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)			(ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)			(ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))

#define SWAB32(x)		( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

__device__ void myriadgroestl_gpu_sha256(uint32_t *message)
{
	uint32_t W1[16];
	uint32_t W2[16];

	// Initialisiere die register a bis h mit der Hash-Tabelle
	uint32_t regs[8];
	uint32_t hash[8];

	// pre
#pragma unroll 8
	for (int k=0; k < 8; k++)
	{
		regs[k] = myr_sha256_gpu_hashTable[k];
		hash[k] = regs[k];
	}
	
#pragma unroll 16
	for(int k=0;k<16;k++)
		W1[k] = SWAB32(message[k]);

// Progress W1
#pragma unroll 16
	for(int j=0;j<16;j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j] + W1[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		
		#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

// Progress W2...W3
#pragma unroll 3
	for(int k=0;k<3;k++)
	{
#pragma unroll 2
		for(int j=0;j<2;j++)
			W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];
#pragma unroll 5
		for(int j=2;j<7;j++)
			W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

#pragma unroll 8
		for(int j=7;j<15;j++)
			W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Rundenfunktion
#pragma unroll 16
		for(int j=0;j<16;j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j + 16 * (k+1)] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		
			#pragma unroll 7
			for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

#pragma unroll 16
		for(int j=0;j<16;j++)
			W1[j] = W2[j];
	}

#pragma unroll 8
	for(int k=0;k<8;k++)
		hash[k] += regs[k];

	/////
	///// Zweite Runde (wegen Msg-Padding)
	/////
#pragma unroll 8
	for(int k=0;k<8;k++)
		regs[k] = hash[k];

	W1[0] = SWAB32(0x80);
#pragma unroll 14
	for(int k=1;k<15;k++)
		W1[k] = 0;
	W1[15] = 512;

// Progress W1
#pragma unroll 16
	for(int j=0;j<16;j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j] + W1[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		
		#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

// Progress W2...W3
#pragma unroll 3
	for(int k=0;k<3;k++)
	{
#pragma unroll 2
		for(int j=0;j<2;j++)
			W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];
#pragma unroll 5
		for(int j=2;j<7;j++)
			W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

#pragma unroll 8
		for(int j=7;j<15;j++)
			W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

		W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

		// Rundenfunktion
#pragma unroll 16
		for(int j=0;j<16;j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + myr_sha256_gpu_constantTable[j + 16 * (k+1)] + W2[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		
			#pragma unroll 7
			for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

#pragma unroll 16
		for(int j=0;j<16;j++)
			W1[j] = W2[j];
	}

#pragma unroll 8
	for(int k=0;k<8;k++)
		hash[k] += regs[k];

	//// FERTIG

#pragma unroll 8
	for(int k=0;k<8;k++)
		message[k] = SWAB32(hash[k]);
}

#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

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
#define T0up(x) tex1Dfetch(t0up1, x)
#define T0dn(x) tex1Dfetch(t0dn1, x)
#define T1up(x) tex1Dfetch(t1up1, x)
#define T1dn(x) tex1Dfetch(t1dn1, x)
#define T2up(x) tex1Dfetch(t2up1, x)
#define T2dn(x) tex1Dfetch(t2dn1, x)
#define T3up(x) tex1Dfetch(t3up1, x)
#define T3dn(x) tex1Dfetch(t3dn1, x)
#endif
#else
#define USE_SHARED 1
// a healthy mix between shared and textured access provides the highest speed on Compute 3.0 and 3.5!
#define T0up(x) (*((uint32_t*)mixtabs + (    (x))))
#define T0dn(x) tex1Dfetch(t0dn1, x)
#define T1up(x) tex1Dfetch(t1up1, x)
#define T1dn(x) (*((uint32_t*)mixtabs + (768+(x))))
#define T2up(x) tex1Dfetch(t2up1, x)
#define T2dn(x) (*((uint32_t*)mixtabs + (1280+(x))))
#define T3up(x) (*((uint32_t*)mixtabs + (1536+(x))))
#define T3dn(x) tex1Dfetch(t3dn1, x)
#endif

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

#define SWAB32(x)		( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )


__device__ __forceinline__ void myriadgroestl_perm_P(uint32_t *a, char *mixtabs)
{
	uint32_t t[32];

//#pragma unroll 14
	for(int r=0;r<14;r++)
	{
		switch(r)
		{
			case 0:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 0); break;
			case 1:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 1); break;
			case 2:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 2); break;
			case 3:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 3); break;
			case 4:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 4); break;
			case 5:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 5); break;
			case 6:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 6); break;
			case 7:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 7); break;
			case 8:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 8); break;
			case 9:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 9); break;
			case 10:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 10); break;
			case 11:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 11); break;
			case 12:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 12); break;
			case 13:
#pragma unroll 16
				for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k * 0x10, 13); break;
		}

        // RBTT
#pragma unroll 16
        for(int k=0;k<32;k+=2)
        {
            uint32_t t0_0 = B32_0(a[(k     ) & 0x1f]), t9_0  = B32_0(a[(k +  9) & 0x1f]);
            uint32_t t2_1 = B32_1(a[(k +  2) & 0x1f]), t11_1 = B32_1(a[(k + 11) & 0x1f]);
            uint32_t t4_2 = B32_2(a[(k +  4) & 0x1f]), t13_2 = B32_2(a[(k + 13) & 0x1f]);
            uint32_t t6_3 = B32_3(a[(k +  6) & 0x1f]), t23_3 = B32_3(a[(k + 23) & 0x1f]);
        
            t[k + 0] =  T0up( t0_0 ) ^ T1up(  t2_1 ) ^ T2up(  t4_2 ) ^ T3up(  t6_3 ) ^ 
                        T0dn( t9_0 ) ^ T1dn( t11_1 ) ^ T2dn( t13_2 ) ^ T3dn( t23_3 );

            t[k + 1] =  T0dn( t0_0 ) ^ T1dn(  t2_1 ) ^ T2dn(  t4_2 ) ^ T3dn(  t6_3 ) ^ 
                        T0up( t9_0 ) ^ T1up( t11_1 ) ^ T2up( t13_2 ) ^ T3up( t23_3 );
        }
#pragma unroll 32
        for(int k=0;k<32;k++)
            a[k] = t[k];
    }
}

__device__ __forceinline__ void myriadgroestl_perm_Q(uint32_t *a, char *mixtabs)
{	
//#pragma unroll 14
	for(int r=0;r<14;r++)
	{
		uint32_t t[32];

		switch(r)
		{
			case 0:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 0); a[(k*2)+1] ^= QC32dn(k * 0x10, 0);} break;
			case 1:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 1); a[(k*2)+1] ^= QC32dn(k * 0x10, 1);} break;
			case 2:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 2); a[(k*2)+1] ^= QC32dn(k * 0x10, 2);} break;
			case 3:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 3); a[(k*2)+1] ^= QC32dn(k * 0x10, 3);} break;
			case 4:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 4); a[(k*2)+1] ^= QC32dn(k * 0x10, 4);} break;
			case 5:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 5); a[(k*2)+1] ^= QC32dn(k * 0x10, 5);} break;
			case 6:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 6); a[(k*2)+1] ^= QC32dn(k * 0x10, 6);} break;
			case 7:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 7); a[(k*2)+1] ^= QC32dn(k * 0x10, 7);} break;
			case 8:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 8); a[(k*2)+1] ^= QC32dn(k * 0x10, 8);} break;
			case 9:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 9); a[(k*2)+1] ^= QC32dn(k * 0x10, 9);} break;
			case 10:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 10); a[(k*2)+1] ^= QC32dn(k * 0x10, 10);} break;
			case 11:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 11); a[(k*2)+1] ^= QC32dn(k * 0x10, 11);} break;
			case 12:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 12); a[(k*2)+1] ^= QC32dn(k * 0x10, 12);} break;
			case 13:
	#pragma unroll 16
				for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k * 0x10, 13); a[(k*2)+1] ^= QC32dn(k * 0x10, 13);} break;
		}

        // RBTT
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
        for(int k=0;k<32;k++)
            a[k] = t[k];
    }
}

__global__ void 
myriadgroestl_gpu_hash(int threads, uint32_t startNounce, uint32_t *resNounce)
{
#if USE_SHARED
	extern __shared__ char mixtabs[];

	if (threadIdx.x < 256)
	{
		*((uint32_t*)mixtabs + (    threadIdx.x)) = tex1Dfetch(t0up1, threadIdx.x);
		*((uint32_t*)mixtabs + (256+threadIdx.x)) = tex1Dfetch(t0dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (512+threadIdx.x)) = tex1Dfetch(t1up1, threadIdx.x);
		*((uint32_t*)mixtabs + (768+threadIdx.x)) = tex1Dfetch(t1dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (1024+threadIdx.x)) = tex1Dfetch(t2up1, threadIdx.x);
		*((uint32_t*)mixtabs + (1280+threadIdx.x)) = tex1Dfetch(t2dn1, threadIdx.x);
		*((uint32_t*)mixtabs + (1536+threadIdx.x)) = tex1Dfetch(t3up1, threadIdx.x);
		*((uint32_t*)mixtabs + (1792+threadIdx.x)) = tex1Dfetch(t3dn1, threadIdx.x);
	}

	__syncthreads();
#endif

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
		// GROESTL
		uint32_t message[32];
		uint32_t state[32];

#pragma unroll 32
		for(int k=0;k<32;k++) message[k] = myriadgroestl_gpu_msg[k];

		uint32_t nounce = startNounce + thread;
		message[19] = SWAB32(nounce);

#pragma unroll 32
		for(int u=0;u<32;u++) state[u] = message[u];
		state[31] ^= 0x20000;

		// Perm
#if USE_SHARED
		myriadgroestl_perm_P(state, mixtabs);
		state[31] ^= 0x20000;
		myriadgroestl_perm_Q(message, mixtabs);
#else
		myriadgroestl_perm_P(state, NULL);
		state[31] ^= 0x20000;
		myriadgroestl_perm_Q(message, NULL);
#endif
#pragma unroll 32
		for(int u=0;u<32;u++) state[u] ^= message[u];

#pragma unroll 32
		for(int u=0;u<32;u++) message[u] = state[u];

#if USE_SHARED
		myriadgroestl_perm_P(message, mixtabs);
#else
		myriadgroestl_perm_P(message, NULL);
#endif

#pragma unroll 32
		for(int u=0;u<32;u++) state[u] ^= message[u];

        uint32_t out_state[16];
#pragma unroll 16
		for(int u=0;u<16;u++) out_state[u] = state[u+16];
        myriadgroestl_gpu_sha256(out_state);
        
        int i, position = -1;
        bool rc = true;

#pragma unroll 8
        for (i = 7; i >= 0; i--) {
            if (out_state[i] > pTarget[i]) {
                if(position < i) {
                    position = i;
                    rc = false;
                }
             }
             if (out_state[i] < pTarget[i]) {
                if(position < i) {
                    position = i;
                    rc = true;
                }
             }
        }

        if(rc == true)
            if(resNounce[0] > nounce)
                resNounce[0] = nounce;
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

// Setup-Funktionen
__host__ void myriadgroestl_cpu_init(int thr_id, int threads)
{
	cudaSetDevice(device_map[thr_id]);
	
	cudaMemcpyToSymbol(	myr_sha256_gpu_hashTable,
						myr_sha256_cpu_hashTable,
						sizeof(uint32_t) * 8 );

	cudaMemcpyToSymbol(	myr_sha256_gpu_constantTable,
						myr_sha256_cpu_constantTable,
						sizeof(uint32_t) * 64 );

    cudaGetDeviceProperties(&props, device_map[thr_id]);

	// Texturen mit obigem Makro initialisieren
	texDef(t0up1, d_T0up, T0up_cpu, sizeof(uint32_t)*256);
	texDef(t0dn1, d_T0dn, T0dn_cpu, sizeof(uint32_t)*256);
	texDef(t1up1, d_T1up, T1up_cpu, sizeof(uint32_t)*256);
	texDef(t1dn1, d_T1dn, T1dn_cpu, sizeof(uint32_t)*256);
	texDef(t2up1, d_T2up, T2up_cpu, sizeof(uint32_t)*256);
	texDef(t2dn1, d_T2dn, T2dn_cpu, sizeof(uint32_t)*256);
	texDef(t3up1, d_T3up, T3up_cpu, sizeof(uint32_t)*256);
	texDef(t3dn1, d_T3dn, T3dn_cpu, sizeof(uint32_t)*256);

    // Speicher für Gewinner-Nonce belegen
    cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)); 
}

__host__ void myriadgroestl_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
    // Nachricht expandieren und setzen
    uint32_t msgBlock[32];

    memset(msgBlock, 0, sizeof(uint32_t) * 32);
    memcpy(&msgBlock[0], data, 80);

    // Erweitere die Nachricht auf den Nachrichtenblock (padding)
    // Unsere Nachricht hat 80 Byte
    msgBlock[20] = 0x80;
    msgBlock[31] = 0x01000000;

    // groestl512 braucht hierfür keinen CPU-Code (die einzige Runde wird
    // auf der GPU ausgeführt)

    // Blockheader setzen (korrekte Nonce und Hefty Hash fehlen da drin noch)
    cudaMemcpyToSymbol( myriadgroestl_gpu_msg,
                        msgBlock,
                        128);

    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    cudaMemcpyToSymbol( pTarget,
                        pTargetIn,
                        sizeof(uint32_t) * 8 );
}

__host__ void myriadgroestl_cpu_hash(int thr_id, int threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce)
{
	// Compute 3.x und 5.x Geräte am besten mit 768 Threads ansteuern,
	// alle anderen mit 512 Threads.
	int threadsperblock = (props.major >= 3) ? 768 : 512;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
#if USE_SHARED
	size_t shared_size = 8 * 256 * sizeof(uint32_t);
#else
	size_t shared_size = 0;
#endif

//    fprintf(stderr, "threads=%d, %d blocks, %d threads per block, %d bytes shared\n", threads, grid.x, block.x, shared_size);
    //fprintf(stderr, "ThrID: %d\n", thr_id);
    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    myriadgroestl_gpu_hash<<<grid, block, shared_size>>>(threads, startNounce, d_resultNonce[thr_id]);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, 0, thr_id);

    cudaMemcpy(nounce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
