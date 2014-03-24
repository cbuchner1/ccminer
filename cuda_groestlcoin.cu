// Auf Groestlcoin spezialisierte Version von Groestl

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

#define USE_SHARED 1

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// Folgende Definitionen später durch header ersetzen
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// globaler Speicher für alle HeftyHashes aller Threads
__constant__ uint32_t pTarget[8]; // Single GPU
extern uint32_t *d_resultNonce[8];

// globaler Speicher für unsere Ergebnisse
uint32_t *d_hashGROESTLCOINoutput[8];

__constant__ uint32_t groestlcoin_gpu_state[32];
__constant__ uint32_t groestlcoin_gpu_msg[32];
__constant__ uint32_t sha256coin_gpu_constantTable[64];
__constant__ uint32_t sha256coin_gpu_register[8];

#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

#define PC32up(j, r)   ((uint32_t)((j) + (r)))
#define PC32dn(j, r)   0
#define QC32up(j, r)   0xFFFFFFFF
#define QC32dn(j, r)   (((uint32_t)(r) << 24) ^ SPH_T32(~((uint32_t)(j) << 24)))

#define B32_0(x)    ((x) & 0xFF)
#define B32_1(x)    (((x) >> 8) & 0xFF)
#define B32_2(x)    (((x) >> 16) & 0xFF)
#define B32_3(x)    ((x) >> 24)

#define SPH_C32(x)	((uint32_t)(x ## U))
#define C32e(x)     ((SPH_C32(x) >> 24) \
                    | ((SPH_C32(x) >>  8) & SPH_C32(0x0000FF00)) \
                    | ((SPH_C32(x) <<  8) & SPH_C32(0x00FF0000)) \
                    | ((SPH_C32(x) << 24) & SPH_C32(0xFF000000)))

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

#define S(x, n)			(((x) >> (n)) | ((x) << (32 - (n))))
#define R(x, n)			((x) >> (n))
#define Ch(x, y, z)		((x & (y ^ z)) ^ z)
#define Maj(x, y, z)	((x & (y | z)) | (y & z))
#define S0(x)			(S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define S1(x)			(S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define s0(x)			(S(x, 7) ^ S(x, 18) ^ R(x, 3))
#define s1(x)			(S(x, 17) ^ S(x, 19) ^ R(x, 10))

#define SWAB32(x)		( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )


__device__ void groestlcoin_perm_P(uint32_t *a, char *mixtabs)
{
	uint32_t t[32];

//#pragma unroll 14
	for(int r=0;r<14;r++)
	{
#pragma unroll 16
		for(int k=0;k<16;k++)
		{
			a[(k*2)+0] ^= PC32up(k * 0x10, r);
			//a[(k<<1)+1] ^= PC32dn(k * 0x10, r);
		}

		// RBTT
#pragma unroll 16
		for(int k=0;k<32;k+=2)
		{
			t[k + 0] =	T0up( B32_0(a[k & 0x1f]) ) ^ 
						T1up( B32_1(a[(k + 2) & 0x1f]) ) ^ 
						T2up( B32_2(a[(k + 4) & 0x1f]) ) ^ 
						T3up( B32_3(a[(k + 6) & 0x1f]) ) ^ 
						T0dn( B32_0(a[(k + 9) & 0x1f]) ) ^ 
						T1dn( B32_1(a[(k + 11) & 0x1f]) ) ^ 
						T2dn( B32_2(a[(k + 13) & 0x1f]) ) ^ 
						T3dn( B32_3(a[(k + 23) & 0x1f]) );

			t[k + 1] =	T0dn( B32_0(a[k & 0x1f]) ) ^ 
						T1dn( B32_1(a[(k + 2) & 0x1f]) ) ^ 
						T2dn( B32_2(a[(k + 4) & 0x1f]) ) ^ 
						T3dn( B32_3(a[(k + 6) & 0x1f]) ) ^ 
						T0up( B32_0(a[(k + 9) & 0x1f]) ) ^ 
						T1up( B32_1(a[(k + 11) & 0x1f]) ) ^ 
						T2up( B32_2(a[(k + 13) & 0x1f]) ) ^ 
						T3up( B32_3(a[(k + 23) & 0x1f]) );
		}
#pragma unroll 32
		for(int k=0;k<32;k++)
			a[k] = t[k];
	}
}

__device__ void groestlcoin_perm_Q(uint32_t *a, char *mixtabs)
{	
//#pragma unroll 14
	for(int r=0;r<14;r++)
	{
		uint32_t t[32];

#pragma unroll 16
		for(int k=0;k<16;k++)
		{
			a[(k*2)+0] ^= QC32up(k * 0x10, r);
			a[(k*2)+1] ^= QC32dn(k * 0x10, r);
		}

		// RBTT
#pragma unroll 16
		for(int k=0;k<32;k+=2)
		{
			t[k + 0] =	T0up( B32_0(a[(k + 2) & 0x1f]) ) ^ 
						T1up( B32_1(a[(k + 6) & 0x1f]) ) ^ 
						T2up( B32_2(a[(k + 10) & 0x1f]) ) ^ 
						T3up( B32_3(a[(k + 22) & 0x1f]) ) ^ 
						T0dn( B32_0(a[(k + 1) & 0x1f]) ) ^ 
						T1dn( B32_1(a[(k + 5) & 0x1f]) ) ^ 
						T2dn( B32_2(a[(k + 9) & 0x1f]) ) ^ 
						T3dn( B32_3(a[(k + 13) & 0x1f]) );

			t[k + 1] =	T0dn( B32_0(a[(k + 2) & 0x1f]) ) ^ 
						T1dn( B32_1(a[(k + 6) & 0x1f]) ) ^ 
						T2dn( B32_2(a[(k + 10) & 0x1f]) ) ^ 
						T3dn( B32_3(a[(k + 22) & 0x1f]) ) ^ 
						T0up( B32_0(a[(k + 1) & 0x1f]) ) ^ 
						T1up( B32_1(a[(k + 5) & 0x1f]) ) ^ 
						T2up( B32_2(a[(k + 9) & 0x1f]) ) ^ 
						T3up( B32_3(a[(k + 13) & 0x1f]) );
		}
#pragma unroll 32
		for(int k=0;k<32;k++)
			a[k] = t[k];
	}
}
#if USE_SHARED
__global__ void  __launch_bounds__(256) 
#else
__global__ void 
#endif

 groestlcoin_gpu_hash(int threads, uint32_t startNounce, void *outputHash, uint32_t *resNounce)
{
#if USE_SHARED
	extern __shared__ char mixtabs[];

	*((uint32_t*)mixtabs + (    threadIdx.x)) = tex1Dfetch(t0up1, threadIdx.x);
	*((uint32_t*)mixtabs + (256+threadIdx.x)) = tex1Dfetch(t0dn1, threadIdx.x);
	*((uint32_t*)mixtabs + (512+threadIdx.x)) = tex1Dfetch(t1up1, threadIdx.x);
	*((uint32_t*)mixtabs + (768+threadIdx.x)) = tex1Dfetch(t1dn1, threadIdx.x);
	*((uint32_t*)mixtabs + (1024+threadIdx.x)) = tex1Dfetch(t2up1, threadIdx.x);
	*((uint32_t*)mixtabs + (1280+threadIdx.x)) = tex1Dfetch(t2dn1, threadIdx.x);
	*((uint32_t*)mixtabs + (1536+threadIdx.x)) = tex1Dfetch(t3up1, threadIdx.x);
	*((uint32_t*)mixtabs + (1792+threadIdx.x)) = tex1Dfetch(t3dn1, threadIdx.x);

	__syncthreads();
#endif

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
	/////
	///// Lieber groestl, mach, dass es abgeht!!!
	/////
		// GROESTL
		uint32_t message[32];
		uint32_t state[32];
		uint32_t g[32];


#pragma unroll 32
		for(int k=0;k<32;k++)
		{
                        // TODO: die Vorbelegung mit Nullen braucht nicht zwingend aus dem
                        //       constant Memory zu lesen. Das ist Verschwendung von Bandbreite.
			state[k] = groestlcoin_gpu_state[k];
			message[k] = groestlcoin_gpu_msg[k];
		}

		uint32_t nounce = startNounce + thread;
		message[19] = SWAB32(nounce);

#pragma unroll 32
		for(int u=0;u<32;u++)
			g[u] = message[u] ^ state[u];  // TODO: state ist fast ueberall 0.

		// Perm
#if USE_SHARED
		groestlcoin_perm_P(g, mixtabs);        // TODO: g[] entspricht fast genau message[]
		groestlcoin_perm_Q(message, mixtabs);  //       kann man das ausnutzen?
#else
		groestlcoin_perm_P(g, NULL);
		groestlcoin_perm_Q(message, NULL);
#endif
		
#pragma unroll 32
		for(int u=0;u<32;u++)
		{
                        // TODO: kann man evtl. das xor mit g[u] vorziehen hinter die groestlcoin_perm_P Funktion
                        //       was den Registerbedarf senken koennte?
			state[u] ^= g[u] ^ message[u];
			g[u] = state[u];
		}

#if USE_SHARED
		groestlcoin_perm_P(g, mixtabs);
#else
		groestlcoin_perm_P(g, NULL);
#endif

#pragma unroll 32
		for(int u=0;u<32;u++)
			state[u] ^= g[u];

		////
		//// 2. Runde groestl
		////
#pragma unroll 16
		for(int k=0;k<16;k++)
			message[k] = state[k + 16];

#pragma unroll 32
		for(int k=0;k<32;k++)
			state[k] = groestlcoin_gpu_state[k];

#pragma unroll 16
		for(int k=0;k<16;k++)
			message[k+16] = 0;

		message[16] = 0x80;		
		message[31] = 0x01000000;

#pragma unroll 32
		for(int u=0;u<32;u++)
			g[u] = message[u] ^ state[u];

		// Perm
#if USE_SHARED
		groestlcoin_perm_P(g, mixtabs);
		groestlcoin_perm_Q(message, mixtabs);
#else
		groestlcoin_perm_P(g, NULL);
		groestlcoin_perm_Q(message, NULL);
#endif
		
#pragma unroll 32
		for(int u=0;u<32;u++)
		{
			state[u] ^= g[u] ^ message[u];
			g[u] = state[u];
		}

#if USE_SHARED
		groestlcoin_perm_P(g, mixtabs);
#else
		groestlcoin_perm_P(g, NULL);
#endif

#pragma unroll 32
		for(int u=0;u<32;u++)
			state[u] ^= g[u];
		
/*
	#pragma unroll 8
		for(int k=0;k<8;k++)
			hash[k] = state[k+16];
*/

		// kopiere Ergebnis
		/*
#pragma unroll 16
		for(int k=0;k<16;k++)
			((uint32_t*)outputHash)[16*thread+k] = state[k + 16];
			*/
		int i;
		bool rc = true;
	
		for (i = 7; i >= 0; i--) {
			if (state[i+16] > pTarget[i]) {
				rc = false;
				break;
			}
			if (state[i+16] < pTarget[i]) {
				rc = true;
				break;
			}
		}

		if(rc == true)
		{
			if(resNounce[0] > nounce)
			{
				resNounce[0] = nounce;
				/*
				#pragma unroll 8
				for(int k=0;k<8;k++)					
					((uint32_t*)outputHash)[k] = (hash[k]);
				*/
			}
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

// Setup-Funktionen
__host__ void groestlcoin_cpu_init(int thr_id, int threads)
{
	cudaSetDevice(thr_id);
	cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );
// Texturen mit obigem Makro initialisieren
	texDef(t0up1, d_T0up, T0up_cpu, sizeof(uint32_t)*256);
	texDef(t0dn1, d_T0dn, T0dn_cpu, sizeof(uint32_t)*256);
	texDef(t1up1, d_T1up, T1up_cpu, sizeof(uint32_t)*256);
	texDef(t1dn1, d_T1dn, T1dn_cpu, sizeof(uint32_t)*256);
	texDef(t2up1, d_T2up, T2up_cpu, sizeof(uint32_t)*256);
	texDef(t2dn1, d_T2dn, T2dn_cpu, sizeof(uint32_t)*256);
	texDef(t3up1, d_T3up, T3up_cpu, sizeof(uint32_t)*256);
	texDef(t3dn1, d_T3dn, T3dn_cpu, sizeof(uint32_t)*256);

	// setze register 
        // TODO: fast vollstaendige Vorbelegung mit Nullen.
        //       da besteht doch Optimierungspotenzial im GPU Kernel
        //       denn mit Nullen braucht man nicht wirklich rechnen.
	uint32_t groestl_state_init[32];
	memset(groestl_state_init, 0, sizeof(uint32_t) * 32);
	groestl_state_init[31] = 0x20000;

	// state speichern
	cudaMemcpyToSymbol(	groestlcoin_gpu_state,
						groestl_state_init,
						128);

	cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)); 

	// Speicher für alle Ergebnisse belegen (nur für Debug)
	cudaMalloc(&d_hashGROESTLCOINoutput[thr_id], 8 * sizeof(uint32_t) * threads);
}

__host__ void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
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
	cudaMemcpyToSymbol(	groestlcoin_gpu_msg,
						msgBlock,
						128);

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	cudaMemcpyToSymbol(	pTarget,
						pTargetIn,
						sizeof(uint32_t) * 8 );
}

__host__ void groestlcoin_cpu_hash(int thr_id, int threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce)
{
#if USE_SHARED
	const int threadsperblock = 256; // Alignment mit mixtab Grösse. NICHT ÄNDERN
#else
	const int threadsperblock = 512; // so einstellen wie gewünscht ;-)
#endif

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs (abhängig von der Threadanzahl)
#if USE_SHARED
	size_t shared_size = 8 * 256 * sizeof(uint32_t);
#else
	size_t shared_size = 0;
#endif

//	fprintf(stderr, "threads=%d, %d blocks, %d threads per block, %d bytes shared\n", threads, grid.x, block.x, shared_size);
	//fprintf(stderr, "ThrID: %d\n", thr_id);
	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	groestlcoin_gpu_hash<<<grid, block, shared_size>>>(threads, startNounce, d_hashGROESTLCOINoutput[thr_id], d_resultNonce[thr_id]);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, 0, thr_id);

	cudaMemcpy(nounce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	/// Debug
	//cudaMemcpy(outputHashes, d_hashGROESTLCOINoutput[thr_id], 8 * sizeof(uint32_t) * threads, cudaMemcpyDeviceToHost);

	// Nounce
	//cudaMemcpy(nounce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
