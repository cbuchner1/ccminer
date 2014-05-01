// Auf QuarkCoin spezialisierte Version von Groestl

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
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// diese Struktur wird in der Init Funktion angefordert
static cudaDeviceProp props[8];

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

__device__ __forceinline__ void quark_groestl512_perm_P(uint32_t *a, char *mixtabs)
{
    uint32_t t[32];

//#pragma unroll 14
    for(int r=0;r<14;r++)
    {
        switch(r)
        {
            case 0:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 0); break;
            case 1:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 1); break;
            case 2:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 2); break;
            case 3:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 3); break;
            case 4:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 4); break;
            case 5:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 5); break;
            case 6:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 6); break;
            case 7:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 7); break;
            case 8:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 8); break;
            case 9:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 9); break;
            case 10:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 10); break;
            case 11:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 11); break;
            case 12:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 12); break;
            case 13:
#pragma unroll 16
                for(int k=0;k<16;k++) a[(k*2)+0] ^= PC32up(k<< 4, 13); break;
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

__device__ __forceinline__ void quark_groestl512_perm_Q(uint32_t *a, char *mixtabs)
{    
//#pragma unroll 14
    for(int r=0;r<14;r++)
    {
        uint32_t t[32];

        switch(r)
        {
            case 0:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 0); a[(k*2)+1] ^= QC32dn(k<< 4, 0);} break;
            case 1:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 1); a[(k*2)+1] ^= QC32dn(k<< 4, 1);} break;
            case 2:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 2); a[(k*2)+1] ^= QC32dn(k<< 4, 2);} break;
            case 3:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 3); a[(k*2)+1] ^= QC32dn(k<< 4, 3);} break;
            case 4:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 4); a[(k*2)+1] ^= QC32dn(k<< 4, 4);} break;
            case 5:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 5); a[(k*2)+1] ^= QC32dn(k<< 4, 5);} break;
            case 6:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 6); a[(k*2)+1] ^= QC32dn(k<< 4, 6);} break;
            case 7:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 7); a[(k*2)+1] ^= QC32dn(k<< 4, 7);} break;
            case 8:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 8); a[(k*2)+1] ^= QC32dn(k<< 4, 8);} break;
            case 9:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 9); a[(k*2)+1] ^= QC32dn(k<< 4, 9);} break;
            case 10:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 10); a[(k*2)+1] ^= QC32dn(k<< 4, 10);} break;
            case 11:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 11); a[(k*2)+1] ^= QC32dn(k<< 4, 11);} break;
            case 12:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 12); a[(k*2)+1] ^= QC32dn(k<< 4, 12);} break;
            case 13:
    #pragma unroll 16
                for(int k=0;k<16;k++) { a[(k*2)+0] ^= QC32up(k<< 4, 13); a[(k*2)+1] ^= QC32dn(k<< 4, 13);} break;
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
__global__ void  quark_groestl512_gpu_hash_64(int threads, uint32_t startNounce, uint32_t *g_hash, uint32_t *g_nonceVector)
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

        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *inpHash = &g_hash[16 * hashPosition];

#pragma unroll 16
        for(int k=0;k<16;k++) message[k] = inpHash[k];
#pragma unroll 14
        for(int k=1;k<15;k++)
            message[k+16] = 0;

        message[16] = 0x80;
        message[31] = 0x01000000;

#pragma unroll 32
        for(int u=0;u<32;u++) state[u] = message[u];
        state[31] ^= 0x20000;

        // Perm
#if USE_SHARED
        quark_groestl512_perm_P(state, mixtabs);
        state[31] ^= 0x20000;
        quark_groestl512_perm_Q(message, mixtabs);
#else
        quark_groestl512_perm_P(state, NULL);
        state[31] ^= 0x20000;
        quark_groestl512_perm_Q(message, NULL);
#endif
#pragma unroll 32
        for(int u=0;u<32;u++) state[u] ^= message[u];

#pragma unroll 32
        for(int u=0;u<32;u++) message[u] = state[u];

#if USE_SHARED
        quark_groestl512_perm_P(message, mixtabs);
#else
        quark_groestl512_perm_P(message, NULL);
#endif

#pragma unroll 32
        for(int u=0;u<32;u++) state[u] ^= message[u];
        // Erzeugten Hash rausschreiben

        uint32_t *outpHash = &g_hash[16 * hashPosition];

#pragma unroll 16
        for(int k=0;k<16;k++) outpHash[k] = state[k+16];
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
__host__ void quark_groestl512_cpu_init(int thr_id, int threads)
{
    cudaGetDeviceProperties(&props[thr_id], device_map[thr_id]);

// Texturen mit obigem Makro initialisieren
    texDef(t0up1, d_T0up, T0up_cpu, sizeof(uint32_t)*256);
    texDef(t0dn1, d_T0dn, T0dn_cpu, sizeof(uint32_t)*256);
    texDef(t1up1, d_T1up, T1up_cpu, sizeof(uint32_t)*256);
    texDef(t1dn1, d_T1dn, T1dn_cpu, sizeof(uint32_t)*256);
    texDef(t2up1, d_T2up, T2up_cpu, sizeof(uint32_t)*256);
    texDef(t2dn1, d_T2dn, T2dn_cpu, sizeof(uint32_t)*256);
    texDef(t3up1, d_T3up, T3up_cpu, sizeof(uint32_t)*256);
    texDef(t3dn1, d_T3dn, T3dn_cpu, sizeof(uint32_t)*256);
}

__host__ void quark_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    // Compute 3.5 und 5.x Geräte am besten mit 768 Threads ansteuern,
    // alle anderen mit 512 Threads.
    int threadsperblock = ((props[thr_id].major == 3 && props[thr_id].minor == 5) || props[thr_id].major > 3) ? 768 : 512;

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
    quark_groestl512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_nonceVector);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void quark_doublegroestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    // Compute 3.5 und 5.x Geräte am besten mit 768 Threads ansteuern,
    // alle anderen mit 512 Threads.
    int threadsperblock = ((props[thr_id].major == 3 && props[thr_id].minor == 5) || props[thr_id].major > 3) ? 768 : 512;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
#if USE_SHARED
    size_t shared_size = 8 * 256 * sizeof(uint32_t);
#else
    size_t shared_size = 0;
#endif

//  fprintf(stderr, "threads=%d, %d blocks, %d threads per block, %d bytes shared\n", threads, grid.x, block.x, shared_size);
    //fprintf(stderr, "ThrID: %d\n", thr_id);
    quark_groestl512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_nonceVector);
    quark_groestl512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_nonceVector);

    // Strategisches Sleep Kommando zur Senkung der CPU Last
    MyStreamSynchronize(NULL, order, thr_id);
}
