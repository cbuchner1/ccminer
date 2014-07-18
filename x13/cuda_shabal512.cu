/*
 * Quick and dirty addition of Shabal-512 for X15
 * 
 * Built on cbuchner1's implementation, actual hashing code
 * heavily based on phm's sgminer
 *
 */

/*
 * Shabal-512 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  phm
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   phm <phm@inbox.com>
 */
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <memory.h>


#define USE_SHARED 1

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

#define sM    16

#define C32   SPH_C32
#define T32   SPH_T32

#define O1   13
#define O2    9
#define O3    6


#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define SPH_ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#else
    // Kepler (Compute 3.5)
    #define SPH_ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#endif


// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__constant__ uint32_t C_512[16]; 
static const uint32_t C_init_512[] = {
	C32(0xD9BF68D1), C32(0x58BAD750), C32(0x56028CB2), C32(0x8134F359),
	C32(0xB5D469D8), C32(0x941A8CC2), C32(0x418B2A6E), C32(0x04052780),
	C32(0x7F07D787), C32(0x5194358F), C32(0x3C60D665), C32(0xBE97D79A),
	C32(0x950C3434), C32(0xAED9A06D), C32(0x2537DC8D), C32(0x7CDB5969)
};
__constant__ uint32_t A_512[16];
static const uint32_t A_init_512[] = {
	C32(0x20728DFD), C32(0x46C0BD53), C32(0xE782B699), C32(0x55304632),
	C32(0x71B4EF90), C32(0x0EA9E82C), C32(0xDBB930F1), C32(0xFAD06B8B),
	C32(0xBE0CAE40), C32(0x8BD14410), C32(0x76D2ADAC), C32(0x28ACAB7F)
};
__constant__ uint32_t B_512[16];
static const uint32_t B_init_512[] = {
	C32(0xC1099CB7), C32(0x07B385F3), C32(0xE7442C26), C32(0xCC8AD640),
	C32(0xEB6F56C7), C32(0x1EA81AA9), C32(0x73B9D314), C32(0x1DE85D08),
	C32(0x48910A5A), C32(0x893B22DB), C32(0xC5A0DF44), C32(0xBBC4324E),
	C32(0x72D2F240), C32(0x75941D99), C32(0x6D8BDE82), C32(0xA1A7502B)
};

#define INPUT_BLOCK_ADD    { \
		B0 = T32(B0 + M0); \
		B1 = T32(B1 + M1); \
		B2 = T32(B2 + M2); \
		B3 = T32(B3 + M3); \
		B4 = T32(B4 + M4); \
		B5 = T32(B5 + M5); \
		B6 = T32(B6 + M6); \
		B7 = T32(B7 + M7); \
		B8 = T32(B8 + M8); \
		B9 = T32(B9 + M9); \
		BA = T32(BA + MA); \
		BB = T32(BB + MB); \
		BC = T32(BC + MC); \
		BD = T32(BD + MD); \
		BE = T32(BE + ME); \
		BF = T32(BF + MF); \
	} 

#define INPUT_BLOCK_SUB    { \
		C0 = T32(C0 - M0); \
		C1 = T32(C1 - M1); \
		C2 = T32(C2 - M2); \
		C3 = T32(C3 - M3); \
		C4 = T32(C4 - M4); \
		C5 = T32(C5 - M5); \
		C6 = T32(C6 - M6); \
		C7 = T32(C7 - M7); \
		C8 = T32(C8 - M8); \
		C9 = T32(C9 - M9); \
		CA = T32(CA - MA); \
		CB = T32(CB - MB); \
		CC = T32(CC - MC); \
		CD = T32(CD - MD); \
		CE = T32(CE - ME); \
		CF = T32(CF - MF); \
	} 

#define XOR_W    { \
		A00 ^= Wlow; \
		A01 ^= Whigh; \
	} 

#define SWAP(v1, v2)    { \
		uint32_t tmp = (v1); \
		(v1) = (v2); \
		(v2) = tmp; \
	} 

#define SWAP_BC    { \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \
	} 

#define PERM_ELT(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm)    { \
		xa0 = T32((xa0 \
			^ (((xa1 << 15) | (xa1 >> 17)) * 5U) \
			^ xc) * 3U) \
			^ xb1 ^ (xb2 & ~xb3) ^ xm; \
		xb0 = T32(~(((xb0 << 1) | (xb0 >> 31)) ^ xa0)); \
	} 

#define PERM_STEP_0    { \
		PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A04, A03, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A05, A04, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A06, A05, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A07, A06, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A08, A07, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A09, A08, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A0A, A09, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A00, A0B, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A01, A00, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A02, A01, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A03, A02, BF, BC, B8, B5, C9, MF); \
	} 

#define PERM_STEP_1   { \
		PERM_ELT(A04, A03, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A05, A04, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A06, A05, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A07, A06, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A08, A07, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A09, A08, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A0A, A09, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A00, A0B, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A01, A00, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A02, A01, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A03, A02, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A04, A03, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A05, A04, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A06, A05, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A07, A06, BF, BC, B8, B5, C9, MF); \
	} 

#define PERM_STEP_2   { \
		PERM_ELT(A08, A07, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A09, A08, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A0A, A09, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A00, A0B, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A01, A00, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A02, A01, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A03, A02, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A04, A03, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A05, A04, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A06, A05, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A07, A06, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A08, A07, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A09, A08, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A0A, A09, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A0B, A0A, BF, BC, B8, B5, C9, MF); \
	} 

#define APPLY_P    { \
		B0 = T32(B0 << 17) | (B0 >> 15); \
		B1 = T32(B1 << 17) | (B1 >> 15); \
		B2 = T32(B2 << 17) | (B2 >> 15); \
		B3 = T32(B3 << 17) | (B3 >> 15); \
		B4 = T32(B4 << 17) | (B4 >> 15); \
		B5 = T32(B5 << 17) | (B5 >> 15); \
		B6 = T32(B6 << 17) | (B6 >> 15); \
		B7 = T32(B7 << 17) | (B7 >> 15); \
		B8 = T32(B8 << 17) | (B8 >> 15); \
		B9 = T32(B9 << 17) | (B9 >> 15); \
		BA = T32(BA << 17) | (BA >> 15); \
		BB = T32(BB << 17) | (BB >> 15); \
		BC = T32(BC << 17) | (BC >> 15); \
		BD = T32(BD << 17) | (BD >> 15); \
		BE = T32(BE << 17) | (BE >> 15); \
		BF = T32(BF << 17) | (BF >> 15); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \
		A0B = T32(A0B + C6); \
		A0A = T32(A0A + C5); \
		A09 = T32(A09 + C4); \
		A08 = T32(A08 + C3); \
		A07 = T32(A07 + C2); \
		A06 = T32(A06 + C1); \
		A05 = T32(A05 + C0); \
		A04 = T32(A04 + CF); \
		A03 = T32(A03 + CE); \
		A02 = T32(A02 + CD); \
		A01 = T32(A01 + CC); \
		A00 = T32(A00 + CB); \
		A0B = T32(A0B + CA); \
		A0A = T32(A0A + C9); \
		A09 = T32(A09 + C8); \
		A08 = T32(A08 + C7); \
		A07 = T32(A07 + C6); \
		A06 = T32(A06 + C5); \
		A05 = T32(A05 + C4); \
		A04 = T32(A04 + C3); \
		A03 = T32(A03 + C2); \
		A02 = T32(A02 + C1); \
		A01 = T32(A01 + C0); \
		A00 = T32(A00 + CF); \
		A0B = T32(A0B + CE); \
		A0A = T32(A0A + CD); \
		A09 = T32(A09 + CC); \
		A08 = T32(A08 + CB); \
		A07 = T32(A07 + CA); \
		A06 = T32(A06 + C9); \
		A05 = T32(A05 + C8); \
		A04 = T32(A04 + C7); \
		A03 = T32(A03 + C6); \
		A02 = T32(A02 + C5); \
		A01 = T32(A01 + C4); \
		A00 = T32(A00 + C3); \
	} 

#define INCR_W   { \
		if ((Wlow = T32(Wlow + 1)) == 0) \
			Whigh = T32(Whigh + 1); \
	} 

                                        
__global__ void x13_shabal512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;


        uint32_t *inpHash = (uint32_t*)&g_hash[8 * hashPosition];
		
			
union {
uint8_t h1[64];
uint32_t h4[16];
uint64_t h8[8];
} hash;  

		
        
	    #pragma unroll 16
		for (int i=0;i<16;i++) {
			hash.h4[i]= inpHash[i];}
		
///////// input big /////////////////////        
		uint32_t A00 = A_512[0], A01 = A_512[1], A02 = A_512[2], A03 = A_512[3], A04 = A_512[4], A05 = A_512[5], A06 = A_512[6], A07 = A_512[7],
	    A08 = A_512[8], A09 = A_512[9], A0A = A_512[10], A0B = A_512[11];
    uint32_t B0 = B_512[0], B1 = B_512[1], B2 = B_512[2], B3 = B_512[3], B4 = B_512[4], B5 = B_512[5], B6 = B_512[6], B7 = B_512[7],
	    B8 = B_512[8], B9 = B_512[9], BA = B_512[10], BB = B_512[11], BC = B_512[12], BD = B_512[13], BE = B_512[14], BF = B_512[15];
    uint32_t C0 = C_512[0], C1 = C_512[1], C2 = C_512[2], C3 = C_512[3], C4 = C_512[4], C5 = C_512[5], C6 = C_512[6], C7 = C_512[7],
	    C8 = C_512[8], C9 = C_512[9], CA = C_512[10], CB = C_512[11], CC = C_512[12], CD = C_512[13], CE = C_512[14], CF = C_512[15];
    uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;
    uint32_t Wlow = 1, Whigh = 0;

    M0 = hash.h4[0];
    M1 = hash.h4[1];
    M2 = hash.h4[2];
    M3 = hash.h4[3];
    M4 = hash.h4[4];
    M5 = hash.h4[5];
    M6 = hash.h4[6];
    M7 = hash.h4[7];
    M8 = hash.h4[8];
    M9 = hash.h4[9];
    MA = hash.h4[10];
    MB = hash.h4[11];
    MC = hash.h4[12];
    MD = hash.h4[13];
    ME = hash.h4[14];
    MF = hash.h4[15];

    INPUT_BLOCK_ADD;
    XOR_W;
    APPLY_P;
    INPUT_BLOCK_SUB;
    SWAP_BC;
    INCR_W;

    M0 = 0x80;
    M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

    INPUT_BLOCK_ADD;
    XOR_W;
    APPLY_P;

 #pragma unroll 3
 for (unsigned i = 0; i < 3; i ++) {
	SWAP_BC;
	XOR_W;
	APPLY_P;
    }

    hash.h4[0] = B0;
    hash.h4[1] = B1;
    hash.h4[2] = B2;
    hash.h4[3] = B3;
    hash.h4[4] = B4;
    hash.h4[5] = B5;
    hash.h4[6] = B6;
    hash.h4[7] = B7;
    hash.h4[8] = B8;
    hash.h4[9] = B9;
    hash.h4[10] = BA;
    hash.h4[11] = BB;
    hash.h4[12] = BC;
    hash.h4[13] = BD;
    hash.h4[14] = BE;
    hash.h4[15] = BF;
      
      #pragma unroll 16
      for (int u = 0; u < 16; u ++) 
            inpHash[u] = hash.h4[u];    
 }
}


void x13_shabal512_cpu_init(int thr_id, int threads)
{
    
	cudaMemcpyToSymbol(A_512,A_init_512,sizeof(A_init_512),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(B_512,B_init_512,sizeof(B_init_512),0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(C_512,C_init_512,sizeof(C_init_512),0, cudaMemcpyHostToDevice);
}


__host__ void x13_shabal512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{

	const int threadsperblock = 256; // Alignment mit mixtab Grösse. NICHT ÄNDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	x13_shabal512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	MyStreamSynchronize(NULL, order, thr_id);
}
