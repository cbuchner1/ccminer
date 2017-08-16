// aus heavy.cu
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

 __constant__ uint32_t c_PaddedMessage80[32]; // padded message (80 bytes + padding)

static __constant__ uint32_t d_ShaviteInitVector[16];
static const uint32_t h_ShaviteInitVector[] = {
	SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
	SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
	SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
	SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
};

#include "cuda_x11_aes.cu"

static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, 0, 0x0123);
}

static __device__ __forceinline__ void AES_ROUND_NOKEY(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		x0, x1, x2, x3,
		y0, y1, y2, y3);

	x0 = y0;
	x1 = y1;
	x2 = y2;
	x3 = y3;
}

static __device__ __forceinline__ void KEY_EXPAND_ELT(
	const uint32_t* __restrict__ sharedMemory,
	uint32_t &k0, uint32_t &k1, uint32_t &k2, uint32_t &k3)
{
	uint32_t y0, y1, y2, y3;
	aes_round(sharedMemory,
		k0, k1, k2, k3,
		y0, y1, y2, y3);

	k0 = y1;
	k1 = y2;
	k2 = y3;
	k3 = y0;
}

static __device__ void
c512(const uint32_t* sharedMemory, uint32_t *state, uint32_t *msg, uint32_t count)
{
	uint32_t p0, p1, p2, p3, p4, p5, p6, p7;
	uint32_t p8, p9, pA, pB, pC, pD, pE, pF;
	uint32_t x0, x1, x2, x3;
	uint32_t rk00, rk01, rk02, rk03, rk04, rk05, rk06, rk07;
	uint32_t rk08, rk09, rk0A, rk0B, rk0C, rk0D, rk0E, rk0F;
	uint32_t rk10, rk11, rk12, rk13, rk14, rk15, rk16, rk17;
	uint32_t rk18, rk19, rk1A, rk1B, rk1C, rk1D, rk1E, rk1F;
	const uint32_t counter = count;

	p0 = state[0x0];
	p1 = state[0x1];
	p2 = state[0x2];
	p3 = state[0x3];
	p4 = state[0x4];
	p5 = state[0x5];
	p6 = state[0x6];
	p7 = state[0x7];
	p8 = state[0x8];
	p9 = state[0x9];
	pA = state[0xA];
	pB = state[0xB];
	pC = state[0xC];
	pD = state[0xD];
	pE = state[0xE];
	pF = state[0xF];
	/* round 0 */
	rk00 = msg[0];
	x0 = p4 ^ rk00;
	rk01 = msg[1];
	x1 = p5 ^ rk01;
	rk02 = msg[2];
	x2 = p6 ^ rk02;
	rk03 = msg[3];
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 = msg[4];
	x0 ^= rk04;
	rk05 = msg[5];
	x1 ^= rk05;
	rk06 = msg[6];
	x2 ^= rk06;
	rk07 = msg[7];
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 = msg[8];
	x0 ^= rk08;
	rk09 = msg[9];
	x1 ^= rk09;
	rk0A = msg[10];
	x2 ^= rk0A;
	rk0B = msg[11];
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C = msg[12];
	x0 ^= rk0C;
	rk0D = msg[13];
	x1 ^= rk0D;
	rk0E = msg[14];
	x2 ^= rk0E;
	rk0F = msg[15];
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 = msg[16];
	x0 = pC ^ rk10;
	rk11 = msg[17];
	x1 = pD ^ rk11;
	rk12 = msg[18];
	x2 = pE ^ rk12;
	rk13 = msg[19];
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 = msg[20];
	x0 ^= rk14;
	rk15 = msg[21];
	x1 ^= rk15;
	rk16 = msg[22];
	x2 ^= rk16;
	rk17 = msg[23];
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 = msg[24];
	x0 ^= rk18;
	rk19 = msg[25];
	x1 ^= rk19;
	rk1A = msg[26];
	x2 ^= rk1A;
	rk1B = msg[27];
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C = msg[28];
	x0 ^= rk1C;
	rk1D = msg[29];
	x1 ^= rk1D;
	rk1E = msg[30];
	x2 ^= rk1E;
	rk1F = msg[31];
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 1
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	rk00 ^= counter;
	rk03 ^= 0xFFFFFFFF;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	
	rk00 ^= rk19;
	x0 = pC ^ rk00;
	rk01 ^= rk1A;
	x1 = pD ^ rk01;
	rk02 ^= rk1B;
	x2 = pE ^ rk02;
	rk03 ^= rk1C;
	x3 = pF ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk10 ^= rk09;
	x0 = p4 ^ rk10;
	rk11 ^= rk0A;
	x1 = p5 ^ rk11;
	rk12 ^= rk0B;
	x2 = p6 ^ rk12;
	rk13 ^= rk0C;
	x3 = p7 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p8 ^ rk00;
	x1 = p9 ^ rk01;
	x2 = pA ^ rk02;
	x3 = pB ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p0 ^ rk10;
	x1 = p1 ^ rk11;
	x2 = p2 ^ rk12;
	x3 = p3 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk00 ^= rk19;
	x0 = p4 ^ rk00;
	rk01 ^= rk1A;
	x1 = p5 ^ rk01;
	rk02 ^= rk1B;
	x2 = p6 ^ rk02;
	rk03 ^= rk1C;
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 ^= rk09;
	x0 = pC ^ rk10;
	rk11 ^= rk0A;
	x1 = pD ^ rk11;
	rk12 ^= rk0B;
	x2 = pE ^ rk12;
	rk13 ^= rk0C;
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 2
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;	
	rk07 ^= SPH_T32(~counter);
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	
	rk00 ^= rk19;
	x0 = pC ^ rk00;
	rk01 ^= rk1A;
	x1 = pD ^ rk01;
	rk02 ^= rk1B;
	x2 = pE ^ rk02;
	rk03 ^= rk1C;
	x3 = pF ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk10 ^= rk09;
	x0 = p4 ^ rk10;
	rk11 ^= rk0A;
	x1 = p5 ^ rk11;
	rk12 ^= rk0B;
	x2 = p6 ^ rk12;
	rk13 ^= rk0C;
	x3 = p7 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p8 ^ rk00;
	x1 = p9 ^ rk01;
	x2 = pA ^ rk02;
	x3 = pB ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p0 ^ rk10;
	x1 = p1 ^ rk11;
	x2 = p2 ^ rk12;
	x3 = p3 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk00 ^= rk19;
	x0 = p4 ^ rk00;
	rk01 ^= rk1A;
	x1 = p5 ^ rk01;
	rk02 ^= rk1B;
	x2 = p6 ^ rk02;
	rk03 ^= rk1C;
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 ^= rk09;
	x0 = pC ^ rk10;
	rk11 ^= rk0A;
	x1 = pD ^ rk11;
	rk12 ^= rk0B;
	x2 = pE ^ rk12;
	rk13 ^= rk0C;
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	// 3
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	rk1E ^= counter;
	rk1F ^= 0xFFFFFFFF;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	
	rk00 ^= rk19;
	x0 = pC ^ rk00;
	rk01 ^= rk1A;
	x1 = pD ^ rk01;
	rk02 ^= rk1B;
	x2 = pE ^ rk02;
	rk03 ^= rk1C;
	x3 = pF ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;
	rk10 ^= rk09;
	x0 = p4 ^ rk10;
	rk11 ^= rk0A;
	x1 = p5 ^ rk11;
	rk12 ^= rk0B;
	x2 = p6 ^ rk12;
	rk13 ^= rk0C;
	x3 = p7 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	/* round 3, 7, 11 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p8 ^ rk00;
	x1 = p9 ^ rk01;
	x2 = pA ^ rk02;
	x3 = pB ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p0 ^ rk10;
	x1 = p1 ^ rk11;
	x2 = p2 ^ rk12;
	x3 = p3 ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15;
	rk1A ^= rk16;
	rk1B ^= rk17;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	/* round 4, 8, 12 */
	rk00 ^= rk19;
	x0 = p4 ^ rk00;
	rk01 ^= rk1A;
	x1 = p5 ^ rk01;
	rk02 ^= rk1B;
	x2 = p6 ^ rk02;
	rk03 ^= rk1C;
	x3 = p7 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk04 ^= rk1D;
	x0 ^= rk04;
	rk05 ^= rk1E;
	x1 ^= rk05;
	rk06 ^= rk1F;
	x2 ^= rk06;
	rk07 ^= rk00;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk08 ^= rk01;
	x0 ^= rk08;
	rk09 ^= rk02;
	x1 ^= rk09;
	rk0A ^= rk03;
	x2 ^= rk0A;
	rk0B ^= rk04;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk0C ^= rk05;
	x0 ^= rk0C;
	rk0D ^= rk06;
	x1 ^= rk0D;
	rk0E ^= rk07;
	x2 ^= rk0E;
	rk0F ^= rk08;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p0 ^= x0;
	p1 ^= x1;
	p2 ^= x2;
	p3 ^= x3;
	rk10 ^= rk09;
	x0 = pC ^ rk10;
	rk11 ^= rk0A;
	x1 = pD ^ rk11;
	rk12 ^= rk0B;
	x2 = pE ^ rk12;
	rk13 ^= rk0C;
	x3 = pF ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk14 ^= rk0D;
	x0 ^= rk14;
	rk15 ^= rk0E;
	x1 ^= rk15;
	rk16 ^= rk0F;
	x2 ^= rk16;
	rk17 ^= rk10;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk18 ^= rk11;
	x0 ^= rk18;
	rk19 ^= rk12;
	x1 ^= rk19;
	rk1A ^= rk13;
	x2 ^= rk1A;
	rk1B ^= rk14;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	rk1C ^= rk15;
	x0 ^= rk1C;
	rk1D ^= rk16;
	x1 ^= rk1D;
	rk1E ^= rk17;
	x2 ^= rk1E;
	rk1F ^= rk18;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p8 ^= x0;
	p9 ^= x1;
	pA ^= x2;
	pB ^= x3;

	/* round 13 */
	KEY_EXPAND_ELT(sharedMemory, rk00, rk01, rk02, rk03);
	rk00 ^= rk1C;
	rk01 ^= rk1D;
	rk02 ^= rk1E;
	rk03 ^= rk1F;
	x0 = p0 ^ rk00;
	x1 = p1 ^ rk01;
	x2 = p2 ^ rk02;
	x3 = p3 ^ rk03;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk04, rk05, rk06, rk07);
	rk04 ^= rk00;
	rk05 ^= rk01;
	rk06 ^= rk02;
	rk07 ^= rk03;
	x0 ^= rk04;
	x1 ^= rk05;
	x2 ^= rk06;
	x3 ^= rk07;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk08, rk09, rk0A, rk0B);
	rk08 ^= rk04;
	rk09 ^= rk05;
	rk0A ^= rk06;
	rk0B ^= rk07;
	x0 ^= rk08;
	x1 ^= rk09;
	x2 ^= rk0A;
	x3 ^= rk0B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk0C, rk0D, rk0E, rk0F);
	rk0C ^= rk08;
	rk0D ^= rk09;
	rk0E ^= rk0A;
	rk0F ^= rk0B;
	x0 ^= rk0C;
	x1 ^= rk0D;
	x2 ^= rk0E;
	x3 ^= rk0F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	pC ^= x0;
	pD ^= x1;
	pE ^= x2;
	pF ^= x3;
	KEY_EXPAND_ELT(sharedMemory, rk10, rk11, rk12, rk13);
	rk10 ^= rk0C;
	rk11 ^= rk0D;
	rk12 ^= rk0E;
	rk13 ^= rk0F;
	x0 = p8 ^ rk10;
	x1 = p9 ^ rk11;
	x2 = pA ^ rk12;
	x3 = pB ^ rk13;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk14, rk15, rk16, rk17);
	rk14 ^= rk10;
	rk15 ^= rk11;
	rk16 ^= rk12;
	rk17 ^= rk13;
	x0 ^= rk14;
	x1 ^= rk15;
	x2 ^= rk16;
	x3 ^= rk17;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk18, rk19, rk1A, rk1B);
	rk18 ^= rk14;
	rk19 ^= rk15 ^ counter;
	rk1A ^= rk16;
	rk1B ^= rk17 ^ 0xFFFFFFFF;
	x0 ^= rk18;
	x1 ^= rk19;
	x2 ^= rk1A;
	x3 ^= rk1B;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	KEY_EXPAND_ELT(sharedMemory, rk1C, rk1D, rk1E, rk1F);
	rk1C ^= rk18;
	rk1D ^= rk19;
	rk1E ^= rk1A;
	rk1F ^= rk1B;
	x0 ^= rk1C;
	x1 ^= rk1D;
	x2 ^= rk1E;
	x3 ^= rk1F;
	AES_ROUND_NOKEY(sharedMemory, x0, x1, x2, x3);
	p4 ^= x0;
	p5 ^= x1;
	p6 ^= x2;
	p7 ^= x3;
	state[0x0] ^= p8;
	state[0x1] ^= p9;
	state[0x2] ^= pA;
	state[0x3] ^= pB;
	state[0x4] ^= pC;
	state[0x5] ^= pD;
	state[0x6] ^= pE;
	state[0x7] ^= pF;
	state[0x8] ^= p0;
	state[0x9] ^= p1;
	state[0xA] ^= p2;
	state[0xB] ^= p3;
	state[0xC] ^= p4;
	state[0xD] ^= p5;
	state[0xE] ^= p6;
	state[0xF] ^= p7;
}



__global__ void x11_shavite512_gpu_hash_80(int threads, uint32_t startNounce, void *outputHash)
{
	__shared__ uint32_t sharedMemory[1024];

	        aes_gpu_init(sharedMemory);	
  

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
		uint32_t nounce = startNounce + thread;
	
		// kopiere init-state
		uint32_t state[16];


#pragma unroll 16
		for(int i=0;i<16;i++) {
			state[i] = d_ShaviteInitVector[i];}

		uint32_t msg[32];

#pragma unroll 32
		for(int i=0;i<32;i++) {			
			msg[i]  = c_PaddedMessage80[i];}
		    msg[19] = cuda_swab32(nounce);
			msg[20] = 0x80;
			msg[27] = 0x2800000;
			msg[31] = 0x2000000;

		c512(sharedMemory, state, msg,640);

uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;

#pragma unroll 16
		for(int i=0;i<16;i++)
			outHash[i] = state[i];


	} //thread < threads
}
// Die Hash-Funktion
__global__ void x11_shavite512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__shared__ uint32_t sharedMemory[1024];

	aes_gpu_init(sharedMemory);


    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

		// kopiere init-state
		uint32_t state[16];

#pragma unroll 16
		for(int i=0;i<16;i++)
			state[i] = d_ShaviteInitVector[i];

		// nachricht laden
		uint32_t msg[32];

		// f�lle die Nachricht mit 64-byte (vorheriger Hash)
#pragma unroll 16
		for(int i=0;i<16;i++)
			msg[i] = Hash[i];			

		// Nachrichtenende
		msg[16] = 0x80;
#pragma unroll 10
		for(int i=17;i<27;i++)
			msg[i] = 0;

		msg[27] = 0x02000000;
		msg[28] = 0;
		msg[29] = 0;
		msg[30] = 0;
		msg[31] = 0x02000000;

		c512(sharedMemory, state, msg, 512);

#pragma unroll 16
		for(int i=0;i<16;i++)
			Hash[i] = state[i];
    } // thread < threads
}


// Setup-Funktionen
__host__ void x11_shavite512_cpu_init(int thr_id, int threads)
{
	aes_cpu_init();

	cudaMemcpyToSymbol( d_ShaviteInitVector,
                        h_ShaviteInitVector,
                        sizeof(h_ShaviteInitVector),
                        0, cudaMemcpyHostToDevice);
}

__host__ void x11_shavite512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    size_t shared_size = 0;

    x11_shavite512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void x11_shavite512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	x11_shavite512_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}
__host__ void x11_shavite512_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 32*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

