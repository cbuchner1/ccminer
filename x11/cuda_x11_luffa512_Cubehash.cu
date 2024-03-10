/*
 * Merged LUFFA512 64 + CUBE512 64 - from sp
 */

#include "cuda_helper.h"

#define MULT0(a) {\
	tmp = a[7]; \
	a[7] = a[6]; \
	a[6] = a[5]; \
	a[5] = a[4]; \
	a[4] = a[3] ^ tmp; \
	a[3] = a[2] ^ tmp; \
	a[2] = a[1]; \
	a[1] = a[0] ^ tmp; \
	a[0] = tmp; \
}

#define MULT2(a,j) { \
	tmp = a[(j<<3)+7]; \
	a[(j*8)+7] = a[(j*8)+6]; \
	a[(j*8)+6] = a[(j*8)+5]; \
	a[(j*8)+5] = a[(j*8)+4]; \
	a[(j*8)+4] = a[(j*8)+3] ^ tmp; \
	a[(j*8)+3] = a[(j*8)+2] ^ tmp; \
	a[(j*8)+2] = a[(j*8)+1]; \
	a[(j*8)+1] = a[(j*8)+0] ^ tmp; \
	a[j*8] = tmp; \
}

#define TWEAK(a0,a1,a2,a3,j) { \
	a0 = ROTL32(a0,j); \
	a1 = ROTL32(a1,j); \
	a2 = ROTL32(a2,j); \
	a3 = ROTL32(a3,j); \
}

#define STEP(c0,c1) { \
	SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp); \
	SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp); \
	MIXWORD(chainv[0],chainv[4]); \
	MIXWORD(chainv[1],chainv[5]); \
	MIXWORD(chainv[2],chainv[6]); \
	MIXWORD(chainv[3],chainv[7]); \
	ADD_CONSTANT(chainv[0],chainv[4],c0,c1); \
}

#define SUBCRUMB(a0,a1,a2,a3,a4) { \
	a4  = a0; \
	a0 |= a1; \
	a2 ^= a3; \
	a1  = ~a1;\
	a0 ^= a3; \
	a3 &= a4; \
	a1 ^= a3; \
	a3 ^= a2; \
	a2 &= a0; \
	a0  = ~a0;\
	a2 ^= a1; \
	a1 |= a3; \
	a4 ^= a1; \
	a3 ^= a2; \
	a2 &= a1; \
	a1 ^= a0; \
	a0  = a4; \
}

#define MIXWORD(a0,a4) { \
	a4 ^= a0; \
	a0  = ROTL32(a0,2); \
	a0 ^= a4; \
	a4  = ROTL32(a4,14); \
	a4 ^= a0; \
	a0  = ROTL32(a0,10); \
	a0 ^= a4; \
	a4  = ROTL32(a4,1); \
}

#define ADD_CONSTANT(a0,b0,c0,c1) { \
	a0 ^= c0; \
	b0 ^= c1; \
}

__device__ __constant__ uint32_t c_CNS[80] = {
	0x303994a6,0xe0337818,0xc0e65299,0x441ba90d,
	0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f,
	0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4,
	0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
	0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4,
	0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28,
	0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b,
	0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
	0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72,
	0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7,
	0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719,
	0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
	0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91,
	0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be,
	0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5,
	0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
	0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab,
	0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0,
	0x78602649,0x29131ab6,0x8edae952,0x0fc053c3,
	0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31
};

// Precalculated chaining values
__device__ __constant__ uint32_t c_IV[40] = {
	0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,
	0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
	0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,
	0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
	0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,
	0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
	0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,
	0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
	0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,
	0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
};

/***************************************************/
__device__ __forceinline__
static void rnd512(uint32_t *statebuffer, uint32_t *statechainv)
{
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;
	int i,j;

	#pragma unroll
	for(i=0;i<8;i++) {
		t[i] = 0;
		#pragma unroll 5
		for(j=0;j<5;j++)
		   t[i] ^= statechainv[i+8*j];
	}

	MULT0(t);

	#pragma unroll
	for(j=0;j<5;j++) {
		#pragma unroll
		for(i=0;i<8;i++)
			statechainv[i+8*j] ^= t[i];
	}

	#pragma unroll
	for(j=0;j<5;j++) {
		#pragma unroll
		for(i=0;i<8;i++)
			t[i+8*j] = statechainv[i+8*j];
	}

	MULT0(statechainv);
	#pragma unroll 4
	for(j=1;j<5;j++) {
		MULT2(statechainv, j);
	}

	#pragma unroll
	for(j=0;j<5;j++) {
		#pragma unroll
		for(i=0;i<8;i++)
			statechainv[8*j+i] ^= t[8*((j+1)%5)+i];
	}

	#pragma unroll
	for(j=0;j<5;j++) {
		#pragma unroll
		for(i=0;i<8;i++)
			t[i+8*j] = statechainv[i+8*j];
	}

	MULT0(statechainv);
	#pragma unroll 4
	for(j=1;j<5;j++) {
		MULT2(statechainv, j);
	}

	#pragma unroll
	for(j=0;j<5;j++) {
		#pragma unroll
		for(i=0;i<8;i++)
			statechainv[8*j+i] ^= t[8*((j+4)%5)+i];
	}

	#pragma unroll
	for(j=0;j<5;j++) {
		#pragma unroll 8
		for(i=0;i<8;i++)
			statechainv[i+8*j] ^= statebuffer[i];
		MULT0(statebuffer);
	}

	#pragma unroll
	for(i=0;i<8;i++) {
		chainv[i] = statechainv[i];
	}

	#pragma unroll 1
	for(i=0;i<8;i++) {
		STEP(c_CNS[(2*i)],c_CNS[(2*i)+1]);
	}

	#pragma unroll
	for(i=0;i<8;i++) {
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i+8];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],1);

	#pragma unroll 1
	for(i=0;i<8;i++) {
		STEP(c_CNS[(2*i)+16],c_CNS[(2*i)+16+1]);
	}

	#pragma unroll
	for(i=0;i<8;i++) {
		statechainv[i+8] = chainv[i];
		chainv[i] = statechainv[i+16];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],2);

	#pragma unroll 1
	for(i=0;i<8;i++) {
		STEP(c_CNS[(2*i)+32],c_CNS[(2*i)+32+1]);
	}

	#pragma unroll
	for(i=0;i<8;i++) {
		statechainv[i+16] = chainv[i];
		chainv[i] = statechainv[i+24];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],3);

	#pragma unroll 1
	for(i=0;i<8;i++) {
		STEP(c_CNS[(2*i)+48],c_CNS[(2*i)+48+1]);
	}

	#pragma unroll
	for(i=0;i<8;i++) {
		statechainv[i+24] = chainv[i];
		chainv[i] = statechainv[i+32];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],4);

	#pragma unroll 1
	for(i=0;i<8;i++) {
		STEP(c_CNS[(2*i)+64],c_CNS[(2*i)+64+1]);
	}

	#pragma unroll
	for(i=0;i<8;i++) {
		statechainv[i+32] = chainv[i];
	}
}

__device__ __forceinline__
static void rnd512_first(uint32_t state[40], uint32_t buffer[8])
{
	uint32_t chainv[8];
	uint32_t tmp;
	int i, j;

	for (j = 0; j<5; j++) {
		state[8 * j] ^= buffer[0];

		#pragma unroll 7
		for (i = 1; i<8; i++)
			state[i + 8 * j] ^= buffer[i];
		MULT0(buffer);
	}

	#pragma unroll
	for (i = 0; i<8; i++)
		chainv[i] = state[i];

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i] = chainv[i];
		chainv[i] = state[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 8] = chainv[i];
		chainv[i] = state[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 16] = chainv[i];
		chainv[i] = state[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 24] = chainv[i];
		chainv[i] = state[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++)
		state[i + 32] = chainv[i];
}

/***************************************************/
__device__ __forceinline__
static void rnd512_nullhash(uint32_t *state)
{
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;
	int i, j;

	#pragma unroll
	for (i = 0; i<8; i++) {
		t[i] = state[i + 8 * 0];
		#pragma unroll 4
		for (j = 1; j<5; j++)
			t[i] ^= state[i + 8 * j];
	}

	MULT0(t);

	#pragma unroll
	for (j = 0; j<5; j++) {
		#pragma unroll
		for (i = 0; i<8; i++)
			state[i + 8 * j] ^= t[i];
	}

	#pragma unroll
	for (j = 0; j<5; j++) {
		#pragma unroll
		for (i = 0; i<8; i++)
			t[i + 8 * j] = state[i + 8 * j];
	}

	MULT0(state);
	#pragma unroll 4
	for(j=1; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll
	for (j = 0; j<5; j++) {
		#pragma unroll
		for (i = 0; i<8; i++)
			state[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
	}

	#pragma unroll
	for (j = 0; j<5; j++) {
		#pragma unroll 8
		for (i = 0; i<8; i++)
			t[i + 8 * j] = state[i + 8 * j];
	}

	MULT0(state);
	#pragma unroll 4
	for(j=1; j<5; j++) {
		MULT2(state, j);
	}

	#pragma unroll
	for (j = 0; j<5; j++) {
		#pragma unroll
		for (i = 0; i<8; i++)
			state[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
	}

	#pragma unroll
	for (i = 0; i<8; i++)
		chainv[i] = state[i];

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i] = chainv[i];
		chainv[i] = state[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 8] = chainv[i];
		chainv[i] = state[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 16] = chainv[i];
		chainv[i] = state[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 24] = chainv[i];
		chainv[i] = state[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

	#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

	#pragma unroll
	for (i = 0; i<8; i++) {
		state[i + 32] = chainv[i];
	}
}

__device__ __forceinline__
static void Update512(uint32_t *statebuffer, uint32_t *statechainv, const uint32_t *data)
{
	#pragma unroll
	for (int i = 0; i < 8; i++) statebuffer[i] = cuda_swab32(data[i]);
	rnd512_first(statechainv, statebuffer);

	#pragma unroll
	for (int i = 0; i < 8; i++) statebuffer[i] = cuda_swab32(data[i + 8]);
	rnd512(statebuffer, statechainv);
}

/***************************************************/
__device__ __forceinline__
static void finalization512(uint32_t *statebuffer, uint32_t *statechainv, uint32_t *b)
{
	int i,j;

	statebuffer[0] = 0x80000000;
	#pragma unroll 7
	for(int i=1;i<8;i++) statebuffer[i] = 0;
	rnd512(statebuffer, statechainv);

	/*---- blank round with m=0 ----*/
	rnd512_nullhash(statechainv);

	#pragma unroll
	for(i=0;i<8;i++) {
		b[i] = statechainv[i];
		#pragma unroll 4
		for(j=1;j<5;j++) {
			b[i] ^= statechainv[i+8*j];
		}
		b[i] = cuda_swab32((b[i]));
	}

	rnd512_nullhash(statechainv);

	#pragma unroll
	for(i=0;i<8;i++) {
		b[8 + i] = statechainv[i];
		#pragma unroll 4
		for(j=1;j<5;j++) {
			b[8+i] ^= statechainv[i+8*j];
		}
		b[8 + i] = cuda_swab32((b[8 + i]));
	}
}

#define ROUND_EVEN { \
	xg = (x0 + xg); \
	x0 = ROTL32(x0, 7); \
	xh = (x1 + xh); \
	x1 = ROTL32(x1, 7); \
	xi = (x2 + xi); \
	x2 = ROTL32(x2, 7); \
	xj = (x3 + xj); \
	x3 = ROTL32(x3, 7); \
	xk = (x4 + xk); \
	x4 = ROTL32(x4, 7); \
	xl = (x5 + xl); \
	x5 = ROTL32(x5, 7); \
	xm = (x6 + xm); \
	x6 = ROTL32(x6, 7); \
	xn = (x7 + xn); \
	x7 = ROTL32(x7, 7); \
	xo = (x8 + xo); \
	x8 = ROTL32(x8, 7); \
	xp = (x9 + xp); \
	x9 = ROTL32(x9, 7); \
	xq = (xa + xq); \
	xa = ROTL32(xa, 7); \
	xr = (xb + xr); \
	xb = ROTL32(xb, 7); \
	xs = (xc + xs); \
	xc = ROTL32(xc, 7); \
	xt = (xd + xt); \
	xd = ROTL32(xd, 7); \
	xu = (xe + xu); \
	xe = ROTL32(xe, 7); \
	xv = (xf + xv); \
	xf = ROTL32(xf, 7); \
	x8 ^= xg; \
	x9 ^= xh; \
	xa ^= xi; \
	xb ^= xj; \
	xc ^= xk; \
	xd ^= xl; \
	xe ^= xm; \
	xf ^= xn; \
	x0 ^= xo; \
	x1 ^= xp; \
	x2 ^= xq; \
	x3 ^= xr; \
	x4 ^= xs; \
	x5 ^= xt; \
	x6 ^= xu; \
	x7 ^= xv; \
	xi = (x8 + xi); \
	x8 = ROTL32(x8, 11); \
	xj = (x9 + xj); \
	x9 = ROTL32(x9, 11); \
	xg = (xa + xg); \
	xa = ROTL32(xa, 11); \
	xh = (xb + xh); \
	xb = ROTL32(xb, 11); \
	xm = (xc + xm); \
	xc = ROTL32(xc, 11); \
	xn = (xd + xn); \
	xd = ROTL32(xd, 11); \
	xk = (xe + xk); \
	xe = ROTL32(xe, 11); \
	xl = (xf + xl); \
	xf = ROTL32(xf, 11); \
	xq = (x0 + xq); \
	x0 = ROTL32(x0, 11); \
	xr = (x1 + xr); \
	x1 = ROTL32(x1, 11); \
	xo = (x2 + xo); \
	x2 = ROTL32(x2, 11); \
	xp = (x3 + xp); \
	x3 = ROTL32(x3, 11); \
	xu = (x4 + xu); \
	x4 = ROTL32(x4, 11); \
	xv = (x5 + xv); \
	x5 = ROTL32(x5, 11); \
	xs = (x6 + xs); \
	x6 = ROTL32(x6, 11); \
	xt = (x7 + xt); \
	x7 = ROTL32(x7, 11); \
	xc ^= xi; \
	xd ^= xj; \
	xe ^= xg; \
	xf ^= xh; \
	x8 ^= xm; \
	x9 ^= xn; \
	xa ^= xk; \
	xb ^= xl; \
	x4 ^= xq; \
	x5 ^= xr; \
	x6 ^= xo; \
	x7 ^= xp; \
	x0 ^= xu; \
	x1 ^= xv; \
	x2 ^= xs; \
	x3 ^= xt; \
}

#define ROUND_ODD { \
	xj = (xc + xj); \
	xc = ROTL32(xc, 7); \
	xi = (xd + xi); \
	xd = ROTL32(xd, 7); \
	xh = (xe + xh); \
	xe = ROTL32(xe, 7); \
	xg = (xf + xg); \
	xf = ROTL32(xf, 7); \
	xn = (x8 + xn); \
	x8 = ROTL32(x8, 7); \
	xm = (x9 + xm); \
	x9 = ROTL32(x9, 7); \
	xl = (xa + xl); \
	xa = ROTL32(xa, 7); \
	xk = (xb + xk); \
	xb = ROTL32(xb, 7); \
	xr = (x4 + xr); \
	x4 = ROTL32(x4, 7); \
	xq = (x5 + xq); \
	x5 = ROTL32(x5, 7); \
	xp = (x6 + xp); \
	x6 = ROTL32(x6, 7); \
	xo = (x7 + xo); \
	x7 = ROTL32(x7, 7); \
	xv = (x0 + xv); \
	x0 = ROTL32(x0, 7); \
	xu = (x1 + xu); \
	x1 = ROTL32(x1, 7); \
	xt = (x2 + xt); \
	x2 = ROTL32(x2, 7); \
	xs = (x3 + xs); \
	x3 = ROTL32(x3, 7); \
	x4 ^= xj; \
	x5 ^= xi; \
	x6 ^= xh; \
	x7 ^= xg; \
	x0 ^= xn; \
	x1 ^= xm; \
	x2 ^= xl; \
	x3 ^= xk; \
	xc ^= xr; \
	xd ^= xq; \
	xe ^= xp; \
	xf ^= xo; \
	x8 ^= xv; \
	x9 ^= xu; \
	xa ^= xt; \
	xb ^= xs; \
	xh = (x4 + xh); \
	x4 = ROTL32(x4, 11); \
	xg = (x5 + xg); \
	x5 = ROTL32(x5, 11); \
	xj = (x6 + xj); \
	x6 = ROTL32(x6, 11); \
	xi = (x7 + xi); \
	x7 = ROTL32(x7, 11); \
	xl = (x0 + xl); \
	x0 = ROTL32(x0, 11); \
	xk = (x1 + xk); \
	x1 = ROTL32(x1, 11); \
	xn = (x2 + xn); \
	x2 = ROTL32(x2, 11); \
	xm = (x3 + xm); \
	x3 = ROTL32(x3, 11); \
	xp = (xc + xp); \
	xc = ROTL32(xc, 11); \
	xo = (xd + xo); \
	xd = ROTL32(xd, 11); \
	xr = (xe + xr); \
	xe = ROTL32(xe, 11); \
	xq = (xf + xq); \
	xf = ROTL32(xf, 11); \
	xt = (x8 + xt); \
	x8 = ROTL32(x8, 11); \
	xs = (x9 + xs); \
	x9 = ROTL32(x9, 11); \
	xv = (xa + xv); \
	xa = ROTL32(xa, 11); \
	xu = (xb + xu); \
	xb = ROTL32(xb, 11); \
	x0 ^= xh; \
	x1 ^= xg; \
	x2 ^= xj; \
	x3 ^= xi; \
	x4 ^= xl; \
	x5 ^= xk; \
	x6 ^= xn; \
	x7 ^= xm; \
	x8 ^= xp; \
	x9 ^= xo; \
	xa ^= xr; \
	xb ^= xq; \
	xc ^= xt; \
	xd ^= xs; \
	xe ^= xv; \
	xf ^= xu; \
}

#define SIXTEEN_ROUNDS \
	for (int j = 0; j < 8; j ++) { \
		ROUND_EVEN; \
		ROUND_ODD; \
	}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(256, 4)
#endif
void x11_luffaCubehash512_gpu_hash_64(uint32_t threads, uint32_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t statechainv[40] = {
			0x8bb0a761, 0xc2e4aa8b, 0x2d539bc9, 0x381408f8,
			0x478f6633, 0x255a46ff, 0x581c37f7, 0x601c2e8e,
			0x266c5f9d, 0xc34715d8, 0x8900670e, 0x51a540be,
			0xe4ce69fb, 0x5089f4d4, 0x3cc0a506, 0x609bcb02,
			0xa4e3cd82, 0xd24fd6ca, 0xc0f196dc, 0xcf41eafe,
			0x0ff2e673, 0x303804f2, 0xa7b3cd48, 0x677addd4,
			0x66e66a8a, 0x2303208f, 0x486dafb4, 0xc0d37dc6,
			0x634d15af, 0xe5af6747, 0x10af7e38, 0xee7e6428,
			0x01262e5d, 0xc92c2e64, 0x82fee966, 0xcea738d3,
			0x867de2b0, 0xe0714818, 0xda6e831f, 0xa7062529
		};

		uint32_t statebuffer[8];
		uint32_t *const Hash = &g_hash[thread * 16U];

		Update512(statebuffer, statechainv, Hash);
		finalization512(statebuffer, statechainv, Hash);

		//Cubehash

		uint32_t x0 = 0x2AEA2A61, x1 = 0x50F494D4, x2 = 0x2D538B8B, x3 = 0x4167D83E;
		uint32_t x4 = 0x3FEE2313, x5 = 0xC701CF8C, x6 = 0xCC39968E, x7 = 0x50AC5695;
		uint32_t x8 = 0x4D42C787, x9 = 0xA647A8B3, xa = 0x97CF0BEF, xb = 0x825B4537;
		uint32_t xc = 0xEEF864D2, xd = 0xF22090C4, xe = 0xD0E5CD33, xf = 0xA23911AE;
		uint32_t xg = 0xFCD398D9, xh = 0x148FE485, xi = 0x1B017BEF, xj = 0xB6444532;
		uint32_t xk = 0x6A536159, xl = 0x2FF5781C, xm = 0x91FA7934, xn = 0x0DBADEA9;
		uint32_t xo = 0xD65C8A2B, xp = 0xA5A70E75, xq = 0xB1C62456, xr = 0xBC796576;
		uint32_t xs = 0x1921C8F7, xt = 0xE7989AF1, xu = 0x7795D246, xv = 0xD43E3B44;

		x0 ^= Hash[0];
		x1 ^= Hash[1];
		x2 ^= Hash[2];
		x3 ^= Hash[3];
		x4 ^= Hash[4];
		x5 ^= Hash[5];
		x6 ^= Hash[6];
		x7 ^= Hash[7];

		SIXTEEN_ROUNDS;

		x0 ^= Hash[8];
		x1 ^= Hash[9];
		x2 ^= Hash[10];
		x3 ^= Hash[11];
		x4 ^= Hash[12];
		x5 ^= Hash[13];
		x6 ^= Hash[14];
		x7 ^= Hash[15];

		SIXTEEN_ROUNDS;
		x0 ^= 0x80;

		SIXTEEN_ROUNDS;
		xv ^= 1;

		for (int i = 3; i < 13; i++) {
			SIXTEEN_ROUNDS;
		}

		Hash[0] = x0;
		Hash[1] = x1;
		Hash[2] = x2;
		Hash[3] = x3;
		Hash[4] = x4;
		Hash[5] = x5;
		Hash[6] = x6;
		Hash[7] = x7;
		Hash[8] = x8;
		Hash[9] = x9;
		Hash[10] = xa;
		Hash[11] = xb;
		Hash[12] = xc;
		Hash[13] = xd;
		Hash[14] = xe;
		Hash[15] = xf;
	}
}

__host__
void x11_luffaCubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x11_luffaCubehash512_gpu_hash_64 <<<grid, block>>> (threads, d_hash);
	MyStreamSynchronize(NULL, order, thr_id);
}

// Setup
__host__
void x11_luffaCubehash512_cpu_init(int thr_id, uint32_t threads) {}
