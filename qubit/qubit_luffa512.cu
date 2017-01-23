/*******************************************************************************
 * luffa512 for 80-bytes input (with midstate precalc by klausT)
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_helper.h"

static __constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)
static __constant__ uint32_t statebufferpre[8];
static __constant__ uint32_t statechainvpre[40];

#define MULT2(a,j) {\
	tmp = a[7+(8*j)];\
	a[7+(8*j)] = a[6+(8*j)];\
	a[6+(8*j)] = a[5+(8*j)];\
	a[5+(8*j)] = a[4+(8*j)];\
	a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
	a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
	a[2+(8*j)] = a[1+(8*j)];\
	a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
	a[0+(8*j)] = tmp;\
}

#define TWEAK(a0,a1,a2,a3,j) { \
	a0 = (a0<<(j))|(a0>>(32-j));\
	a1 = (a1<<(j))|(a1>>(32-j));\
	a2 = (a2<<(j))|(a2>>(32-j));\
	a3 = (a3<<(j))|(a3>>(32-j));\
}

#define STEP(c0,c1) { \
	SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
	SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
	MIXWORD(chainv[0],chainv[4]);\
	MIXWORD(chainv[1],chainv[5]);\
	MIXWORD(chainv[2],chainv[6]);\
	MIXWORD(chainv[3],chainv[7]);\
	ADD_CONSTANT(chainv[0],chainv[4],c0,c1);\
}

#define SUBCRUMB(a0,a1,a2,a3,a4)\
	a4  = a0;\
	a0 |= a1;\
	a2 ^= a3;\
	a1  = ~a1;\
	a0 ^= a3;\
	a3 &= a4;\
	a1 ^= a3;\
	a3 ^= a2;\
	a2 &= a0;\
	a0  = ~a0;\
	a2 ^= a1;\
	a1 |= a3;\
	a4 ^= a1;\
	a3 ^= a2;\
	a2 &= a1;\
	a1 ^= a0;\
	a0  = a4;

#define MIXWORD(a0,a4)\
	a4 ^= a0;\
	a0  = (a0<<2) | (a0>>(30));\
	a0 ^= a4;\
	a4  = (a4<<14) | (a4>>(18));\
	a4 ^= a0;\
	a0  = (a0<<10) | (a0>>(22));\
	a0 ^= a4;\
	a4  = (a4<<1) | (a4>>(31));

#define ADD_CONSTANT(a0,b0,c0,c1)\
	a0 ^= c0;\
	b0 ^= c1;

/* initial values of chaining variables */
__constant__ uint32_t c_IV[40];
static const uint32_t h_IV[40] = {
	0x6d251e69,0x44b051e0,0x4eaa6fb4,0xdbf78465,
	0x6e292011,0x90152df4,0xee058139,0xdef610bb,
	0xc3b44b95,0xd9d2f256,0x70eee9a0,0xde099fa3,
	0x5d9b0557,0x8fc944b3,0xcf1ccf0e,0x746cd581,
	0xf7efc89d,0x5dba5781,0x04016ce5,0xad659c05,
	0x0306194f,0x666d1836,0x24aa230a,0x8b264ae7,
	0x858075d5,0x36d79cce,0xe571f7d7,0x204b1f67,
	0x35870c6a,0x57e9e923,0x14bcb808,0x7cde72ce,
	0x6c68e9be,0x5ec41e22,0xc825b7c7,0xaffb4363,
	0xf5df3999,0x0fc688f1,0xb07224cc,0x03e86cea};

__constant__ uint32_t c_CNS[80];
static const uint32_t h_CNS[80] = {
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
	0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31};


/***************************************************/
__device__ __forceinline__
void rnd512(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i,j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

	#pragma unroll 8
	for(i=0; i<8; i++) {
		t[i]=0;
		#pragma unroll 5
		for(j=0; j<5; j++)
			t[i] ^= statechainv[i+8*j];
	}

	MULT2(t, 0);

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[i+8*j] ^= t[i];
	}

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			t[i+8*j] = statechainv[i+8*j];
	}

	#pragma unroll
	for(j=0; j<5; j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[8*j+i] ^= t[8*((j+1)%5)+i];
	}

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			t[i+8*j] = statechainv[i+8*j];
	}

	#pragma unroll
	for(j=0; j<5; j++)
		MULT2(statechainv, j);

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[8*j+i] ^= t[8*((j+4)%5)+i];
	}

	#pragma unroll 5
	for(j=0; j<5; j++) {
		#pragma unroll 8
		for(i=0; i<8; i++)
			statechainv[i+8*j] ^= statebuffer[i];
		MULT2(statebuffer, 0);
	}

	#pragma unroll
	for(i=0; i<8; i++)
		chainv[i] = statechainv[i];

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)], c_CNS[(2*i)+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i+8];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],1);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+16], c_CNS[(2*i)+16+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i+8] = chainv[i];
		chainv[i] = statechainv[i+16];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],2);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+32],c_CNS[(2*i)+32+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i+16] = chainv[i];
		chainv[i] = statechainv[i+24];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],3);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+48],c_CNS[(2*i)+48+1]);

	#pragma unroll
	for(i=0; i<8; i++) {
		statechainv[i+24] = chainv[i];
		chainv[i] = statechainv[i+32];
	}

	TWEAK(chainv[4],chainv[5],chainv[6],chainv[7],4);

	#pragma unroll
	for(i=0; i<8; i++)
		STEP(c_CNS[(2*i)+64],c_CNS[(2*i)+64+1]);

	#pragma unroll 8
	for(i=0; i<8; i++)
		statechainv[i+32] = chainv[i];
}

static void rnd512_cpu(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

	for (i = 0; i<8; i++) {
		t[i] = statechainv[i];
		for (j = 1; j<5; j++)
			t[i] ^= statechainv[i + 8 * j];
	}

	MULT2(t, 0);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[i + 8 * j] ^= t[i];
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			t[i + 8 * j] = statechainv[i + 8 * j];
	}

	for (j = 0; j<5; j++)
		MULT2(statechainv, j);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			t[i + 8 * j] = statechainv[i + 8 * j];
	}

	for (j = 0; j<5; j++)
		MULT2(statechainv, j);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++)
			statechainv[i + 8 * j] ^= statebuffer[i];
		MULT2(statebuffer, 0);
	}

	for (i = 0; i<8; i++)
		chainv[i] = statechainv[i];

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i)], h_CNS[(2 * i) + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 16], h_CNS[(2 * i) + 16 + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 32], h_CNS[(2 * i) + 32 + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 48], h_CNS[(2 * i) + 48 + 1]);

	for (i = 0; i<8; i++) {
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

	for (i = 0; i<8; i++)
		STEP(h_CNS[(2 * i) + 64], h_CNS[(2 * i) + 64 + 1]);

	for (i = 0; i<8; i++)
		statechainv[i + 32] = chainv[i];
}

/***************************************************/
__device__ __forceinline__
void Update512(uint32_t* statebuffer, uint32_t *statechainv, const uint32_t *const __restrict__ data)
{
	#pragma unroll
	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32((data[i]));
	rnd512(statebuffer, statechainv);

	#pragma unroll
	for(int i=0; i<8; i++)
		statebuffer[i] = cuda_swab32((data[i+8]));
	rnd512(statebuffer, statechainv);

	#pragma unroll
	for(int i=0; i<4; i++)
		statebuffer[i] = cuda_swab32((data[i+16]));
}


/***************************************************/
__device__ __forceinline__
void finalization512(uint32_t* statebuffer, uint32_t *statechainv, uint32_t *b)
{
	int i,j;

	statebuffer[4] = 0x80000000U;

	#pragma unroll 3
	for(int i=5; i<8; i++)
		statebuffer[i] = 0;
	rnd512(statebuffer, statechainv);

	/*---- blank round with m=0 ----*/
	#pragma unroll
	for(i=0; i<8; i++)
		statebuffer[i] =0;
	rnd512(statebuffer, statechainv);

	#pragma unroll
	for(i=0; i<8; i++) {
		b[i] = 0;
		#pragma unroll 5
		for(j=0; j<5; j++)
			b[i] ^= statechainv[i+8*j];
		b[i] = cuda_swab32((b[i]));
	}

	#pragma unroll
	for(i=0; i<8; i++)
		statebuffer[i]=0;
	rnd512(statebuffer, statechainv);

	#pragma unroll
	for(i=0; i<8; i++)
	{
		b[8+i] = 0;
		#pragma unroll 5
		for(j=0; j<5; j++)
			b[8+i] ^= statechainv[i+8*j];
		b[8+i] = cuda_swab32((b[8+i]));
	}
}


/***************************************************/
__global__
void qubit_luffa512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;
		union {
		uint64_t buf64[16];
		uint32_t buf32[32];
		} buff;

		#pragma unroll 8
		for (int i=8; i < 16; i++)
			buff.buf64[i] = c_PaddedMessage80[i];

		// die Nounce durch die thread-spezifische ersetzen
		buff.buf64[9] = REPLACE_HIDWORD(buff.buf64[9], cuda_swab32(nounce));

		uint32_t statebuffer[8], statechainv[40];

		#pragma unroll
		for (int i = 0; i<4; i++)
			statebuffer[i] = cuda_swab32(buff.buf32[i + 16]);

		#pragma unroll 4
		for (int i = 4; i<8; i++)
			statebuffer[i] = statebufferpre[i];

		#pragma unroll
		for (int i = 0; i<40; i++)
			statechainv[i] = statechainvpre[i];

		uint32_t *outHash = &outputHash[thread * 16];
		finalization512(statebuffer, statechainv, outHash);
	}
}

__host__
void qubit_luffa512_cpu_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_IV, h_IV, sizeof(h_IV), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_CNS, h_CNS, sizeof(h_CNS), 0, cudaMemcpyHostToDevice));
}

__host__
void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash,int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	size_t shared_size = 0;

	qubit_luffa512_gpu_hash_80 <<<grid, block, shared_size>>> (threads, startNounce, d_outputHash);
}

__host__
static void qubit_cpu_precalc(uint32_t* message)
{
	uint32_t statebuffer[8];
	uint32_t statechainv[40] =
	{
		0x6d251e69, 0x44b051e0, 0x4eaa6fb4, 0xdbf78465,
		0x6e292011, 0x90152df4, 0xee058139, 0xdef610bb,
		0xc3b44b95, 0xd9d2f256, 0x70eee9a0, 0xde099fa3,
		0x5d9b0557, 0x8fc944b3, 0xcf1ccf0e, 0x746cd581,
		0xf7efc89d, 0x5dba5781, 0x04016ce5, 0xad659c05,
		0x0306194f, 0x666d1836, 0x24aa230a, 0x8b264ae7,
		0x858075d5, 0x36d79cce, 0xe571f7d7, 0x204b1f67,
		0x35870c6a, 0x57e9e923, 0x14bcb808, 0x7cde72ce,
		0x6c68e9be, 0x5ec41e22, 0xc825b7c7, 0xaffb4363,
		0xf5df3999, 0x0fc688f1, 0xb07224cc, 0x03e86cea
	};

	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32(message[i]);
	rnd512_cpu(statebuffer, statechainv);

	for (int i = 0; i<8; i++)
		statebuffer[i] = cuda_swab32(message[i+8]);

	rnd512_cpu(statebuffer, statechainv);

	cudaMemcpyToSymbol(statebufferpre, statebuffer, sizeof(statebuffer), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(statechainvpre, statechainv, sizeof(statechainv), 0, cudaMemcpyHostToDevice);
}

__host__
void qubit_luffa512_cpu_setBlock_80(void *pdata)
{
	unsigned char PaddedMessage[128];

	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	PaddedMessage[80] = 0x80;
	PaddedMessage[111] = 1;
	PaddedMessage[126] = 0x02;
	PaddedMessage[127] = 0x80;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, sizeof(PaddedMessage), 0, cudaMemcpyHostToDevice));
	qubit_cpu_precalc((uint32_t*) PaddedMessage);
}
