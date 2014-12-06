#include <memory.h>

#include "cuda_helper.h"

#if 0
static __constant__ uint64_t SKEIN_IV512_256[8] = {
	0xCCD044A12FDB3E13, 0xE83590301A79A9EB,
	0x55AEA0614F816E6F, 0x2A2767A4AE9B94DB,
	0xEC06025E74DD7683, 0xE7A436CDC4746251,
	0xC36FBAF9393AD185, 0x3EEDBA1833EDFC13
};
#endif

static __constant__ uint2 vSKEIN_IV512_256[8] = {
	{ 0x2FDB3E13, 0xCCD044A1 },
	{ 0x1A79A9EB, 0xE8359030 },
	{ 0x4F816E6F, 0x55AEA061 },
	{ 0xAE9B94DB, 0x2A2767A4 },
	{ 0x74DD7683, 0xEC06025E },
	{ 0xC4746251, 0xE7A436CD },
	{ 0x393AD185, 0xC36FBAF9 },
	{ 0x33EDFC13, 0x3EEDBA18 }
};

static __constant__ int ROT256[8][4] =
{
	46,36, 19, 37,
	33,27, 14, 42,
	17,49, 36, 39,
	44, 9, 54, 56,
	39,30, 34, 24,
	13,50, 10, 17,
	25,29, 39, 43,
	8, 35, 56, 22,
};

static __constant__ uint2 skein_ks_parity = { 0xA9FC1A22,0x1BD11BDA};
static __constant__ uint2 t12[6] = {
	{ 0x20,	0 },
	{ 0,	0xf0000000 },
	{ 0x20,	0xf0000000 },
	{ 0x08,	0 },
	{ 0,	0xff000000 },
	{ 0x08,	0xff000000 }
};

#if 0
static __constant__ uint64_t t12_30[6] = {
	0x20,
	0xf000000000000000,
	0xf000000000000020,
	0x08,
	0xff00000000000000,
	0xff00000000000008
};
#endif

static __forceinline__ __device__
void Round512v35(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, int ROT)
{
	p0 += p1; p1 = ROL2(p1, ROT256[ROT][0]);  p1 ^= p0;
	p2 += p3; p3 = ROL2(p3, ROT256[ROT][1]);  p3 ^= p2;
	p4 += p5; p5 = ROL2(p5, ROT256[ROT][2]);  p5 ^= p4;
	p6 += p7; p7 = ROL2(p7, ROT256[ROT][3]);  p7 ^= p6;
}


static __forceinline__ __device__
void Round_8_512v35(uint2 *ks, uint2 *ts,
                    uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3,
                    uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, int R)
{
	Round512v35(p0, p1, p2, p3, p4, p5, p6, p7, 0);
	Round512v35(p2, p1, p4, p7, p6, p5, p0, p3, 1);
	Round512v35(p4, p1, p6, p3, p0, p5, p2, p7, 2);
	Round512v35(p6, p1, p0, p7, p2, p5, p4, p3, 3);
	p0 += ks[((R)+0) % 9];   /* inject the key schedule value */
	p1 += ks[((R)+1) % 9];
	p2 += ks[((R)+2) % 9];
	p3 += ks[((R)+3) % 9];
	p4 += ks[((R)+4) % 9];
	p5 += ks[((R)+5) % 9] + ts[((R)+0) % 3];
	p6 += ks[((R)+6) % 9] + ts[((R)+1) % 3];
	p7 += ks[((R)+7) % 9] + make_uint2((R),0);
	Round512v35(p0, p1, p2, p3, p4, p5, p6, p7, 4);
	Round512v35(p2, p1, p4, p7, p6, p5, p0, p3, 5);
	Round512v35(p4, p1, p6, p3, p0, p5, p2, p7, 6);
	Round512v35(p6, p1, p0, p7, p2, p5, p4, p3, 7);
	p0 += ks[((R)+1) % 9];   /* inject the key schedule value */
	p1 += ks[((R)+2) % 9];
	p2 += ks[((R)+3) % 9];
	p3 += ks[((R)+4) % 9];
	p4 += ks[((R)+5) % 9];
	p5 += ks[((R)+6) % 9] + ts[((R)+1) % 3];
	p6 += ks[((R)+7) % 9] + ts[((R)+2) % 3];
	p7 += ks[((R)+8) % 9] + make_uint2((R)+1, 0);
}


__global__ __launch_bounds__(256,3)
void skein256_gpu_hash_32(int threads, uint32_t startNounce, uint64_t *outputHash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h[9];
		uint2 t[3];
		uint2 dt0,dt1,dt2,dt3;
		uint2 p0, p1, p2, p3, p4, p5, p6, p7;

		h[8] = skein_ks_parity;
		for (int i = 0; i<8; i++) {
			h[i] = vSKEIN_IV512_256[i];
			h[8] ^= h[i];
		}

		t[0]=t12[0];
		t[1]=t12[1];
		t[2]=t12[2];

		LOHI(dt0.x,dt0.y,outputHash[thread]);
		LOHI(dt1.x,dt1.y,outputHash[threads+thread]);
		LOHI(dt2.x,dt2.y,outputHash[2*threads+thread]);
		LOHI(dt3.x,dt3.y,outputHash[3*threads+thread]);

		p0 = h[0] + dt0;
		p1 = h[1] + dt1;
		p2 = h[2] + dt2;
		p3 = h[3] + dt3;
		p4 = h[4];
		p5 = h[5] + t[0];
		p6 = h[6] + t[1];
		p7 = h[7];

		#pragma unroll
		for (int i = 1; i<19; i+=2) {
			Round_8_512v35(h,t,p0,p1,p2,p3,p4,p5,p6,p7,i);
		}

		p0 ^= dt0;
		p1 ^= dt1;
		p2 ^= dt2;
		p3 ^= dt3;

		h[0] = p0;
		h[1] = p1;
		h[2] = p2;
		h[3] = p3;
		h[4] = p4;
		h[5] = p5;
		h[6] = p6;
		h[7] = p7;
		h[8] = skein_ks_parity;

		#pragma unroll 8
		for (int i = 0; i<8; i++) {
			h[8] ^= h[i];
		}

		t[0] = t12[3];
		t[1] = t12[4];
		t[2] = t12[5];
		p5 += t[0];  //p5 already equal h[5]
		p6 += t[1];

		#pragma unroll
		for (int i = 1; i<19; i+=2) {
			Round_8_512v35(h, t, p0, p1, p2, p3, p4, p5, p6, p7, i);
		}

		outputHash[thread]           = devectorize(p0);
		outputHash[threads+thread]   = devectorize(p1);
		outputHash[2*threads+thread] = devectorize(p2);
		outputHash[3*threads+thread] = devectorize(p3);
	}
}

__host__
void skein256_cpu_init(int thr_id, int threads)
{
	//empty
}

__host__
void skein256_cpu_hash_32(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	skein256_gpu_hash_32<<<grid, block>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}

