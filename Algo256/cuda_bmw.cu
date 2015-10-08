/**
 * BMW-256 CUDA Implementation - tpruvot 2015
 *
 * Not optimal but close to the sph version and easier to adapt.
 */

#include <stdio.h>
#include <memory.h>

#define SPH_64 1
#define USE_MIDSTATE

extern "C" {
#include "sph/sph_bmw.h"
}

#include "cuda_helper.h"

__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

#ifndef USE_MIDSTATE
__constant__ static sph_u32 IV256[16] = {
	0x40414243, 0x44454647, 0x48494A4B, 0x4C4D4E4F,
	0x50515253, 0x54555657, 0x58595A5B, 0x5C5D5E5F,
	0x60616263, 0x64656667, 0x68696A6B, 0x6C6D6E6F,
	0x70717273, 0x74757677, 0x78797A7B, 0x7C7D7E7F
};
#endif

__constant__ static sph_u32 final_s[16] = {
	0xaaaaaaa0, 0xaaaaaaa1, 0xaaaaaaa2, 0xaaaaaaa3,
	0xaaaaaaa4, 0xaaaaaaa5, 0xaaaaaaa6, 0xaaaaaaa7,
	0xaaaaaaa8, 0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaab,
	0xaaaaaaac, 0xaaaaaaad, 0xaaaaaaae, 0xaaaaaaaf
};

static sph_bmw_small_context* d_midstate[MAX_GPUS];

#define I16_16    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
#define I16_17    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16
#define I16_18    2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17
#define I16_19    3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18
#define I16_20    4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#define I16_21    5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#define I16_22    6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
#define I16_23    7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
#define I16_24    8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
#define I16_25    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
#define I16_26   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
#define I16_27   11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
#define I16_28   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
#define I16_29   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
#define I16_30   14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
#define I16_31   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30

//#define M16_16    0,  1,  3,  4,  7, 10, 11
//#define M16_17    1,  2,  4,  5,  8, 11, 12
#define M16_18    2,  3,  5,  6,  9, 12, 13
#define M16_19    3,  4,  6,  7, 10, 13, 14
#define M16_20    4,  5,  7,  8, 11, 14, 15
#define M16_21    5,  6,  8,  9, 12, 15, 16
#define M16_22    6,  7,  9, 10, 13,  0,  1
#define M16_23    7,  8, 10, 11, 14,  1,  2
#define M16_24    8,  9, 11, 12, 15,  2,  3
#define M16_25    9, 10, 12, 13,  0,  3,  4
#define M16_26   10, 11, 13, 14,  1,  4,  5
#define M16_27   11, 12, 14, 15,  2,  5,  6
#define M16_28   12, 13, 15, 16,  3,  6,  7
#define M16_29   13, 14,  0,  1,  4,  7,  8
#define M16_30   14, 15,  1,  2,  5,  8,  9
#define M16_31   15, 16,  2,  3,  6,  9, 10

#define ss0(x)    (((x) >> 1) ^ ((x) << 3) ^ ROTL32(x,  4) ^ ROTL32(x, 19))
#define ss1(x)    (((x) >> 1) ^ ((x) << 2) ^ ROTL32(x,  8) ^ ROTL32(x, 23))
#define ss2(x)    (((x) >> 2) ^ ((x) << 1) ^ ROTL32(x, 12) ^ ROTL32(x, 25))
#define ss3(x)    (((x) >> 2) ^ ((x) << 2) ^ ROTL32(x, 15) ^ ROTL32(x, 29))
#define ss4(x)    (((x) >> 1) ^ (x))
#define ss5(x)    (((x) >> 2) ^ (x))

#define rs1(x)    ROTL32(x,  3)
#define rs2(x)    ROTL32(x,  7)
#define rs3(x)    ROTL32(x, 13)
#define rs4(x)    ROTL32(x, 16)
#define rs5(x)    ROTL32(x, 19)
#define rs6(x)    ROTL32(x, 23)
#define rs7(x)    ROTL32(x, 27)

#define MAKE_W(tt, i0, op01, i1, op12, i2, op23, i3, op34, i4) \
	tt((data[i0] ^ h[i0]) op01 (data[i1] ^ h[i1]) op12 (data[i2] ^ h[i2]) op23 (data[i3] ^ h[i3]) op34 (data[i4] ^ h[i4]))
//#define Ws0    MAKE_W(SPH_T32,  5, -,  7, +, 10, +, 13, +, 14)
//#define Ws1    MAKE_W(SPH_T32,  6, -,  8, +, 11, +, 14, -, 15)
//#define Ws2    MAKE_W(SPH_T32,  0, +,  7, +,  9, -, 12, +, 15)
//#define Ws3    MAKE_W(SPH_T32,  0, -,  1, +,  8, -, 10, +, 13)
//#define Ws4    MAKE_W(SPH_T32,  1, +,  2, +,  9, -, 11, -, 14)
//#define Ws5    MAKE_W(SPH_T32,  3, -,  2, +, 10, -, 12, +, 15)
//#define Ws6    MAKE_W(SPH_T32,  4, -,  0, -,  3, -, 11, +, 13)
//#define Ws7    MAKE_W(SPH_T32,  1, -,  4, -,  5, -, 12, -, 14)
//#define Ws8    MAKE_W(SPH_T32,  2, -,  5, -,  6, +, 13, -, 15)
//#define Ws9    MAKE_W(SPH_T32,  0, -,  3, +,  6, -,  7, +, 14)
//#define Ws10   MAKE_W(SPH_T32,  8, -,  1, -,  4, -,  7, +, 15)
//#define Ws11   MAKE_W(SPH_T32,  8, -,  0, -,  2, -,  5, +,  9)
//#define Ws12   MAKE_W(SPH_T32,  1, +,  3, -,  6, -,  9, +, 10)
//#define Ws13   MAKE_W(SPH_T32,  2, +,  4, +,  7, +, 10, +, 11)
//#define Ws14   MAKE_W(SPH_T32,  3, -,  5, +,  8, -, 11, -, 12)
//#define Ws15   MAKE_W(SPH_T32, 12, -,  4, -,  6, -,  9, +, 13)

__device__
static void gpu_compress_small(const sph_u32 *data, const sph_u32 h[16], sph_u32 dh[16])
{
		// FOLD MAKE_Qas;

		sph_u32 dx[16];
		for (int i=0; i<16; i++)
			dx[i] = data[i] ^ h[i];

		sph_u32 qt[32];
		qt[ 0] = dx[ 5] - dx[7] + dx[10] + dx[13] + dx[14]; // Ws0
		qt[ 1] = dx[ 6] - dx[8] + dx[11] + dx[14] - dx[15]; // Ws1
		qt[ 2] = dx[ 0] + dx[7] + dx[ 9] - dx[12] + dx[15]; // Ws2
		qt[ 3] = dx[ 0] - dx[1] + dx[ 8] - dx[10] + dx[13]; // Ws3
		qt[ 4] = dx[ 1] + dx[2] + dx[ 9] - dx[11] - dx[14]; // Ws4;
		qt[ 5] = dx[ 3] - dx[2] + dx[10] - dx[12] + dx[15]; // Ws5;
		qt[ 6] = dx[ 4] - dx[0] - dx[ 3] - dx[11] + dx[13]; // Ws6;
		qt[ 7] = dx[ 1] - dx[4] - dx[ 5] - dx[12] - dx[14]; // Ws7;
		qt[ 8] = dx[ 2] - dx[5] - dx[ 6] + dx[13] - dx[15]; // Ws8;
		qt[ 9] = dx[ 0] - dx[3] + dx[ 6] - dx[ 7] + dx[14]; // Ws9;
		qt[10] = dx[ 8] - dx[1] - dx[ 4] - dx[ 7] + dx[15]; // Ws10;
		qt[11] = dx[ 8] - dx[0] - dx[ 2] - dx[ 5] + dx[ 9]; // Ws11;
		qt[12] = dx[ 1] + dx[3] - dx[ 6] - dx[ 9] + dx[10]; // Ws12;
		qt[13] = dx[ 2] + dx[4] + dx[ 7] + dx[10] + dx[11]; // Ws13;
		qt[14] = dx[ 3] - dx[5] + dx[ 8] - dx[11] - dx[12]; // Ws14;
		qt[15] = dx[12] - dx[4] - dx[ 6] - dx[ 9] + dx[13]; // Ws15;

		qt[ 0] = ss0(qt[ 0]) + h[ 1];
		qt[ 1] = ss1(qt[ 1]) + h[ 2];
		qt[ 2] = ss2(qt[ 2]) + h[ 3];
		qt[ 3] = ss3(qt[ 3]) + h[ 4];
		qt[ 4] = ss4(qt[ 4]) + h[ 5];

		qt[ 5] = ss0(qt[ 5]) + h[ 6];
		qt[ 6] = ss1(qt[ 6]) + h[ 7];
		qt[ 7] = ss2(qt[ 7]) + h[ 8];
		qt[ 8] = ss3(qt[ 8]) + h[ 9];
		qt[ 9] = ss4(qt[ 9]) + h[10];

		qt[10] = ss0(qt[10]) + h[11];
		qt[11] = ss1(qt[11]) + h[12];
		qt[12] = ss2(qt[12]) + h[13];
		qt[13] = ss3(qt[13]) + h[14];
		qt[14] = ss4(qt[14]) + h[15];

		qt[15] = ss0(qt[15]) + h[ 0];

		//MAKE_Qbs;
		#define Ks(j)   ((sph_u32)(0x05555555UL * j))
		#define Qs(j)   (qt[j])

		#define expand1s_in(i16, \
				i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, \
				i0m, i1m, i3m, i4m, i7m, i10m, i11m) \
			(ss1(qt[i0]) + ss2(qt[i1]) + ss3(qt[i2]) + ss0(qt[i3]) + ss1(qt[i4]) + ss2(qt[i5]) + ss3(qt[i6]) + ss0(qt[i7]) \
				+ ss1(qt[i8]) + ss2(qt[i9]) + ss3(qt[i10]) + ss0(qt[i11]) + ss1(qt[i12]) + ss2(qt[i13]) + ss3(qt[i14]) + ss0(qt[i15]) \
				+ ((ROTL32(data[i0m], i1m) + ROTL32(data[i3m], i4m)  - ROTL32(data[i10m], i11m) + Ks(i16)) ^ h[i7m]))

		qt[16] = expand1s_in(16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0, 1, 3, 4, 7, 10, 11);
		qt[17] = expand1s_in(17,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  1, 2, 4, 5, 8, 11, 12);

		#define expand2s_inner(qf, i16, \
				i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, \
				i0m, i1m, i3m, i4m, i7m, i10m, i11m) \
			(qf(i0) + rs1(qf(i1)) + qf(i2) + rs2(qf(i3)) \
				+ qf(i4) + rs3(qf(i5)) + qf(i6) + rs4(qf(i7)) + qf(i8) + rs5(qf(i9)) + qf(i10) + rs6(qf(i11)) \
				+ qf(i12) + rs7(qf(i13)) + ss4(qf(i14)) + ss5(qf(i15)) \
				+ ((ROTL32(data[i0m], i1m) + ROTL32(data[i3m], i4m) - ROTL32(data[i10m], i11m) + Ks(i16)) ^ h[i7m]))

#ifdef _MSC_VER
		#define LPAR   (
		#define expand2s(i16) \
			expand2s_(Qs, i16, I16_ ## i16, M16_ ## i16)
		#define expand2s_(qf, i16, ix, iy) \
			expand2s_inner LPAR qf, i16, ix, iy)
#else
		#define expand2s_(i16, ix, iy) \
			expand2s_inner(Qs, i16, ix, iy)
		#define expand2s(i16) \
			expand2s_(i16, I16_ ## i16, M16_ ## i16)
#endif

		qt[18] = expand2s(18);
		qt[19] = expand2s(19);
		qt[20] = expand2s(20);
		qt[21] = expand2s(21);
		qt[22] = expand2s(22);
		qt[23] = expand2s(23);
		qt[24] = expand2s(24);
		qt[25] = expand2s(25);
		qt[26] = expand2s(26);
		qt[27] = expand2s(27);
		qt[28] = expand2s(28);
		qt[29] = expand2s(29);
		qt[30] = expand2s(30);
		qt[31] = expand2s(31);

		sph_u32 xl, xh;
		xl = Qs(16) ^ Qs(17) ^ Qs(18) ^ Qs(19) ^ Qs(20) ^ Qs(21) ^ Qs(22) ^ Qs(23);

		xh = xl ^ Qs(24) ^ Qs(25) ^ Qs(26) ^ Qs(27)	^ Qs(28) ^ Qs(29) ^ Qs(30) ^ Qs(31);

		dh[ 0] = ((xh <<  5) ^ (Qs(16) >>  5) ^ data[ 0]) + (xl ^ Qs(24) ^ Qs(0));
		dh[ 1] = ((xh >>  7) ^ (Qs(17) <<  8) ^ data[ 1]) + (xl ^ Qs(25) ^ Qs(1));
		dh[ 2] = ((xh >>  5) ^ (Qs(18) <<  5) ^ data[ 2]) + (xl ^ Qs(26) ^ Qs(2));
		dh[ 3] = ((xh >>  1) ^ (Qs(19) <<  5) ^ data[ 3]) + (xl ^ Qs(27) ^ Qs(3));
		dh[ 4] = ((xh >>  3) ^ (Qs(20) <<  0) ^ data[ 4]) + (xl ^ Qs(28) ^ Qs(4));
		dh[ 5] = ((xh <<  6) ^ (Qs(21) >>  6) ^ data[ 5]) + (xl ^ Qs(29) ^ Qs(5));
		dh[ 6] = ((xh >>  4) ^ (Qs(22) <<  6) ^ data[ 6]) + (xl ^ Qs(30) ^ Qs(6));
		dh[ 7] = ((xh >> 11) ^ (Qs(23) <<  2) ^ data[ 7]) + (xl ^ Qs(31) ^ Qs(7));

		dh[ 8] = ROTL32(dh[4],  9) + (xh ^ Qs(24) ^ data[ 8]) + ((xl << 8) ^ Qs(23) ^ Qs( 8));
		dh[ 9] = ROTL32(dh[5], 10) + (xh ^ Qs(25) ^ data[ 9]) + ((xl >> 6) ^ Qs(16) ^ Qs( 9));
		dh[10] = ROTL32(dh[6], 11) + (xh ^ Qs(26) ^ data[10]) + ((xl << 6) ^ Qs(17) ^ Qs(10));
		dh[11] = ROTL32(dh[7], 12) + (xh ^ Qs(27) ^ data[11]) + ((xl << 4) ^ Qs(18) ^ Qs(11));
		dh[12] = ROTL32(dh[0], 13) + (xh ^ Qs(28) ^ data[12]) + ((xl >> 3) ^ Qs(19) ^ Qs(12));
		dh[13] = ROTL32(dh[1], 14) + (xh ^ Qs(29) ^ data[13]) + ((xl >> 4) ^ Qs(20) ^ Qs(13));
		dh[14] = ROTL32(dh[2], 15) + (xh ^ Qs(30) ^ data[14]) + ((xl >> 7) ^ Qs(21) ^ Qs(14));
		dh[15] = ROTL32(dh[3], 16) + (xh ^ Qs(31) ^ data[15]) + ((xl >> 2) ^ Qs(22) ^ Qs(15));
}

#ifndef USE_MIDSTATE

__device__
static void gpu_bmw256_init(sph_bmw_small_context *sc)
{
	memcpy(sc->H, IV256, sizeof sc->H);
	sc->ptr = 0;
	sc->bit_count = 0;
}

__device__
static void gpu_bmw256(sph_bmw_small_context *sc, const void *data, size_t len)
{
	sph_u32 htmp[16];
	sph_u32 *h1, *h2;
	unsigned char *buf = sc->buf;
	size_t ptr = sc->ptr;

	sc->bit_count += (sph_u64)len << 3;

	h1 = sc->H;
	h2 = htmp;
	while (len > 0) {
		size_t clen;

		clen = (sizeof sc->buf) - ptr;
		if (clen > len)
			clen = len;
		memcpy(buf + ptr, data, clen);
		data = (const unsigned char *)data + clen;
		len -= clen;
		ptr += clen;
		if (ptr == sizeof sc->buf) {
			sph_u32 *ht;

			gpu_compress_small((sph_u32 *) buf, h1, h2);
			ht = h1;
			h1 = h2;
			h2 = ht;
			ptr = 0;
		}
	}
	sc->ptr = ptr;
	if (h1 != sc->H)
		memcpy(sc->H, h1, sizeof sc->H);
}

#endif

#define sph_enc64le(ptr, x) \
	*((uint64_t*)(ptr)) = x
#define sph_enc64le_aligned sph_enc64le

__device__
static void gpu_bmw256_close(sph_bmw_small_context *sc, uint2 *out)
{
	unsigned char *buf = sc->buf;
	size_t ptr = sc->ptr;

	buf[ptr ++] = 0x80;
	sph_u32 *h = sc->H;

	sph_u32 h1[16];
	if (ptr > (sizeof sc->buf) - 8) {
		memset(buf + ptr, 0, (sizeof sc->buf) - ptr);
		gpu_compress_small((sph_u32 *) buf, h, h1);
		ptr = 0;
		h = h1;
	}
	memset(buf + ptr, 0, sizeof(sc->buf) - 8 - ptr);

	sph_enc64le_aligned(buf + sizeof(sc->buf) - 8, SPH_T64(sc->bit_count));

	sph_u32 h2[16];
	gpu_compress_small((sph_u32 *) buf, h, h2);
	gpu_compress_small(h2, final_s, h1);

	uint64_t* h64 = (uint64_t*) (&h1[8]);
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		out[i] = vectorize(h64[i]);
	}
}

__global__ /* __launch_bounds__(256, 3) */
void bmw256_gpu_hash_80(uint32_t threads, uint32_t startNonce, uint64_t *g_hash, sph_bmw256_context *d_midstate, int swap)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nonce = startNonce + thread;
		nonce = swap ? cuda_swab32(nonce): nonce;

#ifndef USE_MIDSTATE
		uint2 hash[10];
		#pragma unroll
		for(int i=0;i<9;i++)
			hash[i] = vectorize(c_PaddedMessage80[i]);
		hash[9] = make_uint2(c_PaddedMessage80[9], nonce);

		sph_bmw256_context ctx;
		gpu_bmw256_init(&ctx);
		gpu_bmw256(&ctx, (void*) hash, 80);
#else
		sph_bmw256_context ctx;
		ctx.ptr = 16; ctx.bit_count = 640;
		uint2 *buf = (uint2 *) ctx.buf;
		buf[0] = vectorize(c_PaddedMessage80[8]);
		buf[1] = make_uint2(c_PaddedMessage80[9], nonce);
		#pragma unroll
		for(int i=0;i<16;i++)
			ctx.H[i] = d_midstate->H[i];
#endif
		gpu_bmw256_close(&ctx, (uint2*) &g_hash[thread << 2]);
	}
}

__host__
void bmw256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash, int swap)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	bmw256_gpu_hash_80<<<grid, block>>>(threads, startNonce, (uint64_t*)d_outputHash, d_midstate[thr_id], swap);
}

__host__
void bmw256_setBlock_80(int thr_id, void *pdata)
{
	uint64_t PaddedMessage[16];
	memcpy(PaddedMessage, pdata, 80);
	memset(&PaddedMessage[10], 0, 48);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, sizeof(PaddedMessage), 0, cudaMemcpyHostToDevice));

	sph_bmw256_context ctx;
	sph_bmw256_init(&ctx);
	sph_bmw256(&ctx, (void*) PaddedMessage, 80);
	CUDA_SAFE_CALL(cudaMemcpy(d_midstate[thr_id], &ctx, sizeof(sph_bmw256_context), cudaMemcpyHostToDevice));
}

__host__
void bmw256_midstate_init(int thr_id, uint32_t threads)
{
	cudaMalloc(&d_midstate[thr_id], sizeof(sph_bmw256_context));
}

__host__
void bmw256_midstate_free(int thr_id)
{
	cudaFree(d_midstate[thr_id]);
}
