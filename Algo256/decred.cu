/**
 * Blake-256 Decred 180-Bytes input Cuda Kernel
 *
 * Tanguy Pruvot, Alexis Provos - Feb/Sep 2016
 */

#include <stdint.h>
#include <memory.h>
#include <miner.h>

extern "C" {
#include <sph/sph_blake.h>
}

/* threads per block */
#define TPB 640

/* max count of found nonces in one call (like sgminer) */
#define MAX_RESULTS 4

/* hash by cpu with blake 256 */
extern "C" void decred_hash(void *output, const void *input)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 180);
	sph_blake256_close(&ctx, output);
}

#include <cuda_helper.h>

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#define atomicInc(p, max) (*p)++
#endif

__constant__ uint32_t _ALIGN(16) c_h[2];
__constant__ uint32_t _ALIGN(16) c_data[32];
__constant__ uint32_t _ALIGN(16) c_xors[215];

/* Buffers of candidate nonce(s) */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

#define ROR8(a)  __byte_perm(a, 0, 0x0321)
#define ROL16(a) __byte_perm(a, 0, 0x1032)

/* macro bodies */
#define pxorGS(a,b,c,d) { \
	v[a]+= c_xors[i++] + v[b]; \
	v[d] = ROL16(v[d] ^ v[a]); \
	v[c]+= v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
	v[a]+= c_xors[i++] + v[b]; \
	v[d] = ROR8(v[d] ^ v[a]); \
	v[c]+= v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
}

#define pxorGS2(a,b,c,d, a1,b1,c1,d1) {\
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);           v[d1] = ROL16(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);      v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);            v[d1] = ROR8(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);       v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
}

#define pxory1GS2(a,b,c,d, a1,b1,c1,d1) { \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);           v[d1] = ROL16(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);      v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= (c_xors[i++]^nonce) + v[b1]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);            v[d1] = ROR8(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);       v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
}

#define pxory0GS2(a,b,c,d, a1,b1,c1,d1) { \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);           v[d1] = ROL16(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);      v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[ a]+= (c_xors[i++]^nonce) + v[ b];    v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);            v[d1] = ROR8(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);       v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
}

#define pxorx1GS2(a,b,c,d, a1,b1,c1,d1) { \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= (c_xors[i++]^nonce) + v[b1]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);           v[d1] = ROL16(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);      v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);            v[d1] = ROR8(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);       v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
}

#define pxorx0GS2(a,b,c,d, a1,b1,c1,d1) { \
	v[ a]+= (c_xors[i++]^nonce) + v[ b];    v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROL16(v[ d] ^ v[ a]);           v[d1] = ROL16(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 12);      v[b1] = ROTR32(v[b1] ^ v[c1], 12); \
	v[ a]+= c_xors[i++] + v[ b];            v[a1]+= c_xors[i++] + v[b1]; \
	v[ d] = ROR8(v[ d] ^ v[ a]);            v[d1] = ROR8(v[d1] ^ v[a1]); \
	v[ c]+= v[ d];                          v[c1]+= v[d1]; \
	v[ b] = ROTR32(v[ b] ^ v[ c], 7);       v[b1] = ROTR32(v[b1] ^ v[c1], 7); \
}

__global__ __launch_bounds__(TPB,1)
void decred_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint32_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		uint32_t v[16];
		#pragma unroll
		for(int i=0; i<16; i+=4) {
			*(uint4*)&v[i] = *(uint4*)&c_data[i];
		}

		const uint32_t nonce = startNonce + thread;
		v[ 1]+= (nonce ^ 0x13198A2E);
		v[13] = ROR8(v[13] ^ v[1]);
		v[ 9]+= v[13];
		v[ 5] = ROTR32(v[5] ^ v[9], 7);

		int i = 0;
		v[ 1]+= c_xors[i++];// + v[ 6];
		v[ 0]+= v[5];
		v[12] = ROL16(v[12] ^ v[ 1]);         v[15] = ROL16(v[15] ^ v[ 0]);
		v[11]+= v[12];                        v[10]+= v[15];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 12);    v[ 5] = ROTR32(v[5] ^ v[10], 12);
		v[ 1]+= c_xors[i++] + v[ 6];          v[ 0]+= c_xors[i++] + v[ 5];
		v[12] = ROR8(v[12] ^ v[ 1]);          v[15] = ROR8(v[15] ^ v[ 0]);
		v[11]+= v[12];                        v[10]+= v[15];
		v[ 6] = ROTR32(v[ 6] ^ v[11], 7);     v[ 5] = ROTR32(v[ 5] ^ v[10], 7);

		pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxory1GS2( 2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorx1GS2( 0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorx1GS2( 0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorx1GS2( 2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxory1GS2( 2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxory1GS2( 0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorx1GS2( 2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxory0GS2( 2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorx0GS2( 2, 7, 8, 13, 3, 4, 9, 14);
		pxory1GS2( 0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxory1GS2( 2, 7, 8, 13, 3, 4, 9, 14);
		pxorGS2(   0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorx1GS2( 0, 5, 10, 15, 1, 6, 11, 12); pxorGS2(   2, 7, 8, 13, 3, 4, 9, 14);
		pxorx1GS2( 0, 4, 8, 12, 1, 5, 9, 13); pxorGS2(   2, 6, 10, 14, 3, 7, 11, 15); pxorGS2(   0, 5, 10, 15, 1, 6, 11, 12); pxorGS(    2, 7, 8, 13);

		if ((c_h[1]^v[15]) == v[7]) {
			v[ 3] += c_xors[i++] + v[4];
			v[14] = ROL16(v[14] ^ v[3]);
			v[ 9] += v[14];
			v[ 4] = ROTR32(v[4] ^ v[9], 12);
			v[ 3] += c_xors[i++] + v[4];
			v[14] = ROR8(v[14] ^ v[3]);
			if(cuda_swab32((c_h[0]^v[6]^v[14])) <= highTarget) {
				uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX)+1;
				resNonce[pos] = nonce;
				return;
			}
		}
	}
}

__host__
void decred_cpu_setBlock_52(const uint32_t *input)
{
/*
	Precompute everything possible and pass it on constant memory
*/
	const uint32_t z[16] = {
		0x243F6A88U, 0x85A308D3U, 0x13198A2EU, 0x03707344U,
		0xA4093822U, 0x299F31D0U, 0x082EFA98U, 0xEC4E6C89U,
		0x452821E6U, 0x38D01377U, 0xBE5466CFU, 0x34E90C6CU,
		0xC0AC29B7U, 0xC97C50DDU, 0x3F84D5B5U, 0xB5470917U
	};

	int i=0;
	uint32_t _ALIGN(64) preXOR[215];
	uint32_t _ALIGN(64)   data[16];
	uint32_t _ALIGN(64)      m[16];
	uint32_t _ALIGN(64)      h[ 2];

	sph_blake256_context ctx;
	sph_blake256_set_rounds(14);
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 128);

	data[ 0] = ctx.H[0];
	data[ 1] = ctx.H[1];
	data[ 2] = ctx.H[2];
	data[ 3] = ctx.H[3];
	data[ 4] = ctx.H[4];
	data[ 5] = ctx.H[5];
	data[ 8] = ctx.H[6];

	data[12] = swab32(input[35]);
	data[13] = ctx.H[7];

	// pre swab32
	m[ 0] = swab32(input[32]);	m[ 1] = swab32(input[33]);
	m[ 2] = swab32(input[34]);	m[ 3] = 0;
	m[ 4] = swab32(input[36]);	m[ 5] = swab32(input[37]);
	m[ 6] = swab32(input[38]);	m[ 7] = swab32(input[39]);
	m[ 8] = swab32(input[40]);	m[ 9] = swab32(input[41]);
	m[10] = swab32(input[42]);	m[11] = swab32(input[43]);
	m[12] = swab32(input[44]);	m[13] = 0x80000001;
	m[14] = 0;
	m[15] = 0x000005a0;

	h[ 0] = data[ 8];
	h[ 1] = data[13];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_h,h, 8, 0, cudaMemcpyHostToDevice));

	data[ 0]+= (m[ 0] ^ z[1]) + data[ 4];
	data[12]  = SPH_ROTR32(z[4] ^ SPH_C32(0x5A0) ^ data[ 0], 16);

	data[ 8] = z[0]+data[12];
	data[ 4] = SPH_ROTR32(data[ 4] ^ data[ 8], 12);
	data[ 0]+= (m[ 1] ^ z[0]) + data[ 4];
	data[12] = SPH_ROTR32(data[12] ^ data[ 0],8);
	data[ 8]+= data[12];
	data[ 4] = SPH_ROTR32(data[ 4] ^ data[ 8], 7);

	data[ 1]+= (m[ 2] ^ z[3]) + data[ 5];
	data[13] = SPH_ROTR32((z[5] ^ SPH_C32(0x5A0)) ^ data[ 1], 16);
	data[ 9] = z[1]+data[13];
	data[ 5] = SPH_ROTR32(data[ 5] ^ data[ 9], 12);
	data[ 1]+= data[ 5]; //+nonce ^ ...

	data[ 2]+= (m[ 4] ^ z[5]) + h[ 0];
	data[14] = SPH_ROTR32(z[6] ^ data[ 2],16);
	data[10] = z[2] + data[14];
	data[ 6] = SPH_ROTR32(h[ 0] ^ data[10], 12);
	data[ 2]+= (m[ 5] ^ z[4]) + data[ 6];
	data[14] = SPH_ROTR32(data[14] ^ data[ 2], 8);
	data[10]+= data[14];
	data[ 6] = SPH_ROTR32(data[ 6] ^ data[10], 7);

	data[ 3]+= (m[ 6] ^ z[7]) + h[ 1];
	data[15] = SPH_ROTR32(z[7] ^ data[ 3],16);
	data[11] = z[3] + data[15];
	data[ 7] = SPH_ROTR32(h[ 1] ^ data[11], 12);
	data[ 3]+= (m[ 7] ^ z[6]) + data[ 7];
	data[15] = SPH_ROTR32(data[15] ^ data[ 3],8);
	data[11]+= data[15];
	data[ 7] = SPH_ROTR32(data[11] ^ data[ 7], 7);
	data[ 0]+= m[ 8] ^ z[9];

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, data, 64, 0, cudaMemcpyHostToDevice));

#define precalcXORGS(x,y) { \
	preXOR[i++]= (m[x] ^ z[y]); \
	preXOR[i++]= (m[y] ^ z[x]); \
}
#define precalcXORGS2(x,y,x1,y1){\
	preXOR[i++] = (m[ x] ^ z[ y]);\
	preXOR[i++] = (m[x1] ^ z[y1]);\
	preXOR[i++] = (m[ y] ^ z[ x]);\
	preXOR[i++] = (m[y1] ^ z[x1]);\
}
	precalcXORGS(10,11);
	preXOR[ 0]+=data[ 6];
	preXOR[i++] = (m[9] ^ z[8]);
	precalcXORGS2(12,13,14,15);
	precalcXORGS2(14,10, 4, 8);
	precalcXORGS2( 9,15,13, 6);
	precalcXORGS2( 1,12, 0, 2);
	precalcXORGS2(11, 7, 5, 3);
	precalcXORGS2(11, 8,12, 0);
	precalcXORGS2( 5, 2,15,13);
	precalcXORGS2(10,14, 3, 6);
	precalcXORGS2( 7, 1, 9, 4);
	precalcXORGS2( 7, 9, 3, 1);
	precalcXORGS2(13,12,11,14);
	precalcXORGS2( 2, 6, 5,10);
	precalcXORGS2( 4, 0,15, 8);
	precalcXORGS2( 9, 0, 5, 7);
	precalcXORGS2( 2, 4,10,15);
	precalcXORGS2(14, 1,11,12);
	precalcXORGS2( 6, 8, 3,13);
	precalcXORGS2( 2,12, 6,10);
	precalcXORGS2( 0,11, 8, 3);
	precalcXORGS2( 4,13, 7, 5);
	precalcXORGS2(15,14, 1, 9);
	precalcXORGS2(12, 5, 1,15);
	precalcXORGS2(14,13, 4,10);
	precalcXORGS2( 0, 7, 6, 3);
	precalcXORGS2( 9, 2, 8,11);
	precalcXORGS2(13,11, 7,14);
	precalcXORGS2(12, 1, 3, 9);
	precalcXORGS2( 5, 0,15, 4);
	precalcXORGS2( 8, 6, 2,10);
	precalcXORGS2( 6,15,14, 9);
	precalcXORGS2(11, 3, 0, 8);
	precalcXORGS2(12, 2,13, 7);
	precalcXORGS2( 1, 4,10, 5);
	precalcXORGS2(10, 2, 8, 4);
	precalcXORGS2( 7, 6, 1, 5);
	precalcXORGS2(15,11, 9,14);
	precalcXORGS2( 3,12,13, 0);
	precalcXORGS2( 0, 1, 2, 3);
	precalcXORGS2( 4, 5, 6, 7);
	precalcXORGS2( 8, 9,10,11);
	precalcXORGS2(12,13,14,15);
	precalcXORGS2(14,10, 4, 8);
	precalcXORGS2( 9,15,13, 6);
	precalcXORGS2( 1,12, 0, 2);
	precalcXORGS2(11, 7, 5, 3);
	precalcXORGS2(11, 8,12, 0);
	precalcXORGS2( 5, 2,15,13);
	precalcXORGS2(10,14, 3, 6);
	precalcXORGS2( 7, 1, 9, 4);
	precalcXORGS2( 7, 9, 3, 1);
	precalcXORGS2(13,12,11,14);
	precalcXORGS2( 2, 6, 5,10);
	precalcXORGS( 4, 0);
	precalcXORGS(15, 8);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_xors, preXOR, 215*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

/* ############################################################################################################################### */

static bool init[MAX_GPUS] = { 0 };

// nonce position is different in decred
#define DCR_NONCE_OFT32 35

extern "C" int scanhash_decred(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[48];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = &pdata[DCR_NONCE_OFT32];

	const uint32_t first_nonce = *pnonce;
	const uint32_t targetHigh = opt_benchmark ? 0x1ULL : ptarget[6];

	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 29 : 25;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	const dim3 grid((throughput + TPB-1)/(TPB));
	const dim3 block(TPB);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], MAX_RESULTS*sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], MAX_RESULTS*sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}
	memcpy(endiandata, pdata, 180);

	decred_cpu_setBlock_52(endiandata);
	cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

	do {
		uint32_t* resNonces = h_resNonce[thr_id];

		if (resNonces[0]) cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));

		// GPU HASH
		decred_gpu_hash_nonce <<<grid, block>>> (throughput, (*pnonce), d_resNonce[thr_id], targetHigh);

		*hashes_done = (*pnonce) - first_nonce + throughput;

		// first cell contains the valid nonces count
		cudaMemcpy(resNonces, d_resNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (resNonces[0])
		{
			uint32_t _ALIGN(64) vhash[8];

			cudaMemcpy(resNonces, d_resNonce[thr_id], (resNonces[0]+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost);

			be32enc(&endiandata[DCR_NONCE_OFT32], resNonces[1]);
			decred_hash(vhash, endiandata);
			if (vhash[6] <= ptarget[6] && fulltest(vhash, ptarget))
			{
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[0] = swab32(resNonces[1]);
				*pnonce = work->nonces[0];

				// search for another nonce
				for(uint32_t n=2; n <= resNonces[0]; n++)
				{
					be32enc(&endiandata[DCR_NONCE_OFT32], resNonces[n]);
					decred_hash(vhash, endiandata);
					if (vhash[6] <= ptarget[6] && fulltest(vhash, ptarget)) {
						work->nonces[1] = swab32(resNonces[n]);
						if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]) {
							// we really want the best first ? depends...
							work->shareratio[1] = work->shareratio[0];
							work->sharediff[1] = work->sharediff[0];
							xchg(work->nonces[1], work->nonces[0]);
							work_set_target_ratio(work, vhash);
							work->valid_nonces++;
						} else if (work->valid_nonces == 1) {
							bn_set_target_ratio(work, vhash, 1);
							work->valid_nonces++;
						}
						work->valid_nonces = 2; // MAX_NONCES submit limited to 2

						gpulog(LOG_DEBUG, thr_id, "multiple nonces 1:%08x (%g) %u:%08x (%g)",
							work->nonces[0], work->sharediff[0], n, work->nonces[1], work->sharediff[1]);

					} else if (vhash[6] > ptarget[6]) {
						gpu_increment_reject(thr_id);
						if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result %u for %08x does not validate on CPU!", n, resNonces[n]);
					}
				}
				return work->valid_nonces;

			} else if (vhash[6] > ptarget[6]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", resNonces[1]);
			}
		}
		*pnonce += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + (*pnonce));

	*hashes_done = (*pnonce) - first_nonce;
	MyStreamSynchronize(NULL, 0, device_map[thr_id]);
	return 0;
}

// cleanup
extern "C" void free_decred(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
