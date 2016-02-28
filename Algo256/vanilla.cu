/**
 * Optimized Blake-256 8-rounds Cuda Kernel (Tested on SM >3.0)
 * Based upon Blake-256 implementation of Tanguy Pruvot - Nov. 2014
 *
 * midstate computation inherited from
 *  https://github.com/wfr/clblake
 *
 * Provos Alexis - Jan. 2016
 * Reviewed by tpruvot - Feb 2016
 */

#include <stdint.h>
#include <memory.h>
#include <emmintrin.h>

#include "miner.h"

extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

/* threads per block and "magic" */
#define TPB 768
#define NPT 224
#define NBN 2

__constant__ uint32_t d_data[16];

/* 16 gpu threads max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* hash by cpu with blake 256 */
extern "C" void vanillahash(void *output, const void *input, int8_t blakerounds)
{
	uchar hash[64];
	sph_blake256_context ctx;

	sph_blake256_set_rounds(blakerounds);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 80);
	sph_blake256_close(&ctx, hash);

	memcpy(output, hash, 32);
}

__global__ __launch_bounds__(TPB,1)
void vanilla_gpu_hash_16_8(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce,const uint32_t highTarget)
{
	uint32_t v[16];
	uint32_t tmp[13];

	const uint32_t thread   = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t step     = gridDim.x * blockDim.x;
	const uint32_t maxNonce = startNonce + threads;

	const uint32_t c_u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	const uint32_t h0 = d_data[0];      const uint32_t h1 = d_data[1];
	const uint32_t h2 = d_data[2];      const uint32_t h3 = d_data[3];
	const uint32_t h4 = d_data[4];      //const uint32_t h5 = d_data[5]; no need
	const uint32_t h6 = d_data[5];      const uint32_t h7 = d_data[6];
	const uint32_t m0 = d_data[7];      const uint32_t m1 = d_data[8];
	const uint32_t m2 = d_data[9];      //le' nonce
	const uint32_t m4 = 0x80000000UL;   const uint32_t m5 = 0;
	const uint32_t m6 = 0;              const uint32_t m7 = 0;
	const uint32_t m8 = 0;              const uint32_t m9 = 0;
	const uint32_t m10 = 0;             const uint32_t m11 = 0;
	const uint32_t m12 = 0;             const uint32_t m13 = 1;
	const uint32_t m14 = 0;             const uint32_t m15 = 640;

	//---MORE PRECOMPUTATIONS
	tmp[ 0] = d_data[10];              tmp[ 1] = d_data[11];
	tmp[ 2] = d_data[12];              tmp[ 3] = c_u256[1] + tmp[2];
	tmp[ 4] = d_data[13];              tmp[ 5] = d_data[14];
	tmp[ 6] = c_u256[2] + tmp[5];      tmp[ 7] = d_data[15];

	tmp[ 5] = __byte_perm(tmp[5] ^ h2,0, 0x0321);   tmp[ 6] += tmp[5];
	tmp[ 7] = ROTR32(tmp[7] ^ tmp[6],7);            tmp[ 8] = __byte_perm(c_u256[7] ^ h3,0, 0x1032);
	tmp[ 9] = c_u256[3] + tmp[8];                   tmp[10] = ROTR32(h7 ^ tmp[9], 12);
	tmp[11] = h3 + c_u256[6] + tmp[10];

	tmp[ 8] = __byte_perm(tmp[8] ^ tmp[11],0, 0x0321);  tmp[ 9] += tmp[8];
	tmp[10] = ROTR32(tmp[10] ^ tmp[9],7);
	//---END OF MORE PRECOMPUTATIONS

	for(uint64_t m3 = startNonce + thread ; m3<maxNonce ; m3+=step){

		//All i need is, h0,h1,h2,h4,h6,h7,m0,m1,m2 ++ tmps (13) //22 vars
		v[0]  = h0;     v[1]  = h1;     v[2]  = h2;     v[3]  = tmp[11];
		v[4]  = h4;     v[5]  = tmp[4]; v[6]  = tmp[7]; v[7]  = tmp[10];
		v[8]  = tmp[1]; v[9]  = tmp[3]; v[10] = tmp[6]; v[11] = tmp[9];
		v[12] = tmp[0]; v[13] = tmp[2]; v[14] = tmp[5]; v[15] = tmp[8];

		v[ 1] += m3 ^ c_u256[2];        v[13] = __byte_perm(v[13] ^ v[1],0, 0x0321);v[ 9] += v[13];     v[5] = ROTR32(v[5] ^ v[9], 7);
		v[ 0] += v[5];                  v[15] = __byte_perm(v[15] ^ v[0],0, 0x1032);v[10] += v[15];     v[5] = ROTR32(v[5] ^ v[10], 12);
		v[ 0] += c_u256[8] + v[5];      v[15] = __byte_perm(v[15] ^ v[0],0, 0x0321);v[10] += v[15];     v[5] = ROTR32(v[5] ^ v[10], 7);

		#define GSPREC(a,b,c,d,x,y) { \
			v[a] += (m##x ^ c_u256[y]) + v[b]; \
			v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
			v[c] += v[d]; \
			v[b] = ROTR32(v[b] ^ v[c], 12); \
			v[a] += (m##y ^ c_u256[x]) + v[b]; \
			v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
			v[c] += v[d]; \
			v[b] = ROTR32(v[b] ^ v[c], 7); \
		}

		GSPREC(1, 6, 11, 12, 10, 11);   GSPREC(2, 7, 8, 13, 12, 13);    GSPREC(3, 4, 9, 14, 14, 15);
		//  { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
		GSPREC(0, 4, 8, 12, 14, 10);    GSPREC(1, 5, 9, 13, 4, 8);      GSPREC(2, 6, 10, 14, 9, 15);    GSPREC(3, 7, 11, 15, 13, 6);
		GSPREC(0, 5, 10, 15, 1, 12);    GSPREC(1, 6, 11, 12, 0, 2);     GSPREC(2, 7, 8, 13, 11, 7);     GSPREC(3, 4, 9, 14, 5, 3);
		//  { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
		GSPREC(0, 4, 8, 12, 11, 8);     GSPREC(1, 5, 9, 13, 12, 0);     GSPREC(2, 6, 10, 14, 5, 2);     GSPREC(3, 7, 11, 15, 15, 13);
		GSPREC(0, 5, 10, 15, 10, 14);   GSPREC(1, 6, 11, 12, 3, 6);     GSPREC(2, 7, 8, 13, 7, 1);      GSPREC(3, 4, 9, 14, 9, 4);
		//  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
		GSPREC(0, 4, 8, 12, 7, 9);      GSPREC(1, 5, 9, 13, 3, 1);      GSPREC(2, 6, 10, 14, 13, 12);   GSPREC(3, 7, 11, 15, 11, 14);
		GSPREC(0, 5, 10, 15, 2, 6);     GSPREC(1, 6, 11, 12, 5, 10);    GSPREC(2, 7, 8, 13, 4, 0);      GSPREC(3, 4, 9, 14, 15, 8);
		//  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
		GSPREC(0, 4, 8, 12, 9, 0);      GSPREC(1, 5, 9, 13, 5, 7);      GSPREC(2, 6, 10, 14, 2, 4);     GSPREC(3, 7, 11, 15, 10, 15);
		GSPREC(0, 5, 10, 15, 14, 1);    GSPREC(1, 6, 11, 12, 11, 12);   GSPREC(2, 7, 8, 13, 6, 8);      GSPREC(3, 4, 9, 14, 3, 13);
		//  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
		GSPREC(0, 4, 8, 12, 2, 12);     GSPREC(1, 5, 9, 13, 6, 10);     GSPREC(2, 6, 10, 14, 0, 11);    GSPREC(3, 7, 11, 15, 8, 3);
		GSPREC(0, 5, 10, 15, 4, 13);    GSPREC(1, 6, 11, 12, 7, 5);     GSPREC(2, 7, 8, 13, 15, 14);    GSPREC(3, 4, 9, 14, 1, 9);
		//  { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
		GSPREC(0, 4, 8, 12, 12, 5);     GSPREC(1, 5, 9, 13, 1, 15);     GSPREC(2, 6, 10, 14, 14, 13);   GSPREC(3, 7, 11, 15, 4, 10);
		GSPREC(0, 5, 10, 15, 0, 7);     GSPREC(1, 6, 11, 12, 6, 3);     GSPREC(2, 7, 8, 13, 9, 2);      GSPREC(3, 4, 9, 14, 8, 11);
		//  { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
		GSPREC(0, 4, 8, 12, 13, 11);    GSPREC(1, 5, 9, 13, 7, 14);     GSPREC(2, 6, 10, 14, 12, 1);    GSPREC(3, 7, 11, 15, 3, 9);

		v[ 0] += (m5 ^ c_u256[0]) + v[5];   v[15] = __byte_perm(v[15] ^ v[0],0, 0x1032);
		v[10] += v[15];                     v[ 5] = ROTR32(v[5] ^ v[10], 12);
		v[ 0] += (m0 ^ c_u256[5]) + v[5];   v[15] = __byte_perm(v[15] ^ v[0],0, 0x0321);

		v[2] += (m8 ^ c_u256[6]) + v[7];    v[13] = __byte_perm(v[13] ^ v[2],0, 0x1032);
		v[8] += v[13];                      v[ 7] = ROTR32(v[7] ^ v[8], 12);
		v[2] += (m6 ^ c_u256[8]) + v[7];    v[13] = __byte_perm(v[13] ^ v[2],0, 0x0321);
		v[8] += v[13];                      v[ 7] = ROTR32(v[7] ^ v[8], 7);

		// only compute h6 & 7
		if((h7^v[7]^v[15])==0){
			GSPREC(1, 6, 11, 12, 15, 4);
			v[ 3] += (m2 ^ c_u256[10]) + v[4];
			v[14]  = __byte_perm(v[14] ^ v[3],0, 0x1032);
			v[ 9] += v[14];
			v[ 4]  = ROTR32(v[4] ^ v[9],12);
			v[ 3] += (m10 ^ c_u256[2]) + v[4];
			v[14]  = __byte_perm(v[14] ^ v[3],0, 0x0321);
			if(cuda_swab32(h6^v[6]^v[14]) <= highTarget) {
#if NBN == 2
			/* keep the smallest nonce, + extra one if found */
			if (m3 < resNonce[0]){
				resNonce[1] = resNonce[0];
				resNonce[0] = m3;
			}
			else
				resNonce[1] = m3;
#else
			resNonce[0] = m3;
#endif
			}
		}
	}
}


#define round(r) \
		/*        column step          */ \
		buf1 = _mm_set_epi32(m.u32[sig[r][ 6]], m.u32[sig[r][ 4]], m.u32[sig[r][ 2]], m.u32[sig[r][ 0]]); \
		buf2  = _mm_set_epi32(z[sig[r][ 7]], z[sig[r][ 5]], z[sig[r][ 3]],z[sig[r][ 1]]); \
		buf1 = _mm_xor_si128( buf1, buf2); \
		row1 = _mm_add_epi32( _mm_add_epi32( row1, buf1), row2 ); \
		buf1  = _mm_set_epi32(z[sig[r][ 6]], z[sig[r][ 4]], z[sig[r][ 2]], z[sig[r][ 0]]); \
		buf2 = _mm_set_epi32(m.u32[sig[r][ 7]], m.u32[sig[r][ 5]], m.u32[sig[r][ 3]], m.u32[sig[r][ 1]]); \
		row4 = _mm_xor_si128( row4, row1 ); \
		row4 = _mm_xor_si128(_mm_srli_epi32( row4, 16 ),_mm_slli_epi32( row4, 16 )); \
		row3 = _mm_add_epi32( row3, row4 );   \
		row2 = _mm_xor_si128( row2, row3 ); \
		buf1 = _mm_xor_si128( buf1, buf2); \
		row2 = _mm_xor_si128(_mm_srli_epi32( row2, 12 ),_mm_slli_epi32( row2, 20 )); \
		row1 = _mm_add_epi32( _mm_add_epi32( row1, buf1), row2 ); \
		row4 = _mm_xor_si128( row4, row1 ); \
		row4 = _mm_xor_si128(_mm_srli_epi32( row4,  8 ),_mm_slli_epi32( row4, 24 )); \
		row3 = _mm_add_epi32( row3, row4 );   \
		row4 = _mm_shuffle_epi32( row4, _MM_SHUFFLE(2,1,0,3) ); \
		row2 = _mm_xor_si128( row2, row3 ); \
		row2 = _mm_xor_si128(_mm_srli_epi32( row2,  7 ),_mm_slli_epi32( row2, 25 )); \
\
		row3 = _mm_shuffle_epi32( row3, _MM_SHUFFLE(1,0,3,2) ); \
		row2 = _mm_shuffle_epi32( row2, _MM_SHUFFLE(0,3,2,1) ); \
\
	   /*       diagonal step         */ \
		buf1 = _mm_set_epi32(m.u32[sig[r][14]], m.u32[sig[r][12]], m.u32[sig[r][10]], m.u32[sig[r][ 8]]); \
		buf2  = _mm_set_epi32(z[sig[r][15]], z[sig[r][13]], z[sig[r][11]], z[sig[r][ 9]]); \
		buf1 = _mm_xor_si128( buf1, buf2); \
		row1 = _mm_add_epi32( _mm_add_epi32( row1, buf1 ), row2 ); \
		buf1  = _mm_set_epi32(z[sig[r][14]], z[sig[r][12]], z[sig[r][10]], z[sig[r][ 8]]); \
		buf2 = _mm_set_epi32(m.u32[sig[r][15]], m.u32[sig[r][13]], m.u32[sig[r][11]], m.u32[sig[r][ 9]]); \
		row4 = _mm_xor_si128( row4, row1 ); \
		buf1 = _mm_xor_si128( buf1, buf2); \
		row4 = _mm_xor_si128(_mm_srli_epi32( row4, 16 ),_mm_slli_epi32( row4, 16 )); \
		row3 = _mm_add_epi32( row3, row4 );   \
		row2 = _mm_xor_si128( row2, row3 ); \
		row2 = _mm_xor_si128(_mm_srli_epi32( row2, 12 ),_mm_slli_epi32( row2, 20 )); \
		row1 = _mm_add_epi32( _mm_add_epi32( row1, buf1 ), row2 ); \
		row4 = _mm_xor_si128( row4, row1 ); \
		row4 = _mm_xor_si128(_mm_srli_epi32( row4,  8 ),_mm_slli_epi32( row4, 24 )); \
		row3 = _mm_add_epi32( row3, row4 );   \
		row4 = _mm_shuffle_epi32( row4, _MM_SHUFFLE(0,3,2,1) ); \
		row2 = _mm_xor_si128( row2, row3 ); \
		row2 = _mm_xor_si128(_mm_srli_epi32( row2,  7 ),_mm_slli_epi32( row2, 25 )); \
\
		row3 = _mm_shuffle_epi32( row3, _MM_SHUFFLE(1,0,3,2) ); \
		row2 = _mm_shuffle_epi32( row2, _MM_SHUFFLE(2,1,0,3) ); \
\

#define LOADU(p)  _mm_loadu_si128( (__m128i *)(p) )

#define BSWAP32(r) do{ \
   r = _mm_shufflehi_epi16(r, _MM_SHUFFLE(2, 3, 0, 1));\
   r = _mm_shufflelo_epi16(r, _MM_SHUFFLE(2, 3, 0, 1));\
   r = _mm_xor_si128(_mm_slli_epi16(r, 8), _mm_srli_epi16(r, 8));\
} while(0)


__host__
void vanilla_cpu_setBlock_16(const uint32_t* endiandata, uint32_t *penddata){

	uint32_t _ALIGN(32) h[16];
	h[0]=0x6A09E667;    h[1]=0xBB67AE85;    h[2]=0x3C6EF372;    h[3]=0xA54FF53A;
	h[4]=0x510E527F;    h[5]=0x9B05688C;    h[6]=0x1F83D9AB;    h[7]=0x5BE0CD19;

	__m128i row1, row2, row3, row4;
	__m128i buf1, buf2;

	union {
		uint32_t u32[16];
		__m128i u128[4];
	} m;
	static const int sig[][16] = {
		{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } , { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
		{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } , {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
		{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } , {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
		{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } , { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
		{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } , { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
		{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } , { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
		{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } , {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 }
	};
	static const uint32_t z[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};
	/* get message */
	m.u128[0] = LOADU(endiandata + 0);
	m.u128[1] = LOADU(endiandata + 4);
	m.u128[2] = LOADU(endiandata + 8);
	m.u128[3] = LOADU(endiandata + 12);
	BSWAP32(m.u128[0]); BSWAP32(m.u128[1]); BSWAP32(m.u128[2]); BSWAP32(m.u128[3]);

	row1 = _mm_set_epi32(h[ 3], h[ 2], h[ 1], h[ 0]);
	row2 = _mm_set_epi32(h[ 7], h[ 6], h[ 5], h[ 4]);
	row3 = _mm_set_epi32(0x03707344, 0x13198A2E, 0x85A308D3, 0x243F6A88);
	row4 = _mm_set_epi32(0xEC4E6C89, 0x082EFA98, 0x299F31D0^512, 0xA4093822^512);

	round( 0);  round( 1);  round( 2);
	round( 3);  round( 4);  round( 5);
	round( 6);  round( 7);

	_mm_store_si128( (__m128i *)m.u32, _mm_xor_si128(row1,row3));
	h[0] ^= m.u32[ 0];  h[1] ^= m.u32[ 1];
	h[2] ^= m.u32[ 2];  h[3] ^= m.u32[ 3];
	_mm_store_si128( (__m128i *)m.u32, _mm_xor_si128(row2,row4));
	h[4] ^= m.u32[ 0];  h[5] ^= m.u32[ 1];
	h[6] ^= m.u32[ 2];  h[7] ^= m.u32[ 3];

	uint32_t tmp = h[5];
	h[ 5] = h[6];
	h[ 6] = h[7];
	h[ 7] = penddata[0];
	h[ 8] = penddata[1];
	h[ 9] = penddata[2];
	h[10] = SPH_C32(0xA4093822) ^ 640;
	h[11] = SPH_C32(0x243F6A88);

	h[ 0] += (h[7] ^ SPH_C32(0x85A308D3)) + h[4];
	h[10]  = SPH_ROTR32(h[10] ^ h[0],16);
	h[11] += h[10];
	h[ 4]  = SPH_ROTR32(h[4] ^ h[11], 12);
	h[ 0] += (h[8] ^ SPH_C32(0x243F6A88)) + h[4];
	h[10]  = SPH_ROTR32(h[10] ^ h[0],8);
	h[11] += h[10];
	h[ 4]  = SPH_ROTR32(h[4] ^ h[11], 7);

	h[1] += (h[ 9] ^ SPH_C32(0x03707344)) + tmp;

	h[12] = SPH_ROTR32(SPH_C32(0x299F31D0) ^ 640 ^ h[1],16);
	h[13] = ROTR32(tmp ^ (SPH_C32(0x85A308D3) + h[12]), 12);

	h[ 1] += h[13];
	h[ 2] += (0x80000000UL ^ SPH_C32(0x299F31D0)) + h[5];

	h[14]  = SPH_ROTR32(SPH_C32(0x082EFA98) ^ h[2], 16);
	h[15]  = SPH_C32(0x13198A2E) + h[14];
	h[15]  = SPH_ROTR32(h[5] ^ h[15], 12);

	h[ 3] += SPH_C32(0xEC4E6C89) + h[6];
	h[ 0] += SPH_C32(0x38D01377);

	h[ 2] += SPH_C32(0xA4093822) + h[15];

	cudaMemcpyToSymbol(d_data, h, 16*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_vanilla(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, const int8_t blakerounds)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce  = pdata[19];
	const uint32_t targetHigh   = ptarget[6];
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 30 : 24;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	int rc = 0;

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		CUDA_CALL_OR_RET_X(cudaHostAlloc((void**)&h_resNonce[thr_id], NBN*sizeof(uint32_t), cudaHostAllocMapped),0);
		CUDA_CALL_OR_RET_X(cudaHostGetDevicePointer((void**)&d_resNonce[thr_id],(void*)h_resNonce[thr_id], 0),0);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];

	for (int k = 0; k < 16; k++)
		be32enc(&endiandata[k], pdata[k]);

	vanilla_cpu_setBlock_16(endiandata,&pdata[16]);

	cudaMemset(d_resNonce[thr_id], 0xff, sizeof(uint32_t));
	const dim3 grid((throughput + (NPT*TPB)-1)/(NPT*TPB));
	const dim3 block(TPB);
	do {
		vanilla_gpu_hash_16_8<<<grid,block>>>(throughput, pdata[19], d_resNonce[thr_id], targetHigh);
		cudaThreadSynchronize();

		if (h_resNonce[thr_id][0] != UINT32_MAX){
			uint32_t vhashcpu[8];
			uint32_t Htarg = (uint32_t)targetHigh;

			for (int k=0; k < 19; k++)
				be32enc(&endiandata[k], pdata[k]);

			be32enc(&endiandata[19], h_resNonce[thr_id][0]);
			vanillahash(vhashcpu, endiandata, blakerounds);

			if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget)){
				rc = 1;
				work_set_target_ratio(work, vhashcpu);
				*hashes_done = pdata[19] - first_nonce + throughput;
				work->nonces[0] = h_resNonce[thr_id][0];
#if NBN > 1
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					work->nonces[1] = h_resNonce[thr_id][1];
					be32enc(&endiandata[19], work->nonces[1]);
					vanillahash(vhashcpu, endiandata, blakerounds);
					if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio) {
						work_set_target_ratio(work, vhashcpu);
						xchg(work->nonces[1], work->nonces[0]);
					}
					rc = 2;
				}
				pdata[21] = work->nonces[1];
#endif
				pdata[19] = work->nonces[0];
				return rc;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce;
	MyStreamSynchronize(NULL, 0, dev_id);
	return rc;
}

// cleanup
extern "C" void free_vanilla(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
