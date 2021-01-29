/***************************************************************************************************
 * SM 2.x SIMD512 CUDA Implementation without shuffle
 *
 * cbuchner 2014 / tpruvot 2015
 */

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 210
#endif

#if __CUDA_ARCH__ < 300

#define T32(x) (x)

#if 0 /* already declared in SM 3+ implementation */
__constant__  uint32_t c_IV_512[32];
const uint32_t h_IV_512[32] = {
	0x0ba16b95, 0x72f999ad, 0x9fecc2ae, 0xba3264fc, 0x5e894929, 0x8e9f30e5, 0x2f1daa37, 0xf0f2c558,
	0xac506643, 0xa90635a5, 0xe25b878b, 0xaab7878f, 0x88817f7a, 0x0a02892b, 0x559a7550, 0x598f657e,
	0x7eef60a1, 0x6b70e3e8, 0x9c1714d1, 0xb958e2a8, 0xab02675e, 0xed1c014f, 0xcd8d65bb, 0xfdb7a257,
	0x09254899, 0xd699c7bc, 0x9019b6dc, 0x2b9022e4, 0x8fa14956, 0x21bf9bd3, 0xb94d0943, 0x6ffddc22
};

__constant__ int c_FFT128_8_16_Twiddle[128];
static const int h_FFT128_8_16_Twiddle[128] = {
	1,   1,   1,   1,   1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
	1,  60,   2, 120,   4,  -17,   8, -34,  16, -68,  32, 121,  64, -15, 128, -30,
	1,  46,  60, -67,   2,   92, 120, 123,   4, -73, -17, -11,   8, 111, -34, -22,
	1, -67, 120, -73,   8,  -22, -68, -70,  64,  81, -30, -46,  -2,-123,  17,-111,
	1,-118,  46, -31,  60,  116, -67, -61,   2,  21,  92, -62, 120, -25, 123,-122,
	1, 116,  92,-122, -17,   84, -22,  18,  32, 114, 117, -49, -30, 118,  67,  62,
	1, -31, -67,  21, 120, -122, -73, -50,   8,   9, -22, -89, -68,  52, -70, 114,
	1, -61, 123, -50, -34,   18, -70, -99, 128, -98,  67,  25,  17,  -9,  35, -79
};

__constant__ int c_FFT256_2_128_Twiddle[128];
static const int h_FFT256_2_128_Twiddle[128] = {
	  1,  41,-118,  45,  46,  87, -31,  14,
	 60,-110, 116,-127, -67,  80, -61,  69,
	  2,  82,  21,  90,  92, -83, -62,  28,
	120,  37, -25,   3, 123, -97,-122,-119,
	  4, -93,  42, -77, -73,  91,-124,  56,
	-17,  74, -50,   6, -11,  63,  13,  19,
	  8,  71,  84, 103, 111, -75,   9, 112,
	-34,-109,-100,  12, -22, 126,  26,  38,
	 16,-115, -89, -51, -35, 107,  18, -33,
	-68,  39,  57,  24, -44,  -5,  52,  76,
	 32,  27,  79,-102, -70, -43,  36, -66,
	121,  78, 114,  48, -88, -10, 104,-105,
	 64,  54, -99,  53, 117, -86,  72, 125,
	-15,-101, -29,  96,  81, -20, -49,  47,
	128, 108,  59, 106, -23,  85,-113,  -7,
	-30,  55, -58, -65, -95, -40, -98,  94
};
#endif

__constant__ int c_FFT[256] = {
	// this is the FFT result in revbin permuted order
	4, -4, 32, -32, -60, 60, 60, -60, 101, -101, 58, -58, 112, -112, -11, 11, -92, 92,
	-119, 119, 42, -42, -82, 82, 32, -32, 32, -32, 121, -121, 17, -17, -47, 47, 63,
	-63, 107, -107, -76, 76, -119, 119, -83, 83, 126, -126, 94, -94, -23, 23, -76,
	76, -47, 47, 92, -92, -117, 117, 73, -73, -53, 53, 88, -88, -80, 80, -47, 47,
	5, -5, 67, -67, 34, -34, 4, -4, 87, -87, -28, 28, -70, 70, -110, 110, -18, 18, 93,
	-93, 51, -51, 36, -36, 118, -118, -106, 106, 45, -45, -108, 108, -44, 44, 117,
	-117, -121, 121, -37, 37, 65, -65, 37, -37, 40, -40, -42, 42, 91, -91, -128, 128,
	-21, 21, 94, -94, -98, 98, -47, 47, 28, -28, 115, -115, 16, -16, -20, 20, 122,
	-122, 115, -115, 46, -46, 84, -84, -127, 127, 57, -57, 127, -127, -80, 80, 24,
	-24, 15, -15, 29, -29, -78, 78, -126, 126, 16, -16, 52, -52, 55, -55, 110, -110,
	-51, 51, -120, 120, -124, 124, -24, 24, -76, 76, 26, -26, -21, 21, -64, 64, -99,
	99, 85, -85, -15, 15, -120, 120, -116, 116, 85, -85, 12, -12, -24, 24, 4, -4,
	79, -79, 76, -76, 23, -23, 4, -4, -108, 108, -20, 20, 73, -73, -42, 42, -7, 7,
	-29, 29, -123, 123, 49, -49, -96, 96, -68, 68, -112, 112, 116, -116, -24, 24, 93,
	-93, -125, 125, -86, 86, 117, -117, -91, 91, 42, -42, 87, -87, -117, 117, 102, -102
};

__constant__ int c_P8[32][8] = {
	{ 2, 66, 34, 98, 18, 82, 50, 114 },
	{ 6, 70, 38, 102, 22, 86, 54, 118 },
	{ 0, 64, 32, 96, 16, 80, 48, 112 },
	{ 4, 68, 36, 100, 20, 84, 52, 116 },
	{ 14, 78, 46, 110, 30, 94, 62, 126 },
	{ 10, 74, 42, 106, 26, 90, 58, 122 },
	{ 12, 76, 44, 108, 28, 92, 60, 124 },
	{ 8, 72, 40, 104, 24, 88, 56, 120 },
	{ 15, 79, 47, 111, 31, 95, 63, 127 },
	{ 13, 77, 45, 109, 29, 93, 61, 125 },
	{ 3, 67, 35, 99, 19, 83, 51, 115 },
	{ 1, 65, 33, 97, 17, 81, 49, 113 },
	{ 9, 73, 41, 105, 25, 89, 57, 121 },
	{ 11, 75, 43, 107, 27, 91, 59, 123 },
	{ 5, 69, 37, 101, 21, 85, 53, 117 },
	{ 7, 71, 39, 103, 23, 87, 55, 119 },
	{ 8, 72, 40, 104, 24, 88, 56, 120 },
	{ 4, 68, 36, 100, 20, 84, 52, 116 },
	{ 14, 78, 46, 110, 30, 94, 62, 126 },
	{ 2, 66, 34, 98, 18, 82, 50, 114 },
	{ 6, 70, 38, 102, 22, 86, 54, 118 },
	{ 10, 74, 42, 106, 26, 90, 58, 122 },
	{ 0, 64, 32, 96, 16, 80, 48, 112 },
	{ 12, 76, 44, 108, 28, 92, 60, 124 },
	{ 134, 198, 166, 230, 150, 214, 182, 246 },
	{ 128, 192, 160, 224, 144, 208, 176, 240 },
	{ 136, 200, 168, 232, 152, 216, 184, 248 },
	{ 142, 206, 174, 238, 158, 222, 190, 254 },
	{ 140, 204, 172, 236, 156, 220, 188, 252 },
	{ 138, 202, 170, 234, 154, 218, 186, 250 },
	{ 130, 194, 162, 226, 146, 210, 178, 242 },
	{ 132, 196, 164, 228, 148, 212, 180, 244 },
};

__constant__ int c_Q8[32][8] = {
	{ 130, 194, 162, 226, 146, 210, 178, 242 },
	{ 134, 198, 166, 230, 150, 214, 182, 246 },
	{ 128, 192, 160, 224, 144, 208, 176, 240 },
	{ 132, 196, 164, 228, 148, 212, 180, 244 },
	{ 142, 206, 174, 238, 158, 222, 190, 254 },
	{ 138, 202, 170, 234, 154, 218, 186, 250 },
	{ 140, 204, 172, 236, 156, 220, 188, 252 },
	{ 136, 200, 168, 232, 152, 216, 184, 248 },
	{ 143, 207, 175, 239, 159, 223, 191, 255 },
	{ 141, 205, 173, 237, 157, 221, 189, 253 },
	{ 131, 195, 163, 227, 147, 211, 179, 243 },
	{ 129, 193, 161, 225, 145, 209, 177, 241 },
	{ 137, 201, 169, 233, 153, 217, 185, 249 },
	{ 139, 203, 171, 235, 155, 219, 187, 251 },
	{ 133, 197, 165, 229, 149, 213, 181, 245 },
	{ 135, 199, 167, 231, 151, 215, 183, 247 },
	{ 9, 73, 41, 105, 25, 89, 57, 121 },
	{ 5, 69, 37, 101, 21, 85, 53, 117 },
	{ 15, 79, 47, 111, 31, 95, 63, 127 },
	{ 3, 67, 35, 99, 19, 83, 51, 115 },
	{ 7, 71, 39, 103, 23, 87, 55, 119 },
	{ 11, 75, 43, 107, 27, 91, 59, 123 },
	{ 1, 65, 33, 97, 17, 81, 49, 113 },
	{ 13, 77, 45, 109, 29, 93, 61, 125 },
	{ 135, 199, 167, 231, 151, 215, 183, 247 },
	{ 129, 193, 161, 225, 145, 209, 177, 241 },
	{ 137, 201, 169, 233, 153, 217, 185, 249 },
	{ 143, 207, 175, 239, 159, 223, 191, 255 },
	{ 141, 205, 173, 237, 157, 221, 189, 253 },
	{ 139, 203, 171, 235, 155, 219, 187, 251 },
	{ 131, 195, 163, 227, 147, 211, 179, 243 },
	{ 133, 197, 165, 229, 149, 213, 181, 245 },
};

#define p8_xor(x) ( ((x)%7) == 0 ? 1 : \
	((x)%7) == 1 ? 6 : \
	((x)%7) == 2 ? 2 : \
	((x)%7) == 3 ? 3 : \
	((x)%7) == 4 ? 5 : \
	((x)%7) == 5 ? 7 : 4 )

/************* the round function ****************/

//#define IF(x, y, z) ((((y) ^ (z)) & (x)) ^ (z))
//#define MAJ(x, y, z) (((z) & (y)) | (((z) | (y)) & (x)))

__device__ __forceinline__
void STEP8_IF(const uint32_t *w, const int i, const int r, const int s, uint32_t *A, const uint32_t *B, const uint32_t *C, uint32_t *D)
{
	uint32_t R[8];
	#pragma unroll 8
	for(int j=0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	#pragma unroll 8
	for(int j=0; j<8; j++) {
		D[j] = D[j] + w[j] + IF(A[j], B[j], C[j]);
		D[j] = T32(ROTL32(T32(D[j]), s) + R[j^p8_xor(i)]);
		A[j] = R[j];
	}
}

__device__ __forceinline__
void STEP8_MAJ(const uint32_t *w, const int i, const int r, const int s, uint32_t *A, const uint32_t *B, const uint32_t *C, uint32_t *D)
{
	uint32_t R[8];
	#pragma unroll 8
	for(int j=0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	#pragma unroll 8
	for(int j=0; j<8; j++) {
		D[j] = D[j] + w[j] + MAJ(A[j], B[j], C[j]);
		D[j] = T32(ROTL32(T32(D[j]), s) + R[j^p8_xor(i)]);
		A[j] = R[j];
	}
}

__device__ __forceinline__
void Round8(uint32_t A[32], const int y[256], int i, int r, int s, int t, int u)
{
	uint32_t w[8][8];
	int code = i<2? 185: 233;

	/*
	 * The FFT output y is in revbin permuted order,
	 * but this is included in the tables P and Q
	 */

	#pragma unroll 8
	for(int a=0; a<8; a++) {
		#pragma unroll 8
		for(int b=0; b<8; b++) {
			w[a][b] = __byte_perm( (y[c_P8[8*i+a][b]] * code), (y[c_Q8[8*i+a][b]] * code), 0x5410);
		}
	}

	STEP8_IF(w[0], 8*i+0, r, s, A, &A[8], &A[16], &A[24]);
	STEP8_IF(w[1], 8*i+1, s, t, &A[24], A, &A[8], &A[16]);
	STEP8_IF(w[2], 8*i+2, t, u, &A[16], &A[24], A, &A[8]);
	STEP8_IF(w[3], 8*i+3, u, r, &A[8], &A[16], &A[24], A);

	STEP8_MAJ(w[4], 8*i+4, r, s, A, &A[8], &A[16], &A[24]);
	STEP8_MAJ(w[5], 8*i+5, s, t, &A[24], A, &A[8], &A[16]);
	STEP8_MAJ(w[6], 8*i+6, t, u, &A[16], &A[24], A, &A[8]);
	STEP8_MAJ(w[7], 8*i+7, u, r, &A[8], &A[16], &A[24], A);
}


/********************* Message expansion ************************/

/*
 * Reduce modulo 257; result is in [-127; 383]
 * REDUCE(x) := (x&255) - (x>>8)
 */
#define REDUCE(x) (((x)&255) - ((x)>>8))

/*
 * Reduce from [-127; 383] to [-128; 128]
 * EXTRA_REDUCE_S(x) := x<=128 ? x : x-257
 */
#define EXTRA_REDUCE_S(x) \
	((x)<=128 ? (x) : (x)-257)

/*
 * Reduce modulo 257; result is in [-128; 128]
 */
#define REDUCE_FULL_S(x) \
	EXTRA_REDUCE_S(REDUCE(x))

__device__ __forceinline__
void FFT_8(int *y, int stripe)
{
	/*
	 * FFT_8 using w=4 as 8th root of unity
	 * Unrolled decimation in frequency (DIF) radix-2 NTT.
	 * Output data is in revbin_permuted order.
	 */
	#define X(i) y[stripe*i]

	#define DO_REDUCE(i) \
		X(i) = REDUCE(X(i))

	#define DO_REDUCE_FULL_S(i) { \
		X(i) = REDUCE(X(i)); \
		X(i) = EXTRA_REDUCE_S(X(i)); \
	}

	#define BUTTERFLY(i,j,n) { \
		int u= X(i); \
		int v= X(j); \
		X(i) = u+v; \
		X(j) = (u-v) << (2*n); \
	}

	BUTTERFLY(0, 4, 0);
	BUTTERFLY(1, 5, 1);
	BUTTERFLY(2, 6, 2);
	BUTTERFLY(3, 7, 3);

	DO_REDUCE(6);
	DO_REDUCE(7);

	BUTTERFLY(0, 2, 0);
	BUTTERFLY(4, 6, 0);
	BUTTERFLY(1, 3, 2);
	BUTTERFLY(5, 7, 2);

	DO_REDUCE(7);

	BUTTERFLY(0, 1, 0);
	BUTTERFLY(2, 3, 0);
	BUTTERFLY(4, 5, 0);
	BUTTERFLY(6, 7, 0);

	DO_REDUCE_FULL_S(0);
	DO_REDUCE_FULL_S(1);
	DO_REDUCE_FULL_S(2);
	DO_REDUCE_FULL_S(3);
	DO_REDUCE_FULL_S(4);
	DO_REDUCE_FULL_S(5);
	DO_REDUCE_FULL_S(6);
	DO_REDUCE_FULL_S(7);

	#undef X
	#undef DO_REDUCE
	#undef DO_REDUCE_FULL_S
	#undef BUTTERFLY
}

__device__ __forceinline__
void FFT_16(int *y, int stripe)
{
	/*
	 * FFT_16 using w=2 as 16th root of unity
	 * Unrolled decimation in frequency (DIF) radix-2 NTT.
	 * Output data is in revbin_permuted order.
	 */

	#define X(i) y[stripe*i]

	#define DO_REDUCE(i) \
		X(i) = REDUCE(X(i))

	#define DO_REDUCE_FULL_S(i) { \
		X(i) = REDUCE(X(i)); \
		X(i) = EXTRA_REDUCE_S(X(i)); \
	}

	#define BUTTERFLY(i,j,n) { \
		int u= X(i); \
		int v= X(j); \
		X(i) = u+v; \
		X(j) = (u-v) << n; \
	}

	BUTTERFLY(0, 8, 0);
	BUTTERFLY(1, 9, 1);
	BUTTERFLY(2, 10, 2);
	BUTTERFLY(3, 11, 3);
	BUTTERFLY(4, 12, 4);
	BUTTERFLY(5, 13, 5);
	BUTTERFLY(6, 14, 6);
	BUTTERFLY(7, 15, 7);

	DO_REDUCE(11);
	DO_REDUCE(12);
	DO_REDUCE(13);
	DO_REDUCE(14);
	DO_REDUCE(15);

	BUTTERFLY( 0, 4, 0);
	BUTTERFLY( 1, 5, 2);
	BUTTERFLY( 2, 6, 4);
	BUTTERFLY( 3, 7, 6);

	BUTTERFLY( 8, 12, 0);
	BUTTERFLY( 9, 13, 2);
	BUTTERFLY(10, 14, 4);
	BUTTERFLY(11, 15, 6);

	DO_REDUCE(5);
	DO_REDUCE(7);
	DO_REDUCE(13);
	DO_REDUCE(15);

	BUTTERFLY( 0, 2, 0);
	BUTTERFLY( 1, 3, 4);
	BUTTERFLY( 4, 6, 0);
	BUTTERFLY( 5, 7, 4);

	BUTTERFLY( 8, 10, 0);
	BUTTERFLY(12, 14, 0);
	BUTTERFLY( 9, 11, 4);
	BUTTERFLY(13, 15, 4);

	BUTTERFLY( 0, 1, 0);
	BUTTERFLY( 2, 3, 0);
	BUTTERFLY( 4, 5, 0);
	BUTTERFLY( 6, 7, 0);

	BUTTERFLY( 8, 9, 0);
	BUTTERFLY(10, 11, 0);
	BUTTERFLY(12, 13, 0);
	BUTTERFLY(14, 15, 0);

	DO_REDUCE_FULL_S( 0);
	DO_REDUCE_FULL_S( 1);
	DO_REDUCE_FULL_S( 2);
	DO_REDUCE_FULL_S( 3);
	DO_REDUCE_FULL_S( 4);
	DO_REDUCE_FULL_S( 5);
	DO_REDUCE_FULL_S( 6);
	DO_REDUCE_FULL_S( 7);
	DO_REDUCE_FULL_S( 8);
	DO_REDUCE_FULL_S( 9);
	DO_REDUCE_FULL_S(10);
	DO_REDUCE_FULL_S(11);
	DO_REDUCE_FULL_S(12);
	DO_REDUCE_FULL_S(13);
	DO_REDUCE_FULL_S(14);
	DO_REDUCE_FULL_S(15);

	#undef X
	#undef DO_REDUCE
	#undef DO_REDUCE_FULL_S
	#undef BUTTERFLY
}

__device__ __forceinline__
void FFT_128_full(int *y)
{
	#pragma unroll 16
	for (int i=0; i<16; i++) {
		FFT_8(y+i,16);
	}

	#pragma unroll 128
	for (int i=0; i<128; i++)
		/*if (i & 7)*/ y[i] = REDUCE(y[i]*c_FFT128_8_16_Twiddle[i]);

	#pragma unroll 8
	for (int i=0; i<8; i++) {
		FFT_16(y+16*i,1);
	}
}

__device__ __forceinline__
void FFT_256_halfzero(int y[256])
{
	/*
	* FFT_256 using w=41 as 256th root of unity.
	* Decimation in frequency (DIF) NTT.
	* Output data is in revbin_permuted order.
	* In place.
	*/
	const int tmp = y[127];

	#pragma unroll 127
	for (int i=0; i<127; i++)
		y[128+i] = REDUCE(y[i] * c_FFT256_2_128_Twiddle[i]);

	/* handle X^255 with an additionnal butterfly */
	y[127] = REDUCE(tmp + 1);
	y[255] = REDUCE((tmp - 1) * c_FFT256_2_128_Twiddle[127]);

	FFT_128_full(y);
	FFT_128_full(y+128);
}

__device__ __forceinline__
void SIMD_Compress(uint32_t A[32], const int *expanded, const uint32_t *M)
{
	uint32_t IV[4][8];

	/* Save the chaining value for the feed-forward */

	#pragma unroll 8
	for(int i=0; i<8; i++) {
		IV[0][i] = A[i];
		IV[1][i] = (&A[8])[i];
		IV[2][i] = (&A[16])[i];
		IV[3][i] = (&A[24])[i];
	}

	/* XOR the message to the chaining value */
	/* we can XOR word-by-word */
	#pragma unroll 8
	for(int i=0; i<8; i++) {
		A[i] ^= M[i];
		(&A[8])[i] ^= M[8+i];
	}

	/* Run the feistel ladders with the expanded message */
	Round8(A, expanded, 0, 3, 23, 17, 27);
	Round8(A, expanded, 1, 28, 19, 22, 7);
	Round8(A, expanded, 2, 29, 9, 15, 5);
	Round8(A, expanded, 3, 4, 13, 10, 25);

	STEP8_IF(IV[0], 32,  4, 13, A, &A[8], &A[16], &A[24]);
	STEP8_IF(IV[1], 33, 13, 10, &A[24], A, &A[8], &A[16]);
	STEP8_IF(IV[2], 34, 10, 25, &A[16], &A[24], A, &A[8]);
	STEP8_IF(IV[3], 35, 25,  4, &A[8], &A[16], &A[24], A);
}


/***************************************************/

__device__ __forceinline__
void SIMDHash(const uint32_t *data, uint32_t *hashval)
{
	uint32_t A[32];
	uint32_t buffer[16];

	#pragma unroll 32
	for (int i=0; i < 32; i++) A[i] = c_IV_512[i];

	#pragma unroll 16
	for (int i=0; i < 16; i++) buffer[i] = data[i];

	/* Message Expansion using Number Theoretical Transform similar to FFT */
	int expanded[256];
	{
		#pragma unroll 16
		for(int i=0; i<64; i+=4) {
			expanded[i+0] = __byte_perm(buffer[i/4],0,0x4440);
			expanded[i+1] = __byte_perm(buffer[i/4],0,0x4441);
			expanded[i+2] = __byte_perm(buffer[i/4],0,0x4442);
			expanded[i+3] = __byte_perm(buffer[i/4],0,0x4443);
		}

		#pragma unroll 16
		for(int i=64; i<128; i+=4) {
			expanded[i+0] = 0;
			expanded[i+1] = 0;
			expanded[i+2] = 0;
			expanded[i+3] = 0;
		}

		FFT_256_halfzero(expanded);
	}

	/* Compression Function */
	 SIMD_Compress(A, expanded, buffer);

	/* Padding Round with known input (hence the FFT can be precomputed) */
	buffer[0] = 512;

	#pragma unroll 15
	for (int i=1; i < 16; i++) buffer[i] = 0;

	SIMD_Compress(A, c_FFT, buffer);

	#pragma unroll 16
	for (int i=0; i < 16; i++)
		hashval[i] = A[i];
}

/***************************************************/
__global__
void x11_simd512_gpu_hash_64_sm2(const uint32_t threads, const uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		const int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*) &g_hash[8 * hashPosition];

		SIMDHash(Hash, Hash);
	}
}

#else
__global__ void x11_simd512_gpu_hash_64_sm2(const uint32_t threads, const uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector) {}
#endif /* __CUDA_ARCH__ < 300 */

__host__
static void x11_simd512_cpu_init_sm2(int thr_id)
{
#ifndef DEVICE_DIRECT_CONSTANTS
	cudaMemcpyToSymbol( c_IV_512, h_IV_512, sizeof(h_IV_512), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( c_FFT128_8_16_Twiddle, h_FFT128_8_16_Twiddle, sizeof(h_FFT128_8_16_Twiddle), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( c_FFT256_2_128_Twiddle, h_FFT256_2_128_Twiddle, sizeof(h_FFT256_2_128_Twiddle), 0, cudaMemcpyHostToDevice);
#endif
}

__host__
static void x11_simd512_cpu_hash_64_sm2(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	x11_simd512_gpu_hash_64_sm2<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	MyStreamSynchronize(NULL, order, thr_id);
}
