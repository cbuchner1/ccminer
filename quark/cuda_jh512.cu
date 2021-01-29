/**
 * JH512 64 and 80 kernels
 *
 * JH80 by tpruvot - 2017 - under GPLv3
 **/
#include <cuda_helper.h>

// #include <stdio.h>  // printf
// #include <unistd.h> // sleep

/* 1344 bytes, align 16 is there to allow ld.const.v4 (made auto. by the compiler) */
__constant__ static __align__(16) uint32_t c_E8_bslice32[42][8] = {
	// Round 0 (Function0)
	{ 0xa2ded572, 0x90d6ab81, 0x67f815df, 0xf6875a4d, 0x0a15847b, 0xc54f9f4e, 0x571523b7, 0x402bd1c3 },
	{ 0xe03a98ea, 0xb4960266, 0x9cfa455c, 0x8a53bbf2, 0x99d2c503, 0x1a1456b5, 0x9a99b266, 0x31a2db88 }, // 1
	{ 0x5c5aa303, 0x8019051c, 0xdb0e199a, 0x1d959e84, 0x0ab23f40, 0xadeb336f, 0x1044c187, 0xdccde75e }, // 2
	{ 0x9213ba10, 0x39812c0a, 0x416bbf02, 0x5078aa37, 0x156578dc, 0xd2bf1a3f, 0xd027bbf7, 0xd3910041 }, // 3
	{ 0x0d5a2d42, 0x0ba75c18, 0x907eccf6, 0xac442bc7, 0x9c9f62dd, 0xd665dfd1, 0xce97c092, 0x23fcc663 }, // 4
	{ 0x036c6e97, 0xbb03f1ee, 0x1ab8e09e, 0xfa618e5d, 0x7e450521, 0xb29796fd, 0xa8ec6c44, 0x97818394 }, // 5
	{ 0x37858e4a, 0x8173fe8a, 0x2f3003db, 0x6c69b8f8, 0x2d8d672a, 0x4672c78a, 0x956a9ffb, 0x14427fc0 }, // 6
	// Round 7 (Function0)
	{ 0x8f15f4c5, 0xb775de52, 0xc45ec7bd, 0xbc88e4ae, 0xa76f4475, 0x1e00b882, 0x80bb118f, 0xf4a3a698 },
	{ 0x338ff48e, 0x20edf1b6, 0x1563a3a9, 0xfde05a7c, 0x24565faa, 0x5ae9ca36, 0x89f9b7d5, 0x362c4206 },
	{ 0x433529ce, 0x591ff5d0, 0x3d98fe4e, 0x86814e6f, 0x74f93a53, 0x81ad9d0e, 0xa74b9a73, 0x9f5ad8af },
	{ 0x670605a7, 0x26077447, 0x6a6234ee, 0x3f1080c6, 0xbe280b8b, 0x6f7ea0e0, 0x2717b96e, 0x7b487ec6 },
	{ 0xa50a550d, 0x81727686, 0xc0a4f84a, 0xd48d6050, 0x9fe7e391, 0x415a9e7e, 0x9ef18e97, 0x62b0e5f3 },
	{ 0xec1f9ffc, 0xf594d74f, 0x7a205440, 0xd895fa9d, 0x001ae4e3, 0x117e2e55, 0x84c9f4ce, 0xa554c324 },
	{ 0x2872df5b, 0xef7c8905, 0x286efebd, 0x2ed349ee, 0xe27ff578, 0x85937e44, 0xb2c4a50f, 0x7f5928eb },
	// Round 14 (Function0)
	{ 0x37695f70, 0x04771bc7, 0x4a3124b3, 0xe720b951, 0xf128865e, 0xe843fe74, 0x65e4d61d, 0x8a87d423 },
	{ 0xa3e8297d, 0xfb301b1d, 0xf2947692, 0xe01bdc5b, 0x097acbdd, 0x4f4924da, 0xc1d9309b, 0xbf829cf2 },
	{ 0x31bae7a4, 0x32fcae3b, 0xffbf70b4, 0x39d3bb53, 0x0544320d, 0xc1c39f45, 0x48bcf8de, 0xa08b29e0 },
	{ 0xfd05c9e5, 0x01b771a2, 0x0f09aef7, 0x95ed44e3, 0x12347094, 0x368e3be9, 0x34f19042, 0x4a982f4f },
	{ 0x631d4088, 0xf14abb7e, 0x15f66ca0, 0x30c60ae2, 0x4b44c147, 0xc5b67046, 0xffaf5287, 0xe68c6ecc },
	{ 0x56a4d5a4, 0x45ce5773, 0x00ca4fbd, 0xadd16430, 0x4b849dda, 0x68cea6e8, 0xae183ec8, 0x67255c14 },
	{ 0xf28cdaa3, 0x20b2601f, 0x16e10ecb, 0x7b846fc2, 0x5806e933, 0x7facced1, 0x9a99949a, 0x1885d1a0 },
	// Round 21 (Function0)
	{ 0xa15b5932, 0x67633d9f, 0xd319dd8d, 0xba6b04e4, 0xc01c9a50, 0xab19caf6, 0x46b4a5aa, 0x7eee560b },
	{ 0xea79b11f, 0x5aac571d, 0x742128a9, 0x76d35075, 0x35f7bde9, 0xfec2463a, 0xee51363b, 0x01707da3 },
	{ 0xafc135f7, 0x15638341, 0x42d8a498, 0xa8db3aea, 0x20eced78, 0x4d3bc3fa, 0x79676b9e, 0x832c8332 },
	{ 0x1f3b40a7, 0x6c4e3ee7, 0xf347271c, 0xfd4f21d2, 0x34f04059, 0x398dfdb8, 0x9a762db7, 0xef5957dc },
	{ 0x490c9b8d, 0xd0ae3b7d, 0xdaeb492b, 0x84558d7a, 0x49d7a25b, 0xf0e9a5f5, 0x0d70f368, 0x658ef8e4 },
	{ 0xf4a2b8a0, 0x92946891, 0x533b1036, 0x4f88e856, 0x9e07a80c, 0x555cb05b, 0x5aec3e75, 0x4cbcbaf8 },
	{ 0x993bbbe3, 0x28acae64, 0x7b9487f3, 0x6db334dc, 0xd6f4da75, 0x50a5346c, 0x5d1c6b72, 0x71db28b8 },
	// Round 28 (Function0)
	{ 0xf2e261f8, 0xf1bcac1c, 0x2a518d10, 0xa23fce43, 0x3364dbe3, 0x3cd1bb67, 0xfc75dd59, 0xb043e802 },
	{ 0xca5b0a33, 0xc3943b92, 0x75a12988, 0x1e4d790e, 0x4d19347f, 0xd7757479, 0x5c5316b4, 0x3fafeeb6 },
	{ 0xf7d4a8ea, 0x5324a326, 0x21391abe, 0xd23c32ba, 0x097ef45c, 0x4a17a344, 0x5127234c, 0xadd5a66d },
	{ 0xa63e1db5, 0xa17cf84c, 0x08c9f2af, 0x4d608672, 0x983d5983, 0xcc3ee246, 0x563c6b91, 0xf6c76e08 },
	{ 0xb333982f, 0xe8b6f406, 0x5e76bcb1, 0x36d4c1be, 0xa566d62b, 0x1582ee74, 0x2ae6c4ef, 0x6321efbc },
	{ 0x0d4ec1fd, 0x1614c17e, 0x69c953f4, 0x16fae006, 0xc45a7da7, 0x3daf907e, 0x26585806, 0x3f9d6328 },
	{ 0xe3f2c9d2, 0x16512a74, 0x0cd29b00, 0x9832e0f2, 0x30ceaa5f, 0xd830eb0d, 0x300cd4b7, 0x9af8cee3 },
	// Round 35 (Function0)
	{ 0x7b9ec54b, 0x574d239b, 0x9279f1b5, 0x316796e6, 0x6ee651ff, 0xf3a6e6cc, 0xd3688604, 0x05750a17 },
	{ 0xd98176b1, 0xb3cb2bf4, 0xce6c3213, 0x47154778, 0x8452173c, 0x825446ff, 0x62a205f8, 0x486a9323 },
	{ 0x0758df38, 0x442e7031, 0x65655e4e, 0x86ca0bd0, 0x897cfcf2, 0xa20940f0, 0x8e5086fc, 0x4e477830 },
	{ 0x39eea065, 0x26b29721, 0x8338f7d1, 0x6ff81301, 0x37e95ef7, 0xd1ed44a3, 0xbd3a2ce4, 0xe7de9fef },
	{ 0x15dfa08b, 0x7ceca7d8, 0xd9922576, 0x7eb027ab, 0xf6f7853c, 0xda7d8d53, 0xbe42dc12, 0xdea83eaa },
	{ 0x93ce25aa, 0xdaef5fc0, 0xd86902bd, 0xa5194a17, 0xfd43f65a, 0x33664d97, 0xf908731a, 0x6a21fd4c },
	{ 0x3198b435, 0xa163d09a, 0x701541db, 0x72409751, 0xbb0f1eea, 0xbf9d75f6, 0x9b54cded, 0xe26f4791 }
	// 42 rounds...
};

/*swapping bits 32i||32i+1||......||32i+15 with bits 32i+16||32i+17||......||32i+31 of 32-bit x*/
//#define SWAP16(x)  (x) = ((((x) & 0x0000ffffUL) << 16) | (((x) & 0xffff0000UL) >> 16));
#define SWAP16(x) (x) = __byte_perm(x, 0, 0x1032);

/*swapping bits 16i||16i+1||......||16i+7  with bits 16i+8||16i+9||......||16i+15 of 32-bit x*/
//#define SWAP8(x)   (x) = ((((x) & 0x00ff00ffUL) << 8) | (((x) & 0xff00ff00UL) >> 8));
#define SWAP8(x) (x) = __byte_perm(x, 0, 0x2301);

/*
__device__ __forceinline__
static void SWAP4(uint32_t &x) {
	uint32_t y = x & 0xF0F0F0F0;
	x = (x ^ y) << 4;
	x |= y >> 4;
}
__device__ __forceinline__
static void SWAP2(uint32_t &x) {
	uint32_t y = (x & 0xCCCCCCCC);
	x = (x ^ y) << 2;
	x |= y >> 2;
}
__device__ __forceinline__
static void SWAP1(uint32_t &x) {
	uint32_t y = (x & 0xAAAAAAAA);
	x = (x ^ y) << 1;
	x |= y >> 1;
}
*/

__device__ __forceinline__
static void SWAP4x4(uint32_t *x) {
	#pragma nounroll
	// y is used as tmp register too
	for (uint32_t y=0; y<4; y++, ++x) {
		asm("and.b32 %1, %0, 0xF0F0F0F0;\n\t"
		"xor.b32 %0, %0, %1; shr.b32 %1, %1, 4;\n\t"
		"vshl.u32.u32.u32.clamp.add %0, %0, 4, %1;"
		: "+r"(*x) : "r"(y));
	}
}

__device__ __forceinline__
static void SWAP2x4(uint32_t *x) {
	#pragma nounroll
	// y is used as tmp register too
	for (uint32_t y=0; y<4; y++, ++x) {
		asm("and.b32 %1, %0, 0xCCCCCCCC;\n\t"
		"xor.b32 %0, %0, %1; shr.b32 %1, %1, 2; \n\t"
		"vshl.u32.u32.u32.clamp.add %0, %0, 2, %1;"
		: "+r"(*x) : "r"(y));
	}
}

__device__ __forceinline__
static void SWAP1x4(uint32_t *x) {
	#pragma nounroll
	// y is used as tmp register too
	for (uint32_t y=0; y<4; y++, ++x) {
		asm("and.b32 %1, %0, 0xAAAAAAAA;\n\t"
		"xor.b32 %0, %0, %1; shr.b32 %1, %1, 1; \n\t"
		"vshl.u32.u32.u32.clamp.add %0, %0, 1, %1;"
		: "+r"(*x) : "r"(y));
	}
}

/* The MDS transform */
#define L(m0,m1,m2,m3,m4,m5,m6,m7) \
      m4 ^= m1;                    \
      m5 ^= m2;                    \
      m6 ^= m0 ^ m3;               \
      m7 ^= m0;                    \
      m0 ^= m5;                    \
      m1 ^= m6;                    \
      m2 ^= m4 ^ m7;               \
      m3 ^= m4;

/* The Sbox */
#define Sbox(m0, m1, m2, m3, cc)   \
      m3  = ~(m3);                 \
      m0 ^= (~(m2)) & cc;          \
      temp0 = cc ^ (m0 & m1);      \
      m0 ^= m2 & m3;               \
      m3 ^= (~(m1)) & m2;          \
      m1 ^= m0 & m2;               \
      m2 ^= m0 & (~(m3));          \
      m0 ^= m1 | m3;               \
      m3 ^= m1 & m2;               \
      m1 ^= temp0 & m0;            \
      m2 ^= temp0;

__device__ __forceinline__
static void Sbox_and_MDS_layer(uint32_t x[8][4], const int rnd)
{
	uint2* cc = (uint2*) &c_E8_bslice32[rnd];

	// Sbox and MDS layer
	#pragma unroll
	for (int i = 0; i < 4; i++, ++cc) {
		uint32_t temp0;
		Sbox(x[0][i], x[2][i], x[4][i], x[6][i], cc->x);
		Sbox(x[1][i], x[3][i], x[5][i], x[7][i], cc->y);
		L(x[0][i], x[2][i], x[4][i], x[6][i], x[1][i], x[3][i], x[5][i], x[7][i]);
	}
}

__device__ __forceinline__
static void RoundFunction0(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 0); // 0, 7, 14 .. 35
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) { // 1, 3, 5, 7 (Even)
		SWAP1x4(x[j]);
		// SWAP1(x[j][0]); SWAP1(x[j][1]); SWAP1(x[j][2]); SWAP1(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction1(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 1);

	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) {
		SWAP2x4(x[j]);
		// SWAP2(x[j][0]); SWAP2(x[j][1]); SWAP2(x[j][2]); SWAP2(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction2(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 2);

	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) {
		SWAP4x4(x[j]);
		// SWAP4(x[j][0]); SWAP4(x[j][1]); SWAP4(x[j][2]); SWAP4(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction3(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 3);

	//uint32_t* xj = x[j];
	#pragma unroll 4
	for (int j = 1; j < 8; j += 2) {
		SWAP8(x[j][0]);
		SWAP8(x[j][1]);
		SWAP8(x[j][2]);
		SWAP8(x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction4(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 4);

	#pragma unroll 4
	for (int j = 1; j < 8; j += 2)
	{
		//uint32_t* xj = x[j];
		#pragma unroll
		for (int i = 0; i < 4; i++)
			SWAP16(x[j][i]);
	}
}

__device__ __forceinline__
static void RoundFunction5(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 5);

	#pragma unroll 4
	for (int j = 1; j < 8; j += 2)
	{
		xchg(x[j][0], x[j][1]);
		xchg(x[j][2], x[j][3]);
	}
}

__device__ __forceinline__
static void RoundFunction6(uint32_t x[8][4], const int rnd)
{
	Sbox_and_MDS_layer(x, rnd + 6);

	#pragma unroll 4
	for (int j = 1; j < 8; j += 2)
	{
		xchg(x[j][0], x[j][2]);
		xchg(x[j][1], x[j][3]);
	}
}

/* The bijective function E8, in bitslice form */
__device__
static void E8(uint32_t x[8][4])
{
	/* perform 6 loops of 7 rounds */
	for (int r = 0; r < 42; r += 7)
	{
		RoundFunction0(x, r);
		RoundFunction1(x, r);
		RoundFunction2(x, r);
		RoundFunction3(x, r);
		RoundFunction4(x, r);
		RoundFunction5(x, r);
		RoundFunction6(x, r);
	}
}

__global__
//__launch_bounds__(256,2)
void quark_jh512_gpu_hash_64(const uint32_t threads, const uint32_t startNounce, uint32_t* g_hash, uint32_t * g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);
		const uint32_t hashPosition = nounce - startNounce;
		uint32_t *Hash = &g_hash[(size_t)16 * hashPosition];

		uint32_t h[16];
		AS_UINT4(&h[ 0]) = AS_UINT4(&Hash[ 0]);
		AS_UINT4(&h[ 4]) = AS_UINT4(&Hash[ 4]);
		AS_UINT4(&h[ 8]) = AS_UINT4(&Hash[ 8]);
		AS_UINT4(&h[12]) = AS_UINT4(&Hash[12]);

		uint32_t x[8][4] = { /* init */
			{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a },
			{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2 },
			{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea },
			{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba },
			{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e },
			{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d },
			{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657 },
			{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc }
		};

		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i/4][i & 3] ^= h[i];

		E8(x);

		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[(i+16)/4][(i+16) & 3] ^= h[i];

		x[0][0] ^= 0x80U;
		x[3][3] ^= 0x00020000U;

		E8(x);

		x[4][0] ^= 0x80U;
		x[7][3] ^= 0x00020000U;

		AS_UINT4(&Hash[ 0]) = AS_UINT4(&x[4][0]);
		AS_UINT4(&Hash[ 4]) = AS_UINT4(&x[5][0]);
		AS_UINT4(&Hash[ 8]) = AS_UINT4(&x[6][0]);
		AS_UINT4(&Hash[12]) = AS_UINT4(&x[7][0]);
	}
}

__host__
void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	quark_jh512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
}

// Setup function
__host__ void  quark_jh512_cpu_init(int thr_id, uint32_t threads) {}

#define WANT_JH80_MIDSTATE
#ifdef WANT_JH80

__constant__
static uint32_t c_PaddedMessage80[20]; // padded message (80 bytes)

__host__
void jh512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, endiandata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
void jh512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t h[20];
		AS_UINT4(&h[ 0]) = AS_UINT4(&c_PaddedMessage80[ 0]);
		AS_UINT4(&h[ 4]) = AS_UINT4(&c_PaddedMessage80[ 4]);
		AS_UINT4(&h[ 8]) = AS_UINT4(&c_PaddedMessage80[ 8]);
		AS_UINT4(&h[12]) = AS_UINT4(&c_PaddedMessage80[12]);
		AS_UINT2(&h[16]) = AS_UINT2(&c_PaddedMessage80[16]);
		h[18] = c_PaddedMessage80[18];
		h[19] = cuda_swab32(startNounce + thread);

		uint32_t x[8][4] = { /* init */
			{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a },
			{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2 },
			{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea },
			{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba },
			{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e },
			{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d },
			{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657 },
			{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc }
		};

		// 1 (could be precomputed)
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[i/4][i & 3] ^= h[i];
		E8(x);
		#pragma unroll
		for (int i = 0; i < 16; i++)
			x[(i+16)/4][(i+16) & 3] ^= h[i];

		// 2 (16 bytes with nonce)
		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[0][i] ^= h[16+i];
		x[1][0] ^= 0x80U;
		E8(x);
		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[4][i] ^= h[16+i];
		x[5][0] ^= 0x80U;

		// 3 close
		x[3][3] ^= 0x80020000U; // 80 bytes = 640bits (0x280)
		E8(x);
		x[7][3] ^= 0x80020000U;

		uint32_t *Hash = &g_outhash[(size_t)16 * thread];
		AS_UINT4(&Hash[ 0]) = AS_UINT4(&x[4][0]);
		AS_UINT4(&Hash[ 4]) = AS_UINT4(&x[5][0]);
		AS_UINT4(&Hash[ 8]) = AS_UINT4(&x[6][0]);
		AS_UINT4(&Hash[12]) = AS_UINT4(&x[7][0]);
	}
}

__host__
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	jh512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, d_hash);
}

#endif

#ifdef WANT_JH80_MIDSTATE

__constant__ static uint32_t c_JHState[32];
__constant__ static uint32_t c_Message[4];

__global__
void jh512_gpu_hash_80(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// 1 (precomputed state)
		uint32_t x[8][4];
		AS_UINT4(&x[0][0]) = AS_UINT4(&c_JHState[ 0]);
		AS_UINT4(&x[1][0]) = AS_UINT4(&c_JHState[ 4]);
		AS_UINT4(&x[2][0]) = AS_UINT4(&c_JHState[ 8]);
		AS_UINT4(&x[3][0]) = AS_UINT4(&c_JHState[12]);

		AS_UINT4(&x[4][0]) = AS_UINT4(&c_JHState[16]);
		AS_UINT4(&x[5][0]) = AS_UINT4(&c_JHState[20]);
		AS_UINT4(&x[6][0]) = AS_UINT4(&c_JHState[24]);
		AS_UINT4(&x[7][0]) = AS_UINT4(&c_JHState[28]);

		// 2 (16 bytes with nonce)
		uint32_t h[4];
		AS_UINT2(&h[0]) = AS_UINT2(&c_Message[0]);
		h[2] = c_Message[2];
		h[3] = cuda_swab32(startNounce + thread);

		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[0][i] ^= h[i];
		x[1][0] ^= 0x80U;
		E8(x);
		#pragma unroll
		for (int i = 0; i < 4; i++)
			x[4][i] ^= h[i];
		x[5][0] ^= 0x80U;

		// 3 close
		x[3][3] ^= 0x80020000U; // 80 bytes = 640bits (0x280)
		E8(x);
		x[7][3] ^= 0x80020000U;

		uint32_t *Hash = &g_outhash[(size_t)16 * thread];
		AS_UINT4(&Hash[ 0]) = AS_UINT4(&x[4][0]);
		AS_UINT4(&Hash[ 4]) = AS_UINT4(&x[5][0]);
		AS_UINT4(&Hash[ 8]) = AS_UINT4(&x[6][0]);
		AS_UINT4(&Hash[12]) = AS_UINT4(&x[7][0]);
	}
}

__host__
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	jh512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, d_hash);
}

extern "C" {
#undef SPH_C32
#undef SPH_T32
#undef SPH_C64
#undef SPH_T64
#include <sph/sph_jh.h>
}

__host__
void jh512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	sph_jh512_context ctx_jh;

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, endiandata, 64);

	cudaMemcpyToSymbol(c_JHState, ctx_jh.H.narrow, 128, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_Message, &endiandata[16], sizeof(c_Message), 0, cudaMemcpyHostToDevice);
}

#endif
