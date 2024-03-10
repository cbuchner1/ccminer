/**
 * sha-512 CUDA implementation.
 * Tanguy Pruvot and Provos Alexis - Jul / Sep 2016
 * Sponsored by LBRY.IO team
 */

//#define USE_ROT_ASM_OPT 0
#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>

#include <miner.h>

static __constant__ _ALIGN(8) uint64_t K_512[80] = {
	0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
	0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
	0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
	0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
	0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
	0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
	0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
	0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
	0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
	0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
	0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
	0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
	0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
	0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
	0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
	0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
	0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
	0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
	0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
	0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817
};

#undef xor3
#define xor3(a,b,c) (a^b^c)

//#define ROR64_8(x) ROTR64(x,8)
__device__ __inline__
uint64_t ROR64_8(const uint64_t u64) {
	const uint2 a = vectorize(u64);
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x0765);
	result.y = __byte_perm(a.y, a.x, 0x4321);
	return devectorize(result);
}

#define bsg5_0(x) xor3(ROTR64(x,28),ROTR64(x,34),ROTR64(x,39))
#define bsg5_1(x) xor3(ROTR64(x,14),ROTR64(x,18),ROTR64(x,41))
#define ssg5_0(x) xor3(ROTR64(x,1), ROR64_8(x), x>>7)
#define ssg5_1(x) xor3(ROTR64(x,19),ROTR64(x,61), x>>6)

#define andor64(a,b,c) ((a & (b | c)) | (b & c))
#define xandx64(e,f,g) (g ^ (e & (g ^ f)))

__device__ __forceinline__
static void sha512_step2(uint64_t *const r,const uint64_t W,const uint64_t K, const int ord)
{
	const uint64_t T1 = r[(15-ord) & 7] + K + W + bsg5_1(r[(12-ord) & 7]) + xandx64(r[(12-ord) & 7],r[(13-ord) & 7],r[(14-ord) & 7]);
	r[(15-ord) & 7] = andor64(r[(8-ord) & 7],r[(9-ord) & 7],r[(10-ord) & 7]) + bsg5_0(r[(8-ord) & 7]) + T1;
	r[(11-ord) & 7]+= T1;
}

/**************************************************************************************************/

__global__
#if CUDA_VERSION > 6050
__launch_bounds__(512,2)
#endif
void lbry_sha512_gpu_hash_32(const uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint64_t IV512[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};
	uint64_t r[8];
	uint64_t W[16];
	if (thread < threads)
	{
		uint64_t *pHash = &g_hash[thread<<3];

		*(uint2x4*)&r[0] = *(uint2x4*)&IV512[0];
		*(uint2x4*)&r[4] = *(uint2x4*)&IV512[4];

		*(uint2x4*)&W[0] = __ldg4((uint2x4*)pHash);

		W[4] = 0x8000000000000000; // end tag

		#pragma unroll
		for (uint32_t i = 5; i < 15; i++) W[i] = 0;

		W[15] = 0x100; // 256 bits

		#pragma unroll 16
		for (int i = 0; i < 16; i ++){
			sha512_step2(r, W[i], K_512[i], i&7);
		}

		#pragma unroll 5
		for (uint32_t i = 16; i < 80; i+=16){
			#pragma unroll
			for (uint32_t j = 0; j<16; j++){
				W[(i + j) & 15] += W[((i + j) - 7) & 15] + ssg5_0(W[((i + j) - 15) & 15]) + ssg5_1(W[((i + j) - 2) & 15]);
			}
			#pragma unroll
			for (uint32_t j = 0; j<16; j++){
				sha512_step2(r, W[j], K_512[i+j], (i+j)&7);
			}
		}

		#pragma unroll 8
		for (uint32_t i = 0; i < 8; i++)
			r[i] = cuda_swab64(r[i] + IV512[i]);

		*(uint2x4*)&pHash[0] = *(uint2x4*)&r[0];
		*(uint2x4*)&pHash[4] = *(uint2x4*)&r[4];

	}
}

__host__
void lbry_sha512_hash_32(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha512_gpu_hash_32 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}

/**************************************************************************************************/

__host__
void lbry_sha512_init(int thr_id)
{
//	cudaMemcpyToSymbol(K_512, K512, 80*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}
