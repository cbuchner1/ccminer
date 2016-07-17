/*
 * sha256 + ripemd CUDA implementation.
 * tpruvot and alexis78
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>

#include <miner.h>

__constant__ static uint32_t _ALIGN(8) c_midstate112[8];
__constant__ static uint32_t _ALIGN(8) c_midbuffer112[8];
__constant__ static uint32_t _ALIGN(8) c_dataEnd112[12];

__constant__  const uint32_t c_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};
__constant__ static uint32_t _ALIGN(16) c_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

static __thread uint32_t* d_resNonces;
__constant__ static uint32_t _ALIGN(8) c_target[2];
__device__ uint64_t d_target[1];

#ifdef __INTELLISENSE__
#define atomicExch(p,y) y
#endif

// ------------------------------------------------------------------------------------------------

static const uint32_t cpu_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};

static const uint32_t cpu_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

__host__
static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h, uint32_t in, const uint32_t Kshared)
{
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
	uint32_t t1 = h + bsg21 + vxandx + Kshared + in;
	uint32_t t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_step2_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h, uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	int pcidx1 = (pc-2)  & 0xF;
	int pcidx2 = (pc-7)  & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ROTR32(inx1, 17) ^ ROTR32(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	uint32_t ssg20 = ROTR32(inx3, 7) ^ ROTR32(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);
	uint32_t t1,t2;

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__host__
static void sha256_round_body_host(uint32_t* in, uint32_t* state, const uint32_t* Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha256_step1_host(a,b,c,d,e,f,g,h,in[0], Kshared[0]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[1], Kshared[1]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[2], Kshared[2]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[3], Kshared[3]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[4], Kshared[4]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[5], Kshared[5]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[6], Kshared[6]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[7], Kshared[7]);
	sha256_step1_host(a,b,c,d,e,f,g,h,in[8], Kshared[8]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[9], Kshared[9]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	for (int i=0; i<3; i++)
	{
		sha256_step2_host(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha256_step2_host(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

#define xor3b(a,b,c) ((a ^ b) ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

__device__ __forceinline__ uint32_t andor32(const uint32_t a, const uint32_t b, const uint32_t c)
{
	uint32_t result;
	asm("{ .reg .u32 m,n,o; // andor32 \n\t"
		"and.b32 m,  %1, %2;\n\t"
		" or.b32 n,  %1, %2;\n\t"
		"and.b32 o,   n, %3;\n\t"
		" or.b32 %0,  m, o ;\n\t"
		"}\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(c)
	);
	return result;
}

__device__ __forceinline__ uint2 vectorizeswap(uint64_t v)
{
	uint2 result;
	asm("mov.b64 {%0,%1},%2; // vectorizeswap \n\t"
		: "=r"(result.y), "=r"(result.x) : "l"(v));
	return result;
}

__device__
__forceinline__
static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h, uint32_t in, const uint32_t Kshared)
{
	uint32_t t1 = bsg2_1(e) + ((((f) ^ (g)) & (e)) ^ (g)) + Kshared + in;
	d = d + h + t1;
	h += t1 + bsg2_0(a) + (((b) & (c)) | (((b) | (c)) & (a)));
}

__device__
static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1,t2;

	int pcidx1 = (pc-2) & 0xF;
	int pcidx2 = (pc-7) & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ssg2_1(inx1);
	uint32_t ssg20 = ssg2_0(inx3);
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a,b,c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__device__ __forceinline__
static void sha256_round_first(uint32_t* in,uint32_t *buf, uint32_t* state, uint32_t* const Kshared)
{
	uint32_t a = buf[0];
	uint32_t b = buf[1];
	uint32_t c = buf[2];
	uint32_t d = buf[3];
	uint32_t e = buf[4];
	uint32_t f = buf[5];
	uint32_t g = buf[6];
	uint32_t h = buf[7];

	// 10 first steps made on host
	sha2_step1(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
	sha2_step1(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha2_step1(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha2_step1(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha2_step1(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	#pragma unroll
	for (int i=0; i<3; i++)
	{
		sha2_step2(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha2_step2(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
	}

	buf[ 0] = state[0] + a;
	buf[ 1] = state[1] + b;
	buf[ 2] = state[2] + c;
	buf[ 3] = state[3] + d;
	buf[ 4] = state[4] + e;
	buf[ 5] = state[5] + f;
	buf[ 6] = state[6] + g;
	buf[ 7] = state[7] + h;
}

__device__
static void sha256_round_body(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step1(a,b,c,d,e,f,g,h,in[0], Kshared[0]);
	sha2_step1(h,a,b,c,d,e,f,g,in[1], Kshared[1]);
	sha2_step1(g,h,a,b,c,d,e,f,in[2], Kshared[2]);
	sha2_step1(f,g,h,a,b,c,d,e,in[3], Kshared[3]);
	sha2_step1(e,f,g,h,a,b,c,d,in[4], Kshared[4]);
	sha2_step1(d,e,f,g,h,a,b,c,in[5], Kshared[5]);
	sha2_step1(c,d,e,f,g,h,a,b,in[6], Kshared[6]);
	sha2_step1(b,c,d,e,f,g,h,a,in[7], Kshared[7]);
	sha2_step1(a,b,c,d,e,f,g,h,in[8], Kshared[8]);
	sha2_step1(h,a,b,c,d,e,f,g,in[9], Kshared[9]);
	sha2_step1(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
	sha2_step1(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
	sha2_step1(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha2_step1(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha2_step1(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha2_step1(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	#pragma unroll
	for (int i=0; i<3; i++)
	{
		sha2_step2(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha2_step2(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

__device__
uint64_t cuda_swab32ll(uint64_t x) {
	return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}

__global__
__launch_bounds__(512,2) /* to force 64 regs */
void lbry_sha256d_gpu_hash_112(const uint32_t threads, const uint32_t startNonce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	extern __shared__ uint32_t s_K[];
	if (threadIdx.x < 64U) s_K[threadIdx.x] = c_K[threadIdx.x];
	//__threadfence_block();
	if (thread < threads)
	{
		uint32_t dat[16];
		#pragma unroll
		for (int i=0; i<11; i++) dat[i] = c_dataEnd112[i];
		dat[11] = startNonce + thread;
		dat[12] = 0x80000000;
		dat[13] = 0;
		dat[14] = 0;
		dat[15] = 0x380;

		uint32_t __align__(8) buf[8], state[8];

		*(uint2x4*)&state[0] = *(uint2x4*)&c_midstate112[0];
		*(uint2x4*)&buf[0]   = *(uint2x4*)&c_midbuffer112[0];

		sha256_round_first(dat, buf, state, c_K); // no shared mem here

		// second sha256

		*(uint2x4*)&dat[0] = *(uint2x4*)&buf[0];

		dat[8] = 0x80000000;

		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		*(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];

		sha256_round_body(dat, buf, s_K);

		// output
		*(uint2*)&buf[0] = vectorizeswap(((uint64_t*)buf)[0]);
		*(uint2*)&buf[2] = vectorizeswap(((uint64_t*)buf)[1]);
		*(uint2*)&buf[4] = vectorizeswap(((uint64_t*)buf)[2]);
		*(uint2*)&buf[6] = vectorizeswap(((uint64_t*)buf)[3]);

		*(uint2x4*)&outputHash[thread*8U] = *(uint2x4*)&buf[0];
	}
}

__host__
void lbry_sha256d_hash_112(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash)
{
	const int threadsperblock = 512;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256d_gpu_hash_112 <<<grid, block, 64*4>>> (threads, startNonce, (uint64_t*) d_outputHash);
}

__host__
void lbry_sha256_init(int thr_id)
{
	cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces, 4*sizeof(uint32_t)));
}

__host__
void lbry_sha256_free(int thr_id)
{
	cudaFree(d_resNonces);
}

__host__
void lbry_sha256_setBlock_112(uint32_t *pdata, uint32_t *ptarget)
{
	uint32_t in[16], buf[8], end[11];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0; i<8;i++) buf[i] = cpu_H256[i];
	for (int i=0;i<11;i++) end[i] = cuda_swab32(pdata[16+i]);
	sha256_round_body_host(in, buf, cpu_K);

	cudaMemcpyToSymbol(c_midstate112, buf, 32, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_dataEnd112,  end, sizeof(end), 0, cudaMemcpyHostToDevice);

	uint32_t a = buf[0];
	uint32_t b = buf[1];
	uint32_t c = buf[2];
	uint32_t d = buf[3];
	uint32_t e = buf[4];
	uint32_t f = buf[5];
	uint32_t g = buf[6];
	uint32_t h = buf[7];

	sha256_step1_host(a,b,c,d,e,f,g,h,end[0], cpu_K[0]);
	sha256_step1_host(h,a,b,c,d,e,f,g,end[1], cpu_K[1]);
	sha256_step1_host(g,h,a,b,c,d,e,f,end[2], cpu_K[2]);
	sha256_step1_host(f,g,h,a,b,c,d,e,end[3], cpu_K[3]);
	sha256_step1_host(e,f,g,h,a,b,c,d,end[4], cpu_K[4]);
	sha256_step1_host(d,e,f,g,h,a,b,c,end[5], cpu_K[5]);
	sha256_step1_host(c,d,e,f,g,h,a,b,end[6], cpu_K[6]);
	sha256_step1_host(b,c,d,e,f,g,h,a,end[7], cpu_K[7]);
	sha256_step1_host(a,b,c,d,e,f,g,h,end[8], cpu_K[8]);
	sha256_step1_host(h,a,b,c,d,e,f,g,end[9], cpu_K[9]);
	sha256_step1_host(g,h,a,b,c,d,e,f,end[10],cpu_K[10]);

	buf[0] = a;
	buf[1] = b;
	buf[2] = c;
	buf[3] = d;
	buf[4] = e;
	buf[5] = f;
	buf[6] = g;
	buf[7] = h;

	cudaMemcpyToSymbol(c_midbuffer112, buf, 32, 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_target, &ptarget[6], sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_target, &ptarget[6], sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

// ------------------------------------------------------------------------------------------

static __constant__ uint32_t c_IV[5] = {
	0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u
};

/*
 * Round functions for RIPEMD-160.
 */
#if 1
#define F1(x, y, z)   ((x) ^ (y) ^ (z))
#define F2(x, y, z)   ((((y) ^ (z)) & (x)) ^ (z))
#define F3(x, y, z)   (((x) | ~(y)) ^ (z))
#define F4(x, y, z)   ((((x) ^ (y)) & (z)) ^ (y))
#define F5(x, y, z)   ((x) ^ ((y) | ~(z)))
#else
#define F1(x, y, z)   xor3b(x,y,z)
#define F2(x, y, z)   xandx(x,y,z)
#define F3(x, y, z)   xornot64(x,y,z)
#define F4(x, y, z)   xandx(z,x,y)
#define F5(x, y, z)   xornt64(x,y,z)
#endif

/*
 * Round constants for RIPEMD-160.
 */
#define K11 0x00000000u
#define K12 0x5A827999u
#define K13 0x6ED9EBA1u
#define K14 0x8F1BBCDCu
#define K15 0xA953FD4Eu

#define K21 0x50A28BE6u
#define K22 0x5C4DD124u
#define K23 0x6D703EF3u
#define K24 0x7A6D76E9u
#define K25 0x00000000u

#define RR(a, b, c, d, e, f, s, r, k) { \
	a = SPH_T32(ROTL32(SPH_T32(a + f(b, c, d) + r + k), s) + e); \
	c = ROTL32(c, 10); \
}

#define ROUND1(a, b, c, d, e, f, s, r, k) \
	RR(a ## 1, b ## 1, c ## 1, d ## 1, e ## 1, f, s, r, K1 ## k)

#define ROUND2(a, b, c, d, e, f, s, r, k) \
	RR(a ## 2, b ## 2, c ## 2, d ## 2, e ## 2, f, s, r, K2 ## k)

#define RIPEMD160_ROUND_BODY(in, h) { \
	uint32_t A1, B1, C1, D1, E1; \
	uint32_t A2, B2, C2, D2, E2; \
	uint32_t tmp; \
\
	A1 = A2 = h[0]; \
	B1 = B2 = h[1]; \
	C1 = C2 = h[2]; \
	D1 = D2 = h[3]; \
	E1 = E2 = h[4]; \
\
	ROUND1(A, B, C, D, E, F1, 11, in[ 0],  1); \
	ROUND1(E, A, B, C, D, F1, 14, in[ 1],  1); \
	ROUND1(D, E, A, B, C, F1, 15, in[ 2],  1); \
	ROUND1(C, D, E, A, B, F1, 12, in[ 3],  1); \
	ROUND1(B, C, D, E, A, F1,  5, in[ 4],  1); \
	ROUND1(A, B, C, D, E, F1,  8, in[ 5],  1); \
	ROUND1(E, A, B, C, D, F1,  7, in[ 6],  1); \
	ROUND1(D, E, A, B, C, F1,  9, in[ 7],  1); \
	ROUND1(C, D, E, A, B, F1, 11, in[ 8],  1); \
	ROUND1(B, C, D, E, A, F1, 13, in[ 9],  1); \
	ROUND1(A, B, C, D, E, F1, 14, in[10],  1); \
	ROUND1(E, A, B, C, D, F1, 15, in[11],  1); \
	ROUND1(D, E, A, B, C, F1,  6, in[12],  1); \
	ROUND1(C, D, E, A, B, F1,  7, in[13],  1); \
	ROUND1(B, C, D, E, A, F1,  9, in[14],  1); \
	ROUND1(A, B, C, D, E, F1,  8, in[15],  1); \
\
	ROUND1(E, A, B, C, D, F2,  7, in[ 7],  2); \
	ROUND1(D, E, A, B, C, F2,  6, in[ 4],  2); \
	ROUND1(C, D, E, A, B, F2,  8, in[13],  2); \
	ROUND1(B, C, D, E, A, F2, 13, in[ 1],  2); \
	ROUND1(A, B, C, D, E, F2, 11, in[10],  2); \
	ROUND1(E, A, B, C, D, F2,  9, in[ 6],  2); \
	ROUND1(D, E, A, B, C, F2,  7, in[15],  2); \
	ROUND1(C, D, E, A, B, F2, 15, in[ 3],  2); \
	ROUND1(B, C, D, E, A, F2,  7, in[12],  2); \
	ROUND1(A, B, C, D, E, F2, 12, in[ 0],  2); \
	ROUND1(E, A, B, C, D, F2, 15, in[ 9],  2); \
	ROUND1(D, E, A, B, C, F2,  9, in[ 5],  2); \
	ROUND1(C, D, E, A, B, F2, 11, in[ 2],  2); \
	ROUND1(B, C, D, E, A, F2,  7, in[14],  2); \
	ROUND1(A, B, C, D, E, F2, 13, in[11],  2); \
	ROUND1(E, A, B, C, D, F2, 12, in[ 8],  2); \
\
	ROUND1(D, E, A, B, C, F3, 11, in[ 3],  3); \
	ROUND1(C, D, E, A, B, F3, 13, in[10],  3); \
	ROUND1(B, C, D, E, A, F3,  6, in[14],  3); \
	ROUND1(A, B, C, D, E, F3,  7, in[ 4],  3); \
	ROUND1(E, A, B, C, D, F3, 14, in[ 9],  3); \
	ROUND1(D, E, A, B, C, F3,  9, in[15],  3); \
	ROUND1(C, D, E, A, B, F3, 13, in[ 8],  3); \
	ROUND1(B, C, D, E, A, F3, 15, in[ 1],  3); \
	ROUND1(A, B, C, D, E, F3, 14, in[ 2],  3); \
	ROUND1(E, A, B, C, D, F3,  8, in[ 7],  3); \
	ROUND1(D, E, A, B, C, F3, 13, in[ 0],  3); \
	ROUND1(C, D, E, A, B, F3,  6, in[ 6],  3); \
	ROUND1(B, C, D, E, A, F3,  5, in[13],  3); \
	ROUND1(A, B, C, D, E, F3, 12, in[11],  3); \
	ROUND1(E, A, B, C, D, F3,  7, in[ 5],  3); \
	ROUND1(D, E, A, B, C, F3,  5, in[12],  3); \
\
	ROUND1(C, D, E, A, B, F4, 11, in[ 1],  4); \
	ROUND1(B, C, D, E, A, F4, 12, in[ 9],  4); \
	ROUND1(A, B, C, D, E, F4, 14, in[11],  4); \
	ROUND1(E, A, B, C, D, F4, 15, in[10],  4); \
	ROUND1(D, E, A, B, C, F4, 14, in[ 0],  4); \
	ROUND1(C, D, E, A, B, F4, 15, in[ 8],  4); \
	ROUND1(B, C, D, E, A, F4,  9, in[12],  4); \
	ROUND1(A, B, C, D, E, F4,  8, in[ 4],  4); \
	ROUND1(E, A, B, C, D, F4,  9, in[13],  4); \
	ROUND1(D, E, A, B, C, F4, 14, in[ 3],  4); \
	ROUND1(C, D, E, A, B, F4,  5, in[ 7],  4); \
	ROUND1(B, C, D, E, A, F4,  6, in[15],  4); \
	ROUND1(A, B, C, D, E, F4,  8, in[14],  4); \
	ROUND1(E, A, B, C, D, F4,  6, in[ 5],  4); \
	ROUND1(D, E, A, B, C, F4,  5, in[ 6],  4); \
	ROUND1(C, D, E, A, B, F4, 12, in[ 2],  4); \
\
	ROUND1(B, C, D, E, A, F5,  9, in[ 4],  5); \
	ROUND1(A, B, C, D, E, F5, 15, in[ 0],  5); \
	ROUND1(E, A, B, C, D, F5,  5, in[ 5],  5); \
	ROUND1(D, E, A, B, C, F5, 11, in[ 9],  5); \
	ROUND1(C, D, E, A, B, F5,  6, in[ 7],  5); \
	ROUND1(B, C, D, E, A, F5,  8, in[12],  5); \
	ROUND1(A, B, C, D, E, F5, 13, in[ 2],  5); \
	ROUND1(E, A, B, C, D, F5, 12, in[10],  5); \
	ROUND1(D, E, A, B, C, F5,  5, in[14],  5); \
	ROUND1(C, D, E, A, B, F5, 12, in[ 1],  5); \
	ROUND1(B, C, D, E, A, F5, 13, in[ 3],  5); \
	ROUND1(A, B, C, D, E, F5, 14, in[ 8],  5); \
	ROUND1(E, A, B, C, D, F5, 11, in[11],  5); \
	ROUND1(D, E, A, B, C, F5,  8, in[ 6],  5); \
	ROUND1(C, D, E, A, B, F5,  5, in[15],  5); \
	ROUND1(B, C, D, E, A, F5,  6, in[13],  5); \
\
	ROUND2(A, B, C, D, E, F5,  8, in[ 5],  1); \
	ROUND2(E, A, B, C, D, F5,  9, in[14],  1); \
	ROUND2(D, E, A, B, C, F5,  9, in[ 7],  1); \
	ROUND2(C, D, E, A, B, F5, 11, in[ 0],  1); \
	ROUND2(B, C, D, E, A, F5, 13, in[ 9],  1); \
	ROUND2(A, B, C, D, E, F5, 15, in[ 2],  1); \
	ROUND2(E, A, B, C, D, F5, 15, in[11],  1); \
	ROUND2(D, E, A, B, C, F5,  5, in[ 4],  1); \
	ROUND2(C, D, E, A, B, F5,  7, in[13],  1); \
	ROUND2(B, C, D, E, A, F5,  7, in[ 6],  1); \
	ROUND2(A, B, C, D, E, F5,  8, in[15],  1); \
	ROUND2(E, A, B, C, D, F5, 11, in[ 8],  1); \
	ROUND2(D, E, A, B, C, F5, 14, in[ 1],  1); \
	ROUND2(C, D, E, A, B, F5, 14, in[10],  1); \
	ROUND2(B, C, D, E, A, F5, 12, in[ 3],  1); \
	ROUND2(A, B, C, D, E, F5,  6, in[12],  1); \
\
	ROUND2(E, A, B, C, D, F4,  9, in[ 6],  2); \
	ROUND2(D, E, A, B, C, F4, 13, in[11],  2); \
	ROUND2(C, D, E, A, B, F4, 15, in[ 3],  2); \
	ROUND2(B, C, D, E, A, F4,  7, in[ 7],  2); \
	ROUND2(A, B, C, D, E, F4, 12, in[ 0],  2); \
	ROUND2(E, A, B, C, D, F4,  8, in[13],  2); \
	ROUND2(D, E, A, B, C, F4,  9, in[ 5],  2); \
	ROUND2(C, D, E, A, B, F4, 11, in[10],  2); \
	ROUND2(B, C, D, E, A, F4,  7, in[14],  2); \
	ROUND2(A, B, C, D, E, F4,  7, in[15],  2); \
	ROUND2(E, A, B, C, D, F4, 12, in[ 8],  2); \
	ROUND2(D, E, A, B, C, F4,  7, in[12],  2); \
	ROUND2(C, D, E, A, B, F4,  6, in[ 4],  2); \
	ROUND2(B, C, D, E, A, F4, 15, in[ 9],  2); \
	ROUND2(A, B, C, D, E, F4, 13, in[ 1],  2); \
	ROUND2(E, A, B, C, D, F4, 11, in[ 2],  2); \
\
	ROUND2(D, E, A, B, C, F3,  9, in[15],  3); \
	ROUND2(C, D, E, A, B, F3,  7, in[ 5],  3); \
	ROUND2(B, C, D, E, A, F3, 15, in[ 1],  3); \
	ROUND2(A, B, C, D, E, F3, 11, in[ 3],  3); \
	ROUND2(E, A, B, C, D, F3,  8, in[ 7],  3); \
	ROUND2(D, E, A, B, C, F3,  6, in[14],  3); \
	ROUND2(C, D, E, A, B, F3,  6, in[ 6],  3); \
	ROUND2(B, C, D, E, A, F3, 14, in[ 9],  3); \
	ROUND2(A, B, C, D, E, F3, 12, in[11],  3); \
	ROUND2(E, A, B, C, D, F3, 13, in[ 8],  3); \
	ROUND2(D, E, A, B, C, F3,  5, in[12],  3); \
	ROUND2(C, D, E, A, B, F3, 14, in[ 2],  3); \
	ROUND2(B, C, D, E, A, F3, 13, in[10],  3); \
	ROUND2(A, B, C, D, E, F3, 13, in[ 0],  3); \
	ROUND2(E, A, B, C, D, F3,  7, in[ 4],  3); \
	ROUND2(D, E, A, B, C, F3,  5, in[13],  3); \
\
	ROUND2(C, D, E, A, B, F2, 15, in[ 8],  4); \
	ROUND2(B, C, D, E, A, F2,  5, in[ 6],  4); \
	ROUND2(A, B, C, D, E, F2,  8, in[ 4],  4); \
	ROUND2(E, A, B, C, D, F2, 11, in[ 1],  4); \
	ROUND2(D, E, A, B, C, F2, 14, in[ 3],  4); \
	ROUND2(C, D, E, A, B, F2, 14, in[11],  4); \
	ROUND2(B, C, D, E, A, F2,  6, in[15],  4); \
	ROUND2(A, B, C, D, E, F2, 14, in[ 0],  4); \
	ROUND2(E, A, B, C, D, F2,  6, in[ 5],  4); \
	ROUND2(D, E, A, B, C, F2,  9, in[12],  4); \
	ROUND2(C, D, E, A, B, F2, 12, in[ 2],  4); \
	ROUND2(B, C, D, E, A, F2,  9, in[13],  4); \
	ROUND2(A, B, C, D, E, F2, 12, in[ 9],  4); \
	ROUND2(E, A, B, C, D, F2,  5, in[ 7],  4); \
	ROUND2(D, E, A, B, C, F2, 15, in[10],  4); \
	ROUND2(C, D, E, A, B, F2,  8, in[14],  4); \
\
	ROUND2(B, C, D, E, A, F1,  8, in[12],  5); \
	ROUND2(A, B, C, D, E, F1,  5, in[15],  5); \
	ROUND2(E, A, B, C, D, F1, 12, in[10],  5); \
	ROUND2(D, E, A, B, C, F1,  9, in[ 4],  5); \
	ROUND2(C, D, E, A, B, F1, 12, in[ 1],  5); \
	ROUND2(B, C, D, E, A, F1,  5, in[ 5],  5); \
	ROUND2(A, B, C, D, E, F1, 14, in[ 8],  5); \
	ROUND2(E, A, B, C, D, F1,  6, in[ 7],  5); \
	ROUND2(D, E, A, B, C, F1,  8, in[ 6],  5); \
	ROUND2(C, D, E, A, B, F1, 13, in[ 2],  5); \
	ROUND2(B, C, D, E, A, F1,  6, in[13],  5); \
	ROUND2(A, B, C, D, E, F1,  5, in[14],  5); \
	ROUND2(E, A, B, C, D, F1, 15, in[ 0],  5); \
	ROUND2(D, E, A, B, C, F1, 13, in[ 3],  5); \
	ROUND2(C, D, E, A, B, F1, 11, in[ 9],  5); \
	ROUND2(B, C, D, E, A, F1, 11, in[11],  5); \
\
	tmp  = (h[1] + C1 + D2); \
	h[1] = (h[2] + D1 + E2); \
	h[2] = (h[3] + E1 + A2); \
	h[3] = (h[4] + A1 + B2); \
	h[4] = (h[0] + B1 + C2); \
	h[0] = tmp; \
}

__global__
__launch_bounds__(640,2) /* 640,2 <= 48 regs, 512,2 <= 64 */
void lbry_sha256d_gpu_hash_final(const uint32_t threads, uint64_t *Hash512, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	extern __shared__ uint32_t s_K[];
	if (threadIdx.x < 64U) s_K[threadIdx.x] = c_K[threadIdx.x];
	//__threadfence_block();
	if (thread < threads)
	{
		uint32_t* input = (uint32_t*) (&Hash512[thread * 8U]);

		uint32_t __align__(8) dat[16];

		*(uint2x4*)&dat[0] = *(uint2x4*)&input[0];

		dat[8] = 0x80;

		#pragma unroll
		for (int i=9;i<16;i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		uint32_t h[5];
		#pragma unroll
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(dat, h);

		uint32_t __align__(8) buf[8]; // align for vectorize
		#pragma unroll
		for (int i=0; i<5; i++)
			buf[i] = h[i];

		// second 32 bytes block hash

		*(uint2x4*)&dat[0] = *(uint2x4*)&input[8];

		dat[8] = 0x80;

		#pragma unroll
		for (int i=9;i<16;i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		#pragma unroll
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(dat, h);

		// first final sha256

		#pragma unroll
		for (int i=0;i<5;i++) dat[i] = cuda_swab32(buf[i]);
		#pragma unroll
		for (int i=0;i<5;i++) dat[i+5] = cuda_swab32(h[i]);
		dat[10] = 0x80000000;
		#pragma unroll
		for (int i=11; i<15; i++) dat[i] = 0;
		dat[15] = 0x140;

		*(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];

		sha256_round_body(dat, buf, c_K); // s_K uses too much regs

		// second sha256

		*(uint2x4*)&dat[0] = *(uint2x4*)&buf[0];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		*(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];

		sha256_round_body(dat, buf, s_K);

		// valid nonces
		const uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
		if (high <= d_target[0]) {
			resNonces[1] = atomicExch(resNonces, thread);
			d_target[0] = high;
		}
	}
}

__host__
void lbry_sha256d_hash_final(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_inputHash, uint32_t *resNonces)
{
	const int threadsperblock = 512;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	cudaMemset(d_resNonces, 0xFF, 2 * sizeof(uint32_t));

	lbry_sha256d_gpu_hash_final <<<grid, block, 64*4>>> (threads, (uint64_t*) d_inputHash, d_resNonces);

	cudaMemcpy(resNonces, d_resNonces, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
	if (resNonces[0] != UINT32_MAX) resNonces[0] += startNonce;
	if (resNonces[1] != UINT32_MAX) resNonces[1] += startNonce;
}
