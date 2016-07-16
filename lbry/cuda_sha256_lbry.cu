/*
 * sha256 CUDA implementation.
 */
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <miner.h>

__constant__ static uint32_t __align__(8) c_midstate112[8];
__constant__ static uint32_t __align__(8) c_dataEnd112[12];

const __constant__  uint32_t __align__(8) c_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};
__constant__ static uint32_t __align__(8) c_K[64];

static __thread uint32_t* d_resNonces;
__constant__ static uint32_t __align__(8) c_target[2];
__device__ uint64_t d_target[1];

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

#define ROTR ROTR32

__host__
static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t in, const uint32_t Kshared)
{
	uint32_t t1,t2;
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_step2_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1,t2;

	int pcidx1 = (pc-2)  & 0xF;
	int pcidx2 = (pc-7)  & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	uint32_t ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

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

__device__ __forceinline__
uint32_t xor3b(const uint32_t a, const uint32_t b, const uint32_t c) {
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0x96; // xor3b"  //0x96 = 0xF0 ^ 0xCC ^ 0xAA
		: "=r"(result) : "r"(a), "r"(b),"r"(c));
#else
	result = a^b^c;
#endif
	return result;
}

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	uint32_t r1 = ROTR32(x,2);
	uint32_t r2 = ROTR32(x,13);
	uint32_t r3 = ROTR32(x,22);
	return xor3b(r1,r2,r3);
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	uint32_t r1 = ROTR32(x,6);
	uint32_t r2 = ROTR32(x,11);
	uint32_t r3 = ROTR32(x,25);
	return xor3b(r1,r2,r3);
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	uint64_t r1 = ROTR32(x,7);
	uint64_t r2 = ROTR32(x,18);
	uint64_t r3 = shr_t32(x,3);
	return xor3b(r1,r2,r3);
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	uint64_t r1 = ROTR32(x,17);
	uint64_t r2 = ROTR32(x,19);
	uint64_t r3 = shr_t32(x,10);
	return xor3b(r1,r2,r3);
}

__device__ __forceinline__ uint32_t andor32(const uint32_t a, const uint32_t b, const uint32_t c)
{
	uint32_t result;
	asm("{\n\t"
		".reg .u32 m,n,o;\n\t"
		"and.b32 m,  %1, %2;\n\t"
		" or.b32 n,  %1, %2;\n\t"
		"and.b32 o,   n, %3;\n\t"
		" or.b32 %0,  m, o ;\n\t"
		"}\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(c)
	);
	return result;
}

__device__ __forceinline__ uint2 vectorizeswap(uint64_t v) {
	uint2 result;
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.y), "=r"(result.x) : "l"(v));
	return result;
}

__device__
static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t in, const uint32_t Kshared)
{
	uint32_t t1,t2;
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
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
/*__launch_bounds__(256,3)*/
void lbry_sha256_gpu_hash_112(const uint32_t threads, const uint32_t startNonce, const bool swabNonce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		uint32_t dat[16];
		#pragma unroll
		for (int i=0;i<11;i++) dat[i] = c_dataEnd112[i]; // pre "swabed"
		dat[11] = swabNonce ? cuda_swab32(nonce) : nonce;
		dat[12] = 0x80000000;
		dat[13] = 0;
		dat[14] = 0;
		dat[15] = 0x380;

		uint32_t __align__(8) buf[8];
		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_midstate112[i];

		sha256_round_body(dat, buf, c_K);

		// output
		uint2* output = (uint2*) (&outputHash[thread * 8U]);
		#pragma unroll
		for (int i=0;i<4;i++) {
			//output[i] = vectorize(cuda_swab32ll(((uint64_t*)buf)[i]));
			output[i] = vectorize(((uint64_t*)buf)[i]); // out without swap, new sha256 after
		}
	}
}

__global__
/*__launch_bounds__(256,3)*/
void lbry_sha256_gpu_hash_32(uint32_t threads, uint64_t *Hash512)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t __align__(8) buf[8]; // align for vectorize
		#pragma unroll
		for (int i=0; i<8; i++) buf[i] = c_H256[i];

		uint32_t* input = (uint32_t*) (&Hash512[thread * 8U]);

		uint32_t dat[16];
		#pragma unroll
		//for (int i=0;i<8;i++) dat[i] = cuda_swab32(input[i]);
		for (int i=0; i<8; i++) dat[i] = input[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		sha256_round_body(dat, buf, c_K);

		// output
		uint2* output = (uint2*) input;
		#pragma unroll
		for (int i=0;i<4;i++) {
			//output[i] = vectorize(cuda_swab32ll(((uint64_t*)buf)[i]));
			output[i] = vectorizeswap(((uint64_t*)buf)[i]);
		}
#ifdef PAD_ZEROS
		#pragma unroll
		for (int i=4; i<8; i++) output[i] = vectorize(0);
#endif
	}
}

__global__
/*__launch_bounds__(256,3)*/
void lbry_sha256d_gpu_hash_112(const uint32_t threads, const uint32_t startNonce, const bool swabNonce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	extern __shared__ uint32_t s_K[];
	//s_K[thread & 63] = c_K[thread & 63];
	if (threadIdx.x < 64U) s_K[threadIdx.x] = c_K[threadIdx.x];
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;

		uint32_t dat[16];
		#pragma unroll
		for (int i=0; i<11; i++) dat[i] = c_dataEnd112[i];
		dat[11] = swabNonce ? cuda_swab32(nonce) : nonce;
		dat[12] = 0x80000000;
		dat[13] = 0;
		dat[14] = 0;
		dat[15] = 0x380;

		uint32_t __align__(8) buf[8];
		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_midstate112[i];

		sha256_round_body(dat, buf, s_K);

		// second sha256

		#pragma unroll
		for (int i=0; i<8; i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int i=0; i<8; i++) buf[i] = c_H256[i];

		sha256_round_body(dat, buf, s_K);

		// output
		uint2* output = (uint2*) (&outputHash[thread * 8U]);
		#pragma unroll
		for (int i=0;i<4;i++) {
		//	//output[i] = vectorize(cuda_swab32ll(((uint64_t*)buf)[i]));
			output[i] = vectorizeswap(((uint64_t*)buf)[i]);
		}
	}
}

__global__
/*__launch_bounds__(256,3)*/
void lbry_sha256_gpu_hash_20x2(uint32_t threads, uint64_t *Hash512)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t __align__(8) buf[8]; // align for vectorize
		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_H256[i];

		uint32_t* input = (uint32_t*) (&Hash512[thread * 8U]);

		uint32_t dat[16];
		#pragma unroll
		for (int i=0;i<5;i++) dat[i] = cuda_swab32(input[i]);
		#pragma unroll
		for (int i=0;i<5;i++) dat[i+5] = cuda_swab32(input[i+8]);
		dat[10] = 0x80000000;
		#pragma unroll
		for (int i=11;i<15;i++) dat[i] = 0;
		dat[15] = 0x140;

		sha256_round_body(dat, buf, c_K);

		// output
		uint2* output = (uint2*) input;
		#pragma unroll
		for (int i=0;i<4;i++) {
			//output[i] = vectorize(cuda_swab32ll(((uint64_t*)buf)[i]));
			output[i] = vectorize(((uint64_t*)buf)[i]);
		}
#ifdef PAD_ZEROS
		#pragma unroll
		for (int i=4; i<8; i++) output[i] = vectorize(0);
#endif
	}
}

__global__
/*__launch_bounds__(256,3)*/
void lbry_sha256d_gpu_hash_20x2(uint32_t threads, uint64_t *Hash512)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	extern __shared__ uint32_t s_K[];
	if (threadIdx.x < 64U) s_K[threadIdx.x] = c_K[threadIdx.x];
	if (thread < threads)
	{
		uint32_t __align__(8) buf[8]; // align for vectorize
		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_H256[i];

		uint32_t* input = (uint32_t*) (&Hash512[thread * 8U]);

		uint32_t dat[16];
		#pragma unroll
		for (int i=0; i<5; i++) dat[i] = cuda_swab32(input[i]);
		#pragma unroll
		for (int i=0; i<5; i++) dat[i+5] = cuda_swab32(input[i+8]);
		dat[10] = 0x80000000;
		#pragma unroll
		for (int i=11;i<15;i++) dat[i] = 0;
		dat[15] = 0x140;

		sha256_round_body(dat, buf, s_K);

		// second sha256

		#pragma unroll
		for (int i=0; i<8; i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_H256[i];

		sha256_round_body(dat, buf, s_K);

		// output
		uint2* output = (uint2*) input;

#ifdef FULL_HASH
		#pragma unroll
		for (int i=0;i<4;i++) {
			output[i] = vectorize(cuda_swab32ll(((uint64_t*)buf)[i]));
			//output[i] = vectorize(((uint64_t*)buf)[i]);
		}
#	ifdef PAD_ZEROS
		#pragma unroll
		for (int i=4; i<8; i++) output[i] = vectorize(0);
#	endif

#else
		//input[6] = cuda_swab32(buf[6]);
		//input[7] = cuda_swab32(buf[7]);
		output[3] = vectorize(cuda_swab32ll(((uint64_t*)buf)[3]));
#endif
	}
}

__host__
void lbry_sha256_init(int thr_id)
{
	//cudaMemcpyToSymbol(c_H256, cpu_H256, sizeof(cpu_H256), 0, cudaMemcpyHostToDevice);
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

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_midstate112, buf, 32, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_dataEnd112,  end, sizeof(end), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, &ptarget[6], sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_target, &ptarget[6], sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void lbry_sha256_hash_112(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash, bool swabNonce, cudaStream_t stream)
{
	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256_gpu_hash_112 <<<grid, block, 0, stream>>> (threads, startNonce, swabNonce, (uint64_t*) d_outputHash);
	cudaGetLastError();
}

__host__
void lbry_sha256_hash_32(int thr_id, uint32_t threads, uint32_t *d_Hash, cudaStream_t stream)
{
	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256_gpu_hash_32 <<<grid, block, 0, stream>>> (threads, (uint64_t*) d_Hash);
}

__host__
void lbry_sha256d_hash_112(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash, bool swabNonce, cudaStream_t stream)
{
	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256d_gpu_hash_112 <<<grid, block, 64*4, stream>>> (threads, startNonce, swabNonce, (uint64_t*) d_outputHash);
}

__host__
void lbry_sha256_hash_20x2(int thr_id, uint32_t threads, uint32_t *d_Hash, cudaStream_t stream)
{
	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256_gpu_hash_20x2 <<<grid, block, 0, stream>>> (threads, (uint64_t*) d_Hash);
}

__host__
void lbry_sha256d_hash_20x2(int thr_id, uint32_t threads, uint32_t *d_Hash, cudaStream_t stream)
{
	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	lbry_sha256d_gpu_hash_20x2 <<<grid, block, 64*4, stream>>> (threads, (uint64_t*) d_Hash);
}

__global__
__launch_bounds__(256,3)
void lbry_sha256d_gpu_hash_final(const uint32_t threads, const uint32_t startNonce, uint64_t *Hash512, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t __align__(8) buf[8]; // align for vectorize
		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_H256[i];

		uint32_t* input = (uint32_t*) (&Hash512[thread * 8U]);

		uint32_t __align__(8) dat[16];
		#pragma unroll
		for (int i=0;i<5;i++) dat[i] = cuda_swab32(input[i]);
		#pragma unroll
		for (int i=0;i<5;i++) dat[i+5] = cuda_swab32(input[i+8]);
		dat[10] = 0x80000000;
		#pragma unroll
		for (int i=11;i<15;i++) dat[i] = 0;
		dat[15] = 0x140;

		sha256_round_body(dat, buf, c_K);

		// second sha256

		#pragma unroll
		for (int i=0;i<8;i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9;i<15;i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
		for (int i=0;i<8;i++) buf[i] = c_H256[i];

		sha256_round_body(dat, buf, c_K);

		// valid nonces
		uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
		if (high <= d_target[0]) {
			// printf("%08x %08x - %016llx %016llx - %08x %08x\n", buf[7], buf[6], high, d_target[0], c_target[1], c_target[0]);
			uint32_t nonce = startNonce + thread;
			resNonces[1] = atomicExch(resNonces, nonce);
			d_target[0] = high;
		}
	}
}

__host__
void lbry_sha256d_hash_final(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_inputHash, uint32_t *resNonces, cudaStream_t stream)
{
	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	CUDA_SAFE_CALL(cudaMemset(d_resNonces, 0xFF, 2 * sizeof(uint32_t)));
	cudaThreadSynchronize();

	lbry_sha256d_gpu_hash_final <<<grid, block, 0, stream>>> (threads, startNonce, (uint64_t*) d_inputHash, d_resNonces);

	cudaThreadSynchronize();

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}