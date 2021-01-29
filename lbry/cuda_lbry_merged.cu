/*
 * LBRY merged kernel CUDA implementation.
 * For compute 5.2 and beyond gpus
 * tpruvot and Provos Alexis - Sep 2016
 * Sponsored by LBRY.IO team
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>

#include <miner.h>

__constant__ static uint32_t _ALIGN(16) c_midstate112[8];
__constant__ static uint32_t _ALIGN(16) c_midbuffer112[8];
__constant__ static uint32_t _ALIGN(16) c_dataEnd112[12];

__constant__ static const uint32_t c_H256[8] = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};
__constant__ static uint32_t _ALIGN(8) c_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

#ifdef __INTELLISENSE__
#define atomicExch(p,y) y
#define __byte_perm(x,y,z) x
#endif

// ------------------------------------------------------------------------------------------------

static const uint32_t cpu_H256[8] = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
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
static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h, uint32_t in, const uint32_t Kshared)
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

__host__
void lbry_sha256_setBlock_112_merged(uint32_t *pdata)
{
	uint32_t in[16], buf[8], end[16];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0; i<8;i++) buf[i] = cpu_H256[i];
	for (int i=0;i<11;i++) end[i] = cuda_swab32(pdata[16+i]);
	sha256_round_body_host(in, buf, cpu_K);

	cudaMemcpyToSymbol(c_midstate112, buf, 32, 0, cudaMemcpyHostToDevice);

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
	sha256_step1_host(f, g, h, a, b, c, d, e, 0, cpu_K[11]);

	buf[0] = a;
	buf[1] = b;
	buf[2] = c;
	buf[3] = d;
	buf[4] = e;
	buf[5] = f;
	buf[6] = g;
	buf[7] = h;

	cudaMemcpyToSymbol(c_midbuffer112, buf, 32, 0, cudaMemcpyHostToDevice);

	end[12] = 0x80000000;
	end[13] = 0;
	end[14] = 0;
	end[15] = 0x380;
	uint32_t x2_0,x2_1;

	x2_0 = ROTR32(end[1], 7) ^ ROTR32(end[1], 18) ^ SPH_T32(end[1] >> 3); //ssg2_0(inx3);//ssg2_0(end[1]);
//	x2_1 = ROTR32(end[14], 17) ^ ROTR32(end[14], 19) ^ SPH_T32(end[14] >> 10) + x2_0; //ssg2_1(inx1); ssg2_1(end[14]) + x2_0;
	end[0] = end[0] + end[9] + x2_0;

	x2_0 = ROTR32(end[2], 7) ^ ROTR32(end[2], 18) ^ SPH_T32(end[2] >> 3);
	x2_1 = (ROTR32(end[15], 17) ^ ROTR32(end[15], 19) ^ SPH_T32(end[15] >> 10)) + x2_0;
	end[1] = end[1] + end[10] + x2_1;

	x2_0 = ROTR32(end[3], 7) ^ ROTR32(end[3], 18) ^ SPH_T32(end[3] >> 3);//ssg2_0(end[3]);
	x2_1 = (ROTR32(end[0], 17) ^ ROTR32(end[0], 19) ^ SPH_T32(end[0] >> 10)) + x2_0;
	end[2]+= x2_1;

	x2_0 = ROTR32(end[4], 7) ^ ROTR32(end[4], 18) ^ SPH_T32(end[4] >> 3);//ssg2_0(end[4]);
	x2_1 = (ROTR32(end[1], 17) ^ ROTR32(end[1], 19) ^ SPH_T32(end[1] >> 10)) + x2_0;
	end[3] = end[3] + end[12] + x2_1;

	x2_0 = ROTR32(end[5], 7) ^ ROTR32(end[5], 18) ^ SPH_T32(end[5] >> 3);//ssg2_0(end[4]);
	end[4] = end[4] + end[13] + x2_0;

	x2_0 = ROTR32(end[6], 7) ^ ROTR32(end[6], 18) ^ SPH_T32(end[6] >> 3);//ssg2_0(end[6]);
	x2_1 = (ROTR32(end[3], 17) ^ ROTR32(end[3], 19) ^ SPH_T32(end[3] >> 10)) + x2_0;
	end[5] = end[5] + end[14] + x2_1;

	x2_0 = ROTR32(end[7], 7) ^ ROTR32(end[7], 18) ^ SPH_T32(end[7] >> 3);//ssg2_0(end[7]);
	end[6] = end[6] + end[15] + x2_0;

	x2_0 = ROTR32(end[8], 7) ^ ROTR32(end[8], 18) ^ SPH_T32(end[8] >> 3);//ssg2_0(end[8]);
	x2_1 = (ROTR32(end[5], 17) ^ ROTR32(end[5], 19) ^ SPH_T32(end[5] >> 10)) + x2_0;
	end[7] = end[7] + end[0] + x2_1;

	x2_0 = ROTR32(end[9], 7) ^ ROTR32(end[9], 18) ^ SPH_T32(end[9] >> 3);//ssg2_0(end[9]);
	end[8] = end[8] + end[1] + x2_0;

	x2_0 = ROTR32(end[10], 7) ^ ROTR32(end[10], 18) ^ SPH_T32(end[10] >> 3);//ssg2_0(end[10]);
	x2_1 = (ROTR32(end[7], 17) ^ ROTR32(end[7], 19) ^ SPH_T32(end[7] >> 10)) + x2_0;
	end[9] = end[9] + x2_1;

	cudaMemcpyToSymbol(c_dataEnd112,  end, sizeof(end), 0, cudaMemcpyHostToDevice);
}

//END OF HOST FUNCTIONS -------------------------------------------------------------------

//SHA256 MACROS ---------------------------------------------------------------------------

#define xor3b(a,b,c) (a ^ b ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x){
	return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x){
	return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x){
	return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x){
	return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

__device__ __forceinline__ uint64_t vectorizeswap(const uint64_t v){
	uint2 result;
	asm volatile ("mov.b64 {%0,%1},%2;" : "=r"(result.y), "=r"(result.x) : "l"(v));
	return devectorize(result);
}

#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define Ch(a, b, c)     (((b^c) & a) ^ c)

__device__
static void sha2_step(const uint32_t a, const uint32_t b,const uint32_t c, uint32_t &d,
	const uint32_t e,const uint32_t f,const uint32_t g, uint32_t &h,
	const uint32_t in, const uint32_t Kshared)
{
	const uint32_t t1 = h + bsg2_1(e) + Ch(e, f, g) + Kshared + in;
	h = t1 + Maj(a, b, c) + bsg2_0(a);
	d+= t1;
}

__device__
static void sha256_round_first(uint32_t *in, uint32_t *buf,
	const uint32_t *state, const uint32_t* __restrict__ Kshared)
{
	uint32_t a = buf[0] + in[11];
	uint32_t b = buf[1];
	uint32_t c = buf[2];
	uint32_t d = buf[3];
	uint32_t e = buf[4] + in[11];
	uint32_t f = buf[5];
	uint32_t g = buf[6];
	uint32_t h = buf[7];

	// 10 first steps made on host
	//sha2_step(f,g,h,a,b,c,d,e,in[11],Kshared[11]);

	sha2_step(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	//in is partially precomputed on host
	in[2]+= in[11];
	in[4]+= ssg2_1(in[2]);
	in[6]+= ssg2_1(in[4]);
	in[8]+= ssg2_1(in[6]);
	in[9]+= in[ 2];

	sha2_step(a,b,c,d,e,f,g,h,in[0], Kshared[16]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], Kshared[17]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], Kshared[18]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], Kshared[19]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], Kshared[20]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], Kshared[21]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], Kshared[22]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], Kshared[23]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], Kshared[24]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], Kshared[25]);

	#pragma unroll 6
	for (uint32_t j = 10; j < 16; j++) {
		const uint32_t x2_0 = ssg2_0(in[(j + 1) & 15]);
		const uint32_t x2_1 = ssg2_1(in[(j + 14) & 15]) + x2_0;
		in[j] = in[j] + in[(j + 9) & 15] + x2_1;
	}

	sha2_step(g,h,a,b,c,d,e,f,in[10],Kshared[26]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],Kshared[27]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],Kshared[28]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],Kshared[29]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],Kshared[30]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],Kshared[31]);

	#pragma unroll 16
	for (uint32_t j = 0; j < 16; j++) {
		const uint32_t x2_0 = ssg2_0(in[(j + 1) & 15]);
		const uint32_t x2_1 = ssg2_1(in[(j + 14) & 15]) + x2_0;
		in[j] = in[j] + in[(j + 9) & 15] + x2_1;
	}

	sha2_step(a,b,c,d,e,f,g,h,in[0], Kshared[16+16]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], Kshared[17+16]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], Kshared[18+16]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], Kshared[19+16]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], Kshared[20+16]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], Kshared[21+16]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], Kshared[22+16]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], Kshared[23+16]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], Kshared[24+16]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], Kshared[25+16]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],Kshared[26+16]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],Kshared[27+16]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],Kshared[28+16]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],Kshared[29+16]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],Kshared[30+16]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],Kshared[31+16]);

	#pragma unroll 16
	for (uint32_t j = 0; j < 16; j++) {
		const uint32_t x2_0 = ssg2_0(in[(j + 1) & 15]);
		const uint32_t x2_1 = ssg2_1(in[(j + 14) & 15]) + x2_0;
		in[j] = in[j] + in[(j + 9) & 15] + x2_1;
	}

	sha2_step(a,b,c,d,e,f,g,h,in[0], Kshared[16+16*2]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], Kshared[17+16*2]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], Kshared[18+16*2]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], Kshared[19+16*2]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], Kshared[20+16*2]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], Kshared[21+16*2]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], Kshared[22+16*2]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], Kshared[23+16*2]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], Kshared[24+16*2]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], Kshared[25+16*2]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],Kshared[26+16*2]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],Kshared[27+16*2]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],Kshared[28+16*2]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],Kshared[29+16*2]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],Kshared[30+16*2]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],Kshared[31+16*2]);

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
static void sha256_round_body(uint32_t *in, uint32_t *state,const uint32_t* Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
	sha2_step(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
	sha2_step(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
	sha2_step(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
	sha2_step(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
	sha2_step(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
	sha2_step(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
	sha2_step(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
	sha2_step(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
	sha2_step(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
	sha2_step(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
	sha2_step(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
	sha2_step(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

	#pragma unroll 3
	for (int i=0; i<3; i++)
	{
		#pragma unroll 16
		for (uint32_t j = 0; j < 16; j++) {
			const uint32_t x2_0 = ssg2_0(in[(j + 1) & 15]);
			const uint32_t x2_1 = ssg2_1(in[(j + 14) & 15]) + x2_0;
			in[j] = in[j] + in[(j + 9) & 15] + x2_1;
		}
		sha2_step(a, b, c, d, e, f, g, h, in[ 0], Kshared[16 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[ 1], Kshared[17 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[ 2], Kshared[18 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[ 3], Kshared[19 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[ 4], Kshared[20 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[ 5], Kshared[21 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[ 6], Kshared[22 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[ 7], Kshared[23 + 16 * i]);
		sha2_step(a, b, c, d, e, f, g, h, in[ 8], Kshared[24 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[ 9], Kshared[25 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[10], Kshared[26 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[11], Kshared[27 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[12], Kshared[28 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[13], Kshared[29 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[14], Kshared[30 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[15], Kshared[31 + 16 * i]);
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
static void sha256_round_body_final(uint32_t *in, uint32_t *state, const uint32_t *Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha2_step(a,b,c,d,e,f,g,h,in[ 0], Kshared[0]);
	sha2_step(h,a,b,c,d,e,f,g,in[ 1], Kshared[1]);
	sha2_step(g,h,a,b,c,d,e,f,in[ 2], Kshared[2]);
	sha2_step(f,g,h,a,b,c,d,e,in[ 3], Kshared[3]);
	sha2_step(e,f,g,h,a,b,c,d,in[ 4], Kshared[4]);
	sha2_step(d,e,f,g,h,a,b,c,in[ 5], Kshared[5]);
	sha2_step(c,d,e,f,g,h,a,b,in[ 6], Kshared[6]);
	sha2_step(b,c,d,e,f,g,h,a,in[ 7], Kshared[7]);
	sha2_step(a,b,c,d,e,f,g,h,in[ 8], Kshared[8]);
	sha2_step(h,a,b,c,d,e,f,g,in[ 9], Kshared[9]);
	sha2_step(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
	sha2_step(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
	sha2_step(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

	#pragma unroll 2
	for (int i=0; i<2; i++)
	{
		#pragma unroll 16
		for (uint32_t j = 0; j < 16; j++) {
			const uint32_t x2_0 = ssg2_0(in[(j + 1) & 15]);
			const uint32_t x2_1 = ssg2_1(in[(j + 14) & 15]) + x2_0;
			in[j] = in[j] + in[(j + 9) & 15] + x2_1;
		}
		sha2_step(a, b, c, d, e, f, g, h, in[ 0], Kshared[16 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[ 1], Kshared[17 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[ 2], Kshared[18 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[ 3], Kshared[19 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[ 4], Kshared[20 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[ 5], Kshared[21 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[ 6], Kshared[22 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[ 7], Kshared[23 + 16 * i]);
		sha2_step(a, b, c, d, e, f, g, h, in[ 8], Kshared[24 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[ 9], Kshared[25 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[10], Kshared[26 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[11], Kshared[27 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[12], Kshared[28 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[13], Kshared[29 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[14], Kshared[30 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[15], Kshared[31 + 16 * i]);
	}
	#pragma unroll 16
	for (uint32_t j = 0; j < 16; j++) {
		const uint32_t x2_0 = ssg2_0(in[(j + 1) & 15]);
		const uint32_t x2_1 = ssg2_1(in[(j + 14) & 15]) + x2_0;
		in[j] = in[j] + in[(j + 9) & 15] + x2_1;
	}
	sha2_step(a, b, c, d, e, f, g, h, in[ 0], Kshared[16 + 16 * 2]);
	sha2_step(h, a, b, c, d, e, f, g, in[ 1], Kshared[17 + 16 * 2]);
	sha2_step(g, h, a, b, c, d, e, f, in[ 2], Kshared[18 + 16 * 2]);
	sha2_step(f, g, h, a, b, c, d, e, in[ 3], Kshared[19 + 16 * 2]);
	sha2_step(e, f, g, h, a, b, c, d, in[ 4], Kshared[20 + 16 * 2]);
	sha2_step(d, e, f, g, h, a, b, c, in[ 5], Kshared[21 + 16 * 2]);
	sha2_step(c, d, e, f, g, h, a, b, in[ 6], Kshared[22 + 16 * 2]);
	sha2_step(b, c, d, e, f, g, h, a, in[ 7], Kshared[23 + 16 * 2]);
	sha2_step(a, b, c, d, e, f, g, h, in[ 8], Kshared[24 + 16 * 2]);
	sha2_step(h, a, b, c, d, e, f, g, in[ 9], Kshared[25 + 16 * 2]);
	sha2_step(g, h, a, b, c, d, e, f, in[10], Kshared[26 + 16 * 2]);
	sha2_step(f, g, h, a, b, c, d, e, in[11], Kshared[27 + 16 * 2]);
	sha2_step(e, f, g, h, a, b, c, d, in[12], Kshared[28 + 16 * 2]);
	sha2_step(d, e, f, g, h, a, b, c, in[13], Kshared[29 + 16 * 2]);

	state[6] += g;
	state[7] += h;
}

//END OF SHA256 MACROS --------------------------------------------------------------------

//SHA512 MACROS ---------------------------------------------------------------------------
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

#define bsg5_0(x) xor3(ROTR64(x,28),ROTR64(x,34),ROTR64(x,39))
#define bsg5_1(x) xor3(ROTR64(x,14),ROTR64(x,18),ROTR64(x,41))
#define ssg5_0(x) xor3(ROTR64(x, 1),ROTR64(x, 8),x>>7)
#define ssg5_1(x) xor3(ROTR64(x,19),ROTR64(x,61),x>>6)

#define andor64(a,b,c) ((a & (b | c)) | (b & c))
#define xandx64(e,f,g) (g ^ (e & (g ^ f)))

__device__ __forceinline__
uint64_t cuda_swab64ll(const uint32_t x, const uint32_t y)
{
	uint64_t r;
	asm("prmt.b32 %1, %1, 0, 0x0123; // swab64ll\n\t"
	    "prmt.b32 %2, %2, 0, 0x0123;\n\t"
	    "mov.b64 %0, {%1,%2};\n\t"
	  : "=l"(r): "r"(x), "r"(y) );
	return r;
}

// RIPEMD MACROS-----------------------------------------------------------------------------
static __constant__ const uint32_t c_IV[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
static __constant__ const uint32_t c_K1[5] = { 0, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E };
static __constant__ const uint32_t c_K2[5] = { 0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0 };

__device__ __forceinline__
static uint32_t ROTATE(const uint32_t x,const uint32_t r) {
	if(r==8)
		return __byte_perm(x, 0, 0x2103);
	else
		return ROTL32(x,r);
}

/*
 * Round functions for RIPEMD-160.
 */
//#define F1(x, y, z)   xor3x(x, y, z)
__device__ __forceinline__
uint32_t F1(const uint32_t a, const uint32_t b, const uint32_t c) {
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm volatile ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b),"r"(c));
	#else
		result = a^b^c;
	#endif
	return result;
}
//#define F2(x, y, z)   ((x & (y ^ z)) ^ z)
__device__ __forceinline__
uint32_t F2(const uint32_t a, const uint32_t b, const uint32_t c) {
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm volatile ("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result) : "r"(a), "r"(b),"r"(c)); //0xCA=((F0∧(CC⊻AA))⊻AA)
	#else
		result = ((a & (b ^ c)) ^ c);
	#endif
	return result;
}
//#define F3(x, y, z)   ((x | ~y) ^ z)
__device__ __forceinline__
uint32_t F3(const uint32_t x, const uint32_t y, const uint32_t z) {
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm volatile ("lop3.b32 %0, %1, %2, %3, 0x59;" : "=r"(result) : "r"(x), "r"(y),"r"(z)); //0x59=((F0∨(¬CC))⊻AA)
	#else
		result = ((x | ~y) ^ z);
	#endif
	return result;
}
//#define F4(x, y, z)   (y ^ ((x ^ y) & z))
__device__ __forceinline__
uint32_t F4(const uint32_t x, const uint32_t y, const uint32_t z) {
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm volatile ("lop3.b32 %0, %1, %2, %3, 0xE4;" : "=r"(result) : "r"(x), "r"(y),"r"(z)); //0xE4=(CC⊻((F0⊻CC)∧AA))
	#else
		result = (y ^ ((x ^ y) & z));
	#endif
	return result;
}
//#define F5(x, y, z)   (x ^ (y | ~z))
__device__ __forceinline__
uint32_t F5(const uint32_t x, const uint32_t y, const uint32_t z) {
	uint32_t result;
	#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
		asm volatile ("lop3.b32 %0, %1, %2, %3, 0x2D;" : "=r"(result) : "r"(x), "r"(y),"r"(z)); //0x2D=(F0⊻(CC∨(¬AA)))
	#else
		result = (x ^ (y | ~z));
	#endif
	return result;
}

/*
 * Round constants for RIPEMD-160.
 */
#define RR(a, b, c, d, e, f, s, r, k) { \
	a = e + ROTATE((a + k + r + f(b, c, d)), s); \
	c = ROTL32(c, 10); \
}

#define ROUND1(a, b, c, d, e, f, s, r, k) \
	RR(a[0], b[0], c[0], d[0], e[0], f, s, r, c_K1[k])

#define ROUND2(a, b, c, d, e, f, s, r, k) \
	RR(a[1], b[1], c[1], d[1], e[1], f, s, r, c_K2[k])

#define RIPEMD160_ROUND_BODY(in, h) { \
	uint32_t A[2], B[2], C[2], D[2], E[2]; \
	uint32_t tmp; \
\
	A[0] = A[1] = h[0]; \
	B[0] = B[1] = h[1]; \
	C[0] = C[1] = h[2]; \
	D[0] = D[1] = h[3]; \
	E[0] = E[1] = h[4]; \
\
	ROUND1(A, B, C, D, E, F1, 11, in[ 0],  0); \
	ROUND1(E, A, B, C, D, F1, 14, in[ 1],  0); \
	ROUND1(D, E, A, B, C, F1, 15, in[ 2],  0); \
	ROUND1(C, D, E, A, B, F1, 12, in[ 3],  0); \
	ROUND1(B, C, D, E, A, F1,  5, in[ 4],  0); \
	ROUND1(A, B, C, D, E, F1,  8, in[ 5],  0); \
	ROUND1(E, A, B, C, D, F1,  7, in[ 6],  0); \
	ROUND1(D, E, A, B, C, F1,  9, in[ 7],  0); \
	ROUND1(C, D, E, A, B, F1, 11, in[ 8],  0); \
	ROUND1(B, C, D, E, A, F1, 13, in[ 9],  0); \
	ROUND1(A, B, C, D, E, F1, 14, in[10],  0); \
	ROUND1(E, A, B, C, D, F1, 15, in[11],  0); \
	ROUND1(D, E, A, B, C, F1,  6, in[12],  0); \
	ROUND1(C, D, E, A, B, F1,  7, in[13],  0); \
	ROUND1(B, C, D, E, A, F1,  9, in[14],  0); \
	ROUND1(A, B, C, D, E, F1,  8, in[15],  0); \
\
	ROUND1(E, A, B, C, D, F2,  7, in[ 7],  1); \
	ROUND1(D, E, A, B, C, F2,  6, in[ 4],  1); \
	ROUND1(C, D, E, A, B, F2,  8, in[13],  1); \
	ROUND1(B, C, D, E, A, F2, 13, in[ 1],  1); \
	ROUND1(A, B, C, D, E, F2, 11, in[10],  1); \
	ROUND1(E, A, B, C, D, F2,  9, in[ 6],  1); \
	ROUND1(D, E, A, B, C, F2,  7, in[15],  1); \
	ROUND1(C, D, E, A, B, F2, 15, in[ 3],  1); \
	ROUND1(B, C, D, E, A, F2,  7, in[12],  1); \
	ROUND1(A, B, C, D, E, F2, 12, in[ 0],  1); \
	ROUND1(E, A, B, C, D, F2, 15, in[ 9],  1); \
	ROUND1(D, E, A, B, C, F2,  9, in[ 5],  1); \
	ROUND1(C, D, E, A, B, F2, 11, in[ 2],  1); \
	ROUND1(B, C, D, E, A, F2,  7, in[14],  1); \
	ROUND1(A, B, C, D, E, F2, 13, in[11],  1); \
	ROUND1(E, A, B, C, D, F2, 12, in[ 8],  1); \
\
	ROUND1(D, E, A, B, C, F3, 11, in[ 3],  2); \
	ROUND1(C, D, E, A, B, F3, 13, in[10],  2); \
	ROUND1(B, C, D, E, A, F3,  6, in[14],  2); \
	ROUND1(A, B, C, D, E, F3,  7, in[ 4],  2); \
	ROUND1(E, A, B, C, D, F3, 14, in[ 9],  2); \
	ROUND1(D, E, A, B, C, F3,  9, in[15],  2); \
	ROUND1(C, D, E, A, B, F3, 13, in[ 8],  2); \
	ROUND1(B, C, D, E, A, F3, 15, in[ 1],  2); \
	ROUND1(A, B, C, D, E, F3, 14, in[ 2],  2); \
	ROUND1(E, A, B, C, D, F3,  8, in[ 7],  2); \
	ROUND1(D, E, A, B, C, F3, 13, in[ 0],  2); \
	ROUND1(C, D, E, A, B, F3,  6, in[ 6],  2); \
	ROUND1(B, C, D, E, A, F3,  5, in[13],  2); \
	ROUND1(A, B, C, D, E, F3, 12, in[11],  2); \
	ROUND1(E, A, B, C, D, F3,  7, in[ 5],  2); \
	ROUND1(D, E, A, B, C, F3,  5, in[12],  2); \
\
	ROUND1(C, D, E, A, B, F4, 11, in[ 1],  3); \
	ROUND1(B, C, D, E, A, F4, 12, in[ 9],  3); \
	ROUND1(A, B, C, D, E, F4, 14, in[11],  3); \
	ROUND1(E, A, B, C, D, F4, 15, in[10],  3); \
	ROUND1(D, E, A, B, C, F4, 14, in[ 0],  3); \
	ROUND1(C, D, E, A, B, F4, 15, in[ 8],  3); \
	ROUND1(B, C, D, E, A, F4,  9, in[12],  3); \
	ROUND1(A, B, C, D, E, F4,  8, in[ 4],  3); \
	ROUND1(E, A, B, C, D, F4,  9, in[13],  3); \
	ROUND1(D, E, A, B, C, F4, 14, in[ 3],  3); \
	ROUND1(C, D, E, A, B, F4,  5, in[ 7],  3); \
	ROUND1(B, C, D, E, A, F4,  6, in[15],  3); \
	ROUND1(A, B, C, D, E, F4,  8, in[14],  3); \
	ROUND1(E, A, B, C, D, F4,  6, in[ 5],  3); \
	ROUND1(D, E, A, B, C, F4,  5, in[ 6],  3); \
	ROUND1(C, D, E, A, B, F4, 12, in[ 2],  3); \
\
	ROUND1(B, C, D, E, A, F5,  9, in[ 4],  4); \
	ROUND1(A, B, C, D, E, F5, 15, in[ 0],  4); \
	ROUND1(E, A, B, C, D, F5,  5, in[ 5],  4); \
	ROUND1(D, E, A, B, C, F5, 11, in[ 9],  4); \
	ROUND1(C, D, E, A, B, F5,  6, in[ 7],  4); \
	ROUND1(B, C, D, E, A, F5,  8, in[12],  4); \
	ROUND1(A, B, C, D, E, F5, 13, in[ 2],  4); \
	ROUND1(E, A, B, C, D, F5, 12, in[10],  4); \
	ROUND1(D, E, A, B, C, F5,  5, in[14],  4); \
	ROUND1(C, D, E, A, B, F5, 12, in[ 1],  4); \
	ROUND1(B, C, D, E, A, F5, 13, in[ 3],  4); \
	ROUND1(A, B, C, D, E, F5, 14, in[ 8],  4); \
	ROUND1(E, A, B, C, D, F5, 11, in[11],  4); \
	ROUND1(D, E, A, B, C, F5,  8, in[ 6],  4); \
	ROUND1(C, D, E, A, B, F5,  5, in[15],  4); \
	ROUND1(B, C, D, E, A, F5,  6, in[13],  4); \
\
	ROUND2(A, B, C, D, E, F5,  8, in[ 5],  0); \
	ROUND2(E, A, B, C, D, F5,  9, in[14],  0); \
	ROUND2(D, E, A, B, C, F5,  9, in[ 7],  0); \
	ROUND2(C, D, E, A, B, F5, 11, in[ 0],  0); \
	ROUND2(B, C, D, E, A, F5, 13, in[ 9],  0); \
	ROUND2(A, B, C, D, E, F5, 15, in[ 2],  0); \
	ROUND2(E, A, B, C, D, F5, 15, in[11],  0); \
	ROUND2(D, E, A, B, C, F5,  5, in[ 4],  0); \
	ROUND2(C, D, E, A, B, F5,  7, in[13],  0); \
	ROUND2(B, C, D, E, A, F5,  7, in[ 6],  0); \
	ROUND2(A, B, C, D, E, F5,  8, in[15],  0); \
	ROUND2(E, A, B, C, D, F5, 11, in[ 8],  0); \
	ROUND2(D, E, A, B, C, F5, 14, in[ 1],  0); \
	ROUND2(C, D, E, A, B, F5, 14, in[10],  0); \
	ROUND2(B, C, D, E, A, F5, 12, in[ 3],  0); \
	ROUND2(A, B, C, D, E, F5,  6, in[12],  0); \
\
	ROUND2(E, A, B, C, D, F4,  9, in[ 6],  1); \
	ROUND2(D, E, A, B, C, F4, 13, in[11],  1); \
	ROUND2(C, D, E, A, B, F4, 15, in[ 3],  1); \
	ROUND2(B, C, D, E, A, F4,  7, in[ 7],  1); \
	ROUND2(A, B, C, D, E, F4, 12, in[ 0],  1); \
	ROUND2(E, A, B, C, D, F4,  8, in[13],  1); \
	ROUND2(D, E, A, B, C, F4,  9, in[ 5],  1); \
	ROUND2(C, D, E, A, B, F4, 11, in[10],  1); \
	ROUND2(B, C, D, E, A, F4,  7, in[14],  1); \
	ROUND2(A, B, C, D, E, F4,  7, in[15],  1); \
	ROUND2(E, A, B, C, D, F4, 12, in[ 8],  1); \
	ROUND2(D, E, A, B, C, F4,  7, in[12],  1); \
	ROUND2(C, D, E, A, B, F4,  6, in[ 4],  1); \
	ROUND2(B, C, D, E, A, F4, 15, in[ 9],  1); \
	ROUND2(A, B, C, D, E, F4, 13, in[ 1],  1); \
	ROUND2(E, A, B, C, D, F4, 11, in[ 2],  1); \
\
	ROUND2(D, E, A, B, C, F3,  9, in[15],  2); \
	ROUND2(C, D, E, A, B, F3,  7, in[ 5],  2); \
	ROUND2(B, C, D, E, A, F3, 15, in[ 1],  2); \
	ROUND2(A, B, C, D, E, F3, 11, in[ 3],  2); \
	ROUND2(E, A, B, C, D, F3,  8, in[ 7],  2); \
	ROUND2(D, E, A, B, C, F3,  6, in[14],  2); \
	ROUND2(C, D, E, A, B, F3,  6, in[ 6],  2); \
	ROUND2(B, C, D, E, A, F3, 14, in[ 9],  2); \
	ROUND2(A, B, C, D, E, F3, 12, in[11],  2); \
	ROUND2(E, A, B, C, D, F3, 13, in[ 8],  2); \
	ROUND2(D, E, A, B, C, F3,  5, in[12],  2); \
	ROUND2(C, D, E, A, B, F3, 14, in[ 2],  2); \
	ROUND2(B, C, D, E, A, F3, 13, in[10],  2); \
	ROUND2(A, B, C, D, E, F3, 13, in[ 0],  2); \
	ROUND2(E, A, B, C, D, F3,  7, in[ 4],  2); \
	ROUND2(D, E, A, B, C, F3,  5, in[13],  2); \
\
	ROUND2(C, D, E, A, B, F2, 15, in[ 8],  3); \
	ROUND2(B, C, D, E, A, F2,  5, in[ 6],  3); \
	ROUND2(A, B, C, D, E, F2,  8, in[ 4],  3); \
	ROUND2(E, A, B, C, D, F2, 11, in[ 1],  3); \
	ROUND2(D, E, A, B, C, F2, 14, in[ 3],  3); \
	ROUND2(C, D, E, A, B, F2, 14, in[11],  3); \
	ROUND2(B, C, D, E, A, F2,  6, in[15],  3); \
	ROUND2(A, B, C, D, E, F2, 14, in[ 0],  3); \
	ROUND2(E, A, B, C, D, F2,  6, in[ 5],  3); \
	ROUND2(D, E, A, B, C, F2,  9, in[12],  3); \
	ROUND2(C, D, E, A, B, F2, 12, in[ 2],  3); \
	ROUND2(B, C, D, E, A, F2,  9, in[13],  3); \
	ROUND2(A, B, C, D, E, F2, 12, in[ 9],  3); \
	ROUND2(E, A, B, C, D, F2,  5, in[ 7],  3); \
	ROUND2(D, E, A, B, C, F2, 15, in[10],  3); \
	ROUND2(C, D, E, A, B, F2,  8, in[14],  3); \
\
	ROUND2(B, C, D, E, A, F1,  8, in[12],  4); \
	ROUND2(A, B, C, D, E, F1,  5, in[15],  4); \
	ROUND2(E, A, B, C, D, F1, 12, in[10],  4); \
	ROUND2(D, E, A, B, C, F1,  9, in[ 4],  4); \
	ROUND2(C, D, E, A, B, F1, 12, in[ 1],  4); \
	ROUND2(B, C, D, E, A, F1,  5, in[ 5],  4); \
	ROUND2(A, B, C, D, E, F1, 14, in[ 8],  4); \
	ROUND2(E, A, B, C, D, F1,  6, in[ 7],  4); \
	ROUND2(D, E, A, B, C, F1,  8, in[ 6],  4); \
	ROUND2(C, D, E, A, B, F1, 13, in[ 2],  4); \
	ROUND2(B, C, D, E, A, F1,  6, in[13],  4); \
	ROUND2(A, B, C, D, E, F1,  5, in[14],  4); \
	ROUND2(E, A, B, C, D, F1, 15, in[ 0],  4); \
	ROUND2(D, E, A, B, C, F1, 13, in[ 3],  4); \
	ROUND2(C, D, E, A, B, F1, 11, in[ 9],  4); \
	ROUND2(B, C, D, E, A, F1, 11, in[11],  4); \
\
	tmp  = h[1] + C[0] + D[1]; \
	h[1] = h[2] + D[0] + E[1]; \
	h[2] = h[3] + E[0] + A[1]; \
	h[3] = h[4] + A[0] + B[1]; \
	h[4] = h[0] + B[0] + C[1]; \
	h[0] = tmp; \
}
// END OF RIPEMD MACROS----------------------------------------------------------------------

__global__
__launch_bounds__(768,1) /* will force 64 regs max on SM 3+ */
void gpu_lbry_merged(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces, const uint64_t target64)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	const uint64_t IV512[8] = {
		0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
		0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
	};

	if (thread < threads)
	{
		uint64_t r[8];
		uint64_t W[16];
		uint32_t dat[16];
		uint32_t buf[8], state[8];
		uint32_t h[5];

		*(uint2x4*)&dat[0] = *(uint2x4*)&c_dataEnd112[0];
		dat[ 8] = c_dataEnd112[ 8];
		dat[ 9] = c_dataEnd112[ 9];
		dat[10] = c_dataEnd112[10];
		dat[11] = startNonce + thread;
		dat[12] = 0x80000000;
		dat[13] = 0;
		dat[14] = 0;
		dat[15] = 0x380;

		*(uint2x4*)&state[0] = *(uint2x4*)&c_midstate112[0];
		*(uint2x4*)&buf[0]   = *(uint2x4*)&c_midbuffer112[0];

		sha256_round_first(dat, buf, state, c_K);

		// second sha256

		#pragma unroll 8
		for(int i=0; i<8; i++){
			dat[i] = buf[i];
		}
		dat[8] = 0x80000000;

		#pragma unroll 6
		for(int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll 8
		for(int i=0; i<8; i++)
			buf[i] = c_H256[i];

		sha256_round_body(dat, buf, c_K);

// SHA512-------------------------------------------------------------------------------------

		#pragma unroll 8
		for(int i=0; i<8; i++)
			r[i] = IV512[i];

		W[0] = vectorizeswap(((uint64_t*)buf)[0]);
		W[1] = vectorizeswap(((uint64_t*)buf)[1]);
		W[2] = vectorizeswap(((uint64_t*)buf)[2]);
		W[3] = vectorizeswap(((uint64_t*)buf)[3]);
		W[4] = 0x8000000000000000; // end tag

		#pragma unroll 10
		for (int i = 5; i < 15; i++)
			W[i] = 0;

		W[15] = 0x100; // 256 bits

		#pragma unroll 16
		for (uint32_t i = 0; i < 16; i++)
		{
//			sha512_step2(r, W[ i], K_512[ i], i&7);
			const uint32_t ord = i&7;
			const uint64_t T1 = r[(15-ord) & 7] + K_512[ i] + W[ i] + bsg5_1(r[(12-ord) & 7]) +
				xandx64(r[(12-ord) & 7], r[(13-ord) & 7], r[(14-ord) & 7]);

			r[(15-ord)& 7] = andor64(r[( 8-ord) & 7], r[( 9-ord) & 7], r[(10-ord) & 7]) + bsg5_0(r[( 8-ord) & 7]) + T1;
			r[(11-ord)& 7] = r[(11-ord)& 7] + T1;
		}

		#pragma unroll 5
		for (uint32_t i = 16; i < 80; i+=16)
		{
			#pragma unroll 16
			for (uint32_t j = 0; j<16; j++)
				W[(i + j) & 15] = W[((i + j) - 7) & 15] + W[(i + j) & 15] + ssg5_0(W[((i + j) - 15) & 15]) + ssg5_1(W[((i + j) - 2) & 15]);

			#pragma unroll 16
			for (uint32_t j = 0; j<16; j++) {

				const uint32_t ord = (i+j)&7;
				const uint64_t T1 = K_512[i+j] + W[ j] + r[(15-ord) & 7] + bsg5_1(r[(12-ord) & 7]) +
					xandx64(r[(12-ord) & 7], r[(13-ord) & 7], r[(14-ord) & 7]);

				r[(15-ord)& 7] = andor64(r[( 8-ord) & 7], r[( 9-ord) & 7], r[(10-ord) & 7]) + bsg5_0(r[( 8-ord) & 7]) + T1;
				r[(11-ord)& 7] = r[(11-ord)& 7] + T1;
			}
		}

//END OF SHA512------------------------------------------------------------------------------
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
			*(uint64_t*)&dat[i*2] = cuda_swab64(r[i] + IV512[i]);
		dat[8] = 0x80;

		#pragma unroll 7
		for (int i=9; i<16; i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		#pragma unroll 5
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(dat, h);

		#pragma unroll 5
		for (int i=0; i<5; i++)
			buf[i] = h[i];

		// second 32 bytes block hash
		#pragma unroll 4
		for (int i=0; i < 4; i++)
			*(uint64_t*)&dat[i*2] = cuda_swab64(r[i+4] + IV512[i+4]);

		dat[8] = 0x80;

		#pragma unroll 7
		for (int i=9; i<16; i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		#pragma unroll 5
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

		RIPEMD160_ROUND_BODY(dat, h);

		// first final sha256

		#pragma unroll 5
		for (int i=0; i<5; i++) dat[i] = cuda_swab32(buf[i]);
		#pragma unroll 5
		for (int i=0; i<5; i++) dat[i+5] = cuda_swab32(h[i]);
		dat[10] = 0x80000000;
		#pragma unroll 4
		for (int i=11; i<15; i++) dat[i] = 0;

		dat[15] = 0x140;

		#pragma unroll 8
		for(int i=0; i<8; i++)
			buf[i] = c_H256[i];

		sha256_round_body(dat, buf, c_K);

		// second sha256

		#pragma unroll 8
		for(int i=0; i<8; i++) {
			dat[i] = buf[i];
		}
		dat[8] = 0x80000000;

		#pragma unroll 8
		for(int i=0; i<8; i++)
			buf[i] = c_H256[i];

		#pragma unroll 6
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		sha256_round_body_final(dat, buf, c_K);

		// valid nonces
		if (cuda_swab64ll(buf[ 6],buf[ 7]) <= target64) {
			uint32_t tmp = atomicExch(&resNonces[0], thread);
			if (tmp != UINT32_MAX)
				resNonces[1] = tmp;
		}
	}
}

__host__
void lbry_merged(int thr_id, uint32_t startNonce, uint32_t threads, uint32_t *d_resNonce, const uint64_t target64)
{
	uint32_t threadsperblock = 768;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	gpu_lbry_merged <<<grid, block>>> (threads,startNonce, d_resNonce, target64);
}
