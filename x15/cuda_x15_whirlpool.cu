/**
 * Whirlpool-512 CUDA implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014-2016 djm34, tpruvot, SP, Provos Alexis
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 * @author djm34 (initial draft)
 * @author tpruvot (dual old/whirlpool modes, midstate)
 * @author SP ("final" function opt and tuning)
 * @author Provos Alexis (Applied partial shared memory utilization, precomputations, merging & tuning for 970/750ti under CUDA7.5 -> +93% increased throughput of whirlpool)
 */


// Change with caution, used by shared mem fetch
#define TPB80 384
#define TPB64 384

extern "C" {
#include <sph/sph_whirlpool.h>
#include <miner.h>
}

#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>
#include <cuda_vectors.h>

#define xor3x(a,b,c) (a^b^c)

#include "cuda_whirlpool_tables.cuh"

__device__ static uint64_t b0[256];
__device__ static uint64_t b7[256];

__constant__ static uint2 precomputed_round_key_64[72];
__constant__ static uint2 precomputed_round_key_80[80];

__device__ static uint2 c_PaddedMessage80[16];

/**
 * Round constants.
 */
__device__ uint2 InitVector_RC[10];

static uint32_t *d_resNonce[MAX_GPUS] = { 0 };

//--------START OF WHIRLPOOL DEVICE MACROS---------------------------------------------------------------------------
__device__ __forceinline__
void static TRANSFER(uint2 *const __restrict__ dst,const uint2 *const __restrict__ src){
	dst[0] = src[ 0];
	dst[1] = src[ 1];
	dst[2] = src[ 2];
	dst[3] = src[ 3];
	dst[4] = src[ 4];
	dst[5] = src[ 5];
	dst[6] = src[ 6];
	dst[7] = src[ 7];
}

__device__ __forceinline__
static uint2 d_ROUND_ELT_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){
	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7){

	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1_LDG(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){

	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= ROR24(__ldg((uint2*)&b0[__byte_perm(in[i5].y, 0, 0x4441)]));
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);
	ret ^= c0;
	return ret;
}

__device__ __forceinline__
static uint2 d_ROUND_ELT1(const uint2 sharedMemory[7][256],const uint2 *const __restrict__ in,const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const uint2 c0){
	uint2 ret = __ldg((uint2*)&b0[__byte_perm(in[i0].x, 0, 0x4440)]);
	ret ^= sharedMemory[1][__byte_perm(in[i1].x, 0, 0x4441)];
	ret ^= sharedMemory[2][__byte_perm(in[i2].x, 0, 0x4442)];
	ret ^= sharedMemory[3][__byte_perm(in[i3].x, 0, 0x4443)];
	ret ^= sharedMemory[4][__byte_perm(in[i4].y, 0, 0x4440)];
	ret ^= sharedMemory[5][__byte_perm(in[i5].y, 0, 0x4441)];
	ret ^= ROR8(__ldg((uint2*)&b7[__byte_perm(in[i6].y, 0, 0x4442)]));//sharedMemory[6][__byte_perm(in[i6].y, 0, 0x4442)]
	ret ^= __ldg((uint2*)&b7[__byte_perm(in[i7].y, 0, 0x4443)]);//sharedMemory[7][__byte_perm(in[i7].y, 0, 0x4443)]
	ret ^= c0;
	return ret;
}

//--------END OF WHIRLPOOL DEVICE MACROS-----------------------------------------------------------------------------

//--------START OF WHIRLPOOL HOST MACROS-----------------------------------------------------------------------------

#define table_skew(val,num) SPH_ROTL64(val,8*num)
#define BYTE(x, n)     ((unsigned)((x) >> (8 * (n))) & 0xFF)

#define ROUND_ELT(table, in, i0, i1, i2, i3, i4, i5, i6, i7) \
	(table[BYTE(in[i0], 0)] \
	^ table_skew(table[BYTE(in[i1], 1)], 1) \
	^ table_skew(table[BYTE(in[i2], 2)], 2) \
	^ table_skew(table[BYTE(in[i3], 3)], 3) \
	^ table_skew(table[BYTE(in[i4], 4)], 4) \
	^ table_skew(table[BYTE(in[i5], 5)], 5) \
	^ table_skew(table[BYTE(in[i6], 6)], 6) \
	^ table_skew(table[BYTE(in[i7], 7)], 7))

#define ROUND(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7)   do { \
		out[0] = ROUND_ELT(table, in, 0, 7, 6, 5, 4, 3, 2, 1) ^ c0; \
		out[1] = ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2) ^ c1; \
		out[2] = ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3) ^ c2; \
		out[3] = ROUND_ELT(table, in, 3, 2, 1, 0, 7, 6, 5, 4) ^ c3; \
		out[4] = ROUND_ELT(table, in, 4, 3, 2, 1, 0, 7, 6, 5) ^ c4; \
		out[5] = ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6) ^ c5; \
		out[6] = ROUND_ELT(table, in, 6, 5, 4, 3, 2, 1, 0, 7) ^ c6; \
		out[7] = ROUND_ELT(table, in, 7, 6, 5, 4, 3, 2, 1, 0) ^ c7; \
	} while (0)

__host__
static void ROUND_KSCHED(const uint64_t *in,uint64_t *out,const uint64_t c){
	const uint64_t *a = in;
	uint64_t *b = out;
	ROUND(old1_T0, a, b, c, 0, 0, 0, 0, 0, 0, 0);
}


//--------END OF WHIRLPOOL HOST MACROS-------------------------------------------------------------------------------

__host__
void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode)
{
	uint64_t* table0 = NULL;

	switch (mode) {
	case 0: /* x15 with rotated T1-T7 (based on T0) */
		table0 = (uint64_t*)plain_T0;
		cudaMemcpyToSymbol(InitVector_RC, plain_RC, 10*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(precomputed_round_key_64, plain_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		break;
	case 1: /* old whirlpool */
		table0 = (uint64_t*)old1_T0;
		cudaMemcpyToSymbol(InitVector_RC, old1_RC, 10*sizeof(uint64_t),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(precomputed_round_key_64, old1_precomputed_round_key_64, 72*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
		break;
	default:
		applog(LOG_ERR,"Bad whirlpool mode");
		exit(0);
	}
	cudaMemcpyToSymbol(b0, table0, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);
	uint64_t table7[256];
	for(int i=0;i<256;i++){
		table7[i] = ROTR64(table0[i],8);
	}
	cudaMemcpyToSymbol(b7, table7, 256*sizeof(uint64_t),0, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)));

	cuda_get_arch(thr_id);
}

__host__
static void whirl_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

__host__
void whirlpool512_setBlock_80(void *pdata, const void *ptarget)
{
	uint64_t PaddedMessage[16];

	memcpy(PaddedMessage, pdata, 80);
	memset(((uint8_t*)&PaddedMessage)+80, 0, 48);
	((uint8_t*)&PaddedMessage)[80] = 0x80; /* ending */

	// compute constant first block
	uint64_t midstate[16] = { 0 };
	whirl_midstate(midstate, pdata);
	memcpy(PaddedMessage, midstate, 64);

	uint64_t round_constants[80];
	uint64_t n[8];

	n[0] = PaddedMessage[0] ^ PaddedMessage[8];    //read data
	n[1] = PaddedMessage[1] ^ PaddedMessage[9];
	n[2] = PaddedMessage[2] ^ 0x0000000000000080; //whirlpool
	n[3] = PaddedMessage[3];
	n[4] = PaddedMessage[4];
	n[5] = PaddedMessage[5];
	n[6] = PaddedMessage[6];
	n[7] = PaddedMessage[7] ^ 0x8002000000000000;

	ROUND_KSCHED(PaddedMessage,round_constants,old1_RC[0]);

	for(int i=1;i<10;i++){
		ROUND_KSCHED(&round_constants[8*(i-1)],&round_constants[8*i],old1_RC[i]);
	}

	//USE the same memory place to store keys and state
	round_constants[ 0]^= old1_T0[BYTE(n[0], 0)]
	 ^ table_skew(old1_T0[BYTE(n[7], 1)], 1) ^ table_skew(old1_T0[BYTE(n[6], 2)], 2) ^ table_skew(old1_T0[BYTE(n[5], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[4], 4)], 4) ^ table_skew(old1_T0[BYTE(n[3], 5)], 5) ^ table_skew(old1_T0[BYTE(n[2], 6)], 6);

	round_constants[ 1]^= old1_T0[BYTE(n[1], 0)]
	 ^ table_skew(old1_T0[BYTE(n[0], 1)], 1) ^ table_skew(old1_T0[BYTE(n[7], 2)], 2) ^ table_skew(old1_T0[BYTE(n[6], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[5], 4)], 4) ^ table_skew(old1_T0[BYTE(n[4], 5)], 5) ^ table_skew(old1_T0[BYTE(n[3], 6)], 6)
	 ^ table_skew(old1_T0[BYTE(n[2], 7)], 7);

	round_constants[ 2]^= old1_T0[BYTE(n[2], 0)]
	 ^ table_skew(old1_T0[BYTE(n[1], 1)], 1) ^ table_skew(old1_T0[BYTE(n[0], 2)], 2) ^ table_skew(old1_T0[BYTE(n[7], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[6], 4)], 4) ^ table_skew(old1_T0[BYTE(n[5], 5)], 5) ^ table_skew(old1_T0[BYTE(n[4], 6)], 6)
	 ^ table_skew(old1_T0[BYTE(n[3], 7)], 7);

	round_constants[ 3]^= old1_T0[BYTE(n[3], 0)]
	 ^ table_skew(old1_T0[BYTE(n[2], 1)], 1) ^ table_skew(old1_T0[BYTE(n[1], 2)], 2) ^ table_skew(old1_T0[BYTE(n[0], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[7], 4)], 4) ^ table_skew(old1_T0[BYTE(n[6], 5)], 5) ^ table_skew(old1_T0[BYTE(n[5], 6)], 6)
	 ^ table_skew(old1_T0[BYTE(n[4], 7)], 7);

	round_constants[ 4]^= old1_T0[BYTE(n[4], 0)]
	 ^ table_skew(old1_T0[BYTE(n[3], 1)], 1) ^ table_skew(old1_T0[BYTE(n[2], 2)], 2) ^ table_skew(old1_T0[BYTE(n[1], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[0], 4)], 4) ^ table_skew(old1_T0[BYTE(n[7], 5)], 5) ^ table_skew(old1_T0[BYTE(n[6], 6)], 6)
	 ^ table_skew(old1_T0[BYTE(n[5], 7)], 7);

	round_constants[ 5]^= old1_T0[BYTE(n[5], 0)]
	 ^ table_skew(old1_T0[BYTE(n[4], 1)], 1) ^ table_skew(old1_T0[BYTE(n[3], 2)], 2) ^ table_skew(old1_T0[BYTE(n[2], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[0], 5)], 5) ^ table_skew(old1_T0[BYTE(n[7], 6)], 6) ^ table_skew(old1_T0[BYTE(n[6], 7)], 7);

	round_constants[ 6]^= old1_T0[BYTE(n[6], 0)]
	 ^ table_skew(old1_T0[BYTE(n[5], 1)], 1) ^ table_skew(old1_T0[BYTE(n[4], 2)], 2) ^ table_skew(old1_T0[BYTE(n[3], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[2], 4)], 4) ^ table_skew(old1_T0[BYTE(n[0], 6)], 6) ^ table_skew(old1_T0[BYTE(n[7], 7)], 7);

	round_constants[ 7]^= old1_T0[BYTE(n[7], 0)]
	 ^ table_skew(old1_T0[BYTE(n[6], 1)], 1) ^ table_skew(old1_T0[BYTE(n[5], 2)], 2) ^ table_skew(old1_T0[BYTE(n[4], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[3], 4)], 4) ^ table_skew(old1_T0[BYTE(n[2], 5)], 5) ^ table_skew(old1_T0[BYTE(n[0], 7)], 7);

	for(int i=1;i<5;i++)
		n[i] = round_constants[i];

	round_constants[ 8]^= table_skew(old1_T0[BYTE(n[4], 4)], 4)
	 ^ table_skew(old1_T0[BYTE(n[3], 5)], 5) ^ table_skew(old1_T0[BYTE(n[2], 6)], 6) ^ table_skew(old1_T0[BYTE(n[1], 7)], 7);

	round_constants[ 9]^= old1_T0[BYTE(n[1], 0)]
	 ^ table_skew(old1_T0[BYTE(n[4], 5)], 5) ^ table_skew(old1_T0[BYTE(n[3], 6)], 6) ^ table_skew(old1_T0[BYTE(n[2], 7)], 7);

	round_constants[10]^= old1_T0[BYTE(n[2], 0)]
	 ^ table_skew(old1_T0[BYTE(n[1], 1)], 1) ^ table_skew(old1_T0[BYTE(n[4], 6)], 6) ^ table_skew(old1_T0[BYTE(n[3], 7)], 7);

	round_constants[11]^= old1_T0[BYTE(n[3], 0)]
	 ^ table_skew(old1_T0[BYTE(n[2], 1)], 1) ^ table_skew(old1_T0[BYTE(n[1], 2)], 2) ^ table_skew(old1_T0[BYTE(n[4], 7)], 7);

	round_constants[12]^= old1_T0[BYTE(n[4], 0)]
	 ^ table_skew(old1_T0[BYTE(n[3], 1)], 1) ^ table_skew(old1_T0[BYTE(n[2], 2)], 2) ^ table_skew(old1_T0[BYTE(n[1], 3)], 3);

	round_constants[13]^= table_skew(old1_T0[BYTE(n[4], 1)], 1) ^ table_skew(old1_T0[BYTE(n[3], 2)], 2)
	 ^ table_skew(old1_T0[BYTE(n[2], 3)], 3) ^ table_skew(old1_T0[BYTE(n[1], 4)], 4);

	round_constants[14]^= table_skew(old1_T0[BYTE(n[4], 2)], 2) ^ table_skew(old1_T0[BYTE(n[3], 3)], 3)
	 ^ table_skew(old1_T0[BYTE(n[2], 4)], 4) ^ table_skew(old1_T0[BYTE(n[1], 5)], 5);

	round_constants[15]^= table_skew(old1_T0[BYTE(n[4], 3)], 3) ^  table_skew(old1_T0[BYTE(n[3], 4)], 4)
	 ^ table_skew(old1_T0[BYTE(n[2], 5)], 5) ^ table_skew(old1_T0[BYTE(n[1], 6)], 6);

	PaddedMessage[0] ^= PaddedMessage[8];

	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 128, 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(precomputed_round_key_80, round_constants, 80*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__
extern void x15_whirlpool_cpu_free(int thr_id)
{
	if (d_resNonce[thr_id])
		cudaFree(d_resNonce[thr_id]);
}

__global__
__launch_bounds__(TPB80,2)
void oldwhirlpool_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t* resNonce, const uint64_t target)
{
	__shared__ uint2 sharedMemory[7][256];

	if (threadIdx.x < 256) {
		const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]);
		sharedMemory[0][threadIdx.x] = tmp;
		sharedMemory[1][threadIdx.x] = ROL8(tmp);
		sharedMemory[2][threadIdx.x] = ROL16(tmp);
		sharedMemory[3][threadIdx.x] = ROL24(tmp);
		sharedMemory[4][threadIdx.x] = SWAPUINT2(tmp);
		sharedMemory[5][threadIdx.x] = ROR24(tmp);
		sharedMemory[6][threadIdx.x] = ROR16(tmp);
	}

	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){

		uint2 hash[8], state[8],n[8], tmp[8];
		uint32_t nonce = cuda_swab32(startNounce + thread);
		uint2 temp = c_PaddedMessage80[9];
		temp.y = nonce;

		/// round 2 ///////
		//////////////////////////////////
		temp = temp ^ c_PaddedMessage80[1];

		*(uint2x4*)&n[ 0]   = *(uint2x4*)&precomputed_round_key_80[ 0];
		*(uint2x4*)&n[ 4]   = *(uint2x4*)&precomputed_round_key_80[ 4];
		*(uint2x4*)&tmp[ 0] = *(uint2x4*)&precomputed_round_key_80[ 8];
		*(uint2x4*)&tmp[ 4] = *(uint2x4*)&precomputed_round_key_80[12];

		n[ 0]^= __ldg((uint2*)&b7[__byte_perm(temp.y, 0, 0x4443)]);
		n[ 5]^= sharedMemory[4][__byte_perm(temp.y, 0, 0x4440)];
		n[ 6]^= sharedMemory[5][__byte_perm(temp.y, 0, 0x4441)];
		n[ 7]^= sharedMemory[6][__byte_perm(temp.y, 0, 0x4442)];

		tmp[ 0]^= __ldg((uint2*)&b0[__byte_perm(n[0].x, 0, 0x4440)]);
		tmp[ 0]^= sharedMemory[1][__byte_perm(n[7].x, 0, 0x4441)];
		tmp[ 0]^= sharedMemory[2][__byte_perm(n[6].x, 0, 0x4442)];
		tmp[ 0]^= sharedMemory[3][__byte_perm(n[5].x, 0, 0x4443)];

		tmp[ 1]^= sharedMemory[1][__byte_perm(n[0].x, 0, 0x4441)];
		tmp[ 1]^= sharedMemory[2][__byte_perm(n[7].x, 0, 0x4442)];
		tmp[ 1]^= sharedMemory[3][__byte_perm(n[6].x, 0, 0x4443)];
		tmp[ 1]^= sharedMemory[4][__byte_perm(n[5].y, 0, 0x4440)];

		tmp[ 2]^= sharedMemory[2][__byte_perm(n[0].x, 0, 0x4442)];
		tmp[ 2]^= sharedMemory[3][__byte_perm(n[7].x, 0, 0x4443)];
		tmp[ 2]^= sharedMemory[4][__byte_perm(n[6].y, 0, 0x4440)];
		tmp[ 2]^= sharedMemory[5][__byte_perm(n[5].y, 0, 0x4441)];

		tmp[ 3]^= sharedMemory[3][__byte_perm(n[0].x, 0, 0x4443)];
		tmp[ 3]^= sharedMemory[4][__byte_perm(n[7].y, 0, 0x4440)];
		tmp[ 3]^= ROR24(__ldg((uint2*)&b0[__byte_perm(n[6].y, 0, 0x4441)]));
		tmp[ 3]^= ROR8(__ldg((uint2*)&b7[__byte_perm(n[5].y, 0, 0x4442)]));

		tmp[ 4]^= sharedMemory[4][__byte_perm(n[0].y, 0, 0x4440)];
		tmp[ 4]^= sharedMemory[5][__byte_perm(n[7].y, 0, 0x4441)];
		tmp[ 4]^= ROR8(__ldg((uint2*)&b7[__byte_perm(n[6].y, 0, 0x4442)]));
		tmp[ 4]^= __ldg((uint2*)&b7[__byte_perm(n[5].y, 0, 0x4443)]);

		tmp[ 5]^= __ldg((uint2*)&b0[__byte_perm(n[5].x, 0, 0x4440)]);
		tmp[ 5]^= sharedMemory[5][__byte_perm(n[0].y, 0, 0x4441)];
		tmp[ 5]^= sharedMemory[6][__byte_perm(n[7].y, 0, 0x4442)];
		tmp[ 5]^= __ldg((uint2*)&b7[__byte_perm(n[6].y, 0, 0x4443)]);

		tmp[ 6]^= __ldg((uint2*)&b0[__byte_perm(n[6].x, 0, 0x4440)]);
		tmp[ 6]^= sharedMemory[1][__byte_perm(n[5].x, 0, 0x4441)];
		tmp[ 6]^= sharedMemory[6][__byte_perm(n[0].y, 0, 0x4442)];
		tmp[ 6]^= __ldg((uint2*)&b7[__byte_perm(n[7].y, 0, 0x4443)]);

		tmp[ 7]^= __ldg((uint2*)&b0[__byte_perm(n[7].x, 0, 0x4440)]);
		tmp[ 7]^= sharedMemory[1][__byte_perm(n[6].x, 0, 0x4441)];
		tmp[ 7]^= sharedMemory[2][__byte_perm(n[5].x, 0, 0x4442)];
		tmp[ 7]^= __ldg((uint2*)&b7[__byte_perm(n[0].y, 0, 0x4443)]);

		TRANSFER(n, tmp);

		for (int i=2; i<10; i++) {
			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_80[i*8+0]);
			tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_80[i*8+1]);
			tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_80[i*8+2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_80[i*8+3]);
			tmp[ 4] = d_ROUND_ELT1_LDG(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_80[i*8+4]);
			tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_80[i*8+5]);
			tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_80[i*8+6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_80[i*8+7]);
			TRANSFER(n, tmp);
		}

		state[0] = c_PaddedMessage80[0] ^ n[0];
		state[1] = c_PaddedMessage80[1] ^ n[1] ^ vectorize(REPLACE_HIDWORD(devectorize(c_PaddedMessage80[9]),nonce));
		state[2] = c_PaddedMessage80[2] ^ n[2] ^ vectorize(0x0000000000000080);
		state[3] = c_PaddedMessage80[3] ^ n[3];
		state[4] = c_PaddedMessage80[4] ^ n[4];
		state[5] = c_PaddedMessage80[5] ^ n[5];
		state[6] = c_PaddedMessage80[6] ^ n[6];
		state[7] = c_PaddedMessage80[7] ^ n[7] ^ vectorize(0x8002000000000000);

		#pragma unroll 2
		for(int r=0;r<2;r++){
			#pragma unroll 8
			for(int i=0;i<8;i++)
				hash[ i] = n[ i] = state[ i];

			uint2 h[8] = {
				{0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
				{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
			};

			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, h[0]);
			tmp[ 1] = d_ROUND_ELT1(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, h[1]);
			tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, h[2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, h[3]);
			tmp[ 4] = d_ROUND_ELT1(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, h[4]);
			tmp[ 5] = d_ROUND_ELT1_LDG(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, h[5]);
			tmp[ 6] = d_ROUND_ELT1(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, h[6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, h[7]);
			TRANSFER(n, tmp);
	//		#pragma unroll 10
			for (int i=1; i <10; i++){
				tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i-1)*8+0]);
				tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i-1)*8+1]);
				tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i-1)*8+2]);
				tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i-1)*8+3]);
				tmp[ 4] = d_ROUND_ELT1(    sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i-1)*8+4]);
				tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i-1)*8+5]);
				tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i-1)*8+6]);
				tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i-1)*8+7]);
				TRANSFER(n, tmp);
			}
			#pragma unroll 8
			for (int i=0; i<8; i++)
				state[i] = n[i] ^ hash[i];

			#pragma unroll 6
			for (int i=1; i<7; i++)
				n[i]=vectorize(0);

			n[0] = vectorize(0x80);
			n[7] = vectorize(0x2000000000000);

			#pragma unroll 8
			for (int i=0; i < 8; i++) {
				h[i] = state[i];
				n[i] = n[i] ^ h[i];
			}

	//		#pragma unroll 10
			for (int i=0; i < 10; i++) {
				tmp[ 0] = d_ROUND_ELT1(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1, InitVector_RC[i]);
				tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
				tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
				tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
				tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
				tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
				tmp[ 6] = d_ROUND_ELT_LDG(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
				tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
				TRANSFER(h, tmp);
				tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
				tmp[ 1] = d_ROUND_ELT1(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
				tmp[ 2] = d_ROUND_ELT1_LDG(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
				tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
				tmp[ 4] = d_ROUND_ELT1(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
				tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
				tmp[ 6] = d_ROUND_ELT1(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
				tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
				TRANSFER(n, tmp);
			}

			state[0] = xor3x(state[0], n[0], vectorize(0x80));
			state[1] = state[1]^ n[1];
			state[2] = state[2]^ n[2];
			state[3] = state[3]^ n[3];
			state[4] = state[4]^ n[4];
			state[5] = state[5]^ n[5];
			state[6] = state[6]^ n[6];
			state[7] = xor3x(state[7], n[7], vectorize(0x2000000000000));
		}

		uint2 h[8] = {
			{0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
			{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
		};

		#pragma unroll 8
		for(int i=0;i<8;i++)
			n[i]=hash[i] = state[ i];

		tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, h[0]);
		tmp[ 1] = d_ROUND_ELT1_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, h[1]);
		tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, h[2]);
		tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, h[3]);
		tmp[ 4] = d_ROUND_ELT1(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, h[4]);
		tmp[ 5] = d_ROUND_ELT1_LDG(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, h[5]);
		tmp[ 6] = d_ROUND_ELT1(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, h[6]);
		tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, h[7]);
		TRANSFER(n, tmp);
//		#pragma unroll 10
		for (int i=1; i <10; i++){
			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i-1)*8+0]);
			tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i-1)*8+1]);
			tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i-1)*8+2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i-1)*8+3]);
			tmp[ 4] = d_ROUND_ELT1(    sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i-1)*8+4]);
			tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i-1)*8+5]);
			tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i-1)*8+6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i-1)*8+7]);
			TRANSFER(n, tmp);
		}

		#pragma unroll 8
		for (int i=0; i<8; i++)
			n[ i] = h[i] = n[i] ^ hash[i];

		uint2 backup = h[ 3];

		n[0]^= vectorize(0x80);
		n[7]^= vectorize(0x2000000000000);

//		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			tmp[ 0] = d_ROUND_ELT1(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1, InitVector_RC[i]);
			tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[ 6] = d_ROUND_ELT_LDG(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[ 1] = d_ROUND_ELT1(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[ 2] = d_ROUND_ELT1_LDG(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[ 4] = d_ROUND_ELT1(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[ 6] = d_ROUND_ELT1(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}
		tmp[ 0] = d_ROUND_ELT1(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1, InitVector_RC[8]);
		tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[ 6] = d_ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
		TRANSFER(h, tmp);
		tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
		tmp[ 1] = d_ROUND_ELT1(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
		tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
		tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
		tmp[ 4] = d_ROUND_ELT1(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
		tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
		tmp[ 6] = d_ROUND_ELT1_LDG(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
		tmp[ 7] = d_ROUND_ELT1(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);

		n[ 3] = backup ^ d_ROUND_ELT(sharedMemory,  h, 3, 2, 1, 0, 7, 6, 5, 4)
			^ d_ROUND_ELT(sharedMemory,tmp, 3, 2, 1, 0, 7, 6, 5, 4);

		if(devectorize(n[3]) <= target) {
			uint32_t tmp = atomicExch(&resNonce[0], thread);
			if (tmp != UINT32_MAX)
				resNonce[1] = tmp;
		}

	} // thread < threads
}

/* only for whirlpool algo, no data out!! */
__host__
void whirlpool512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *h_resNonces, const uint64_t target)
{
	dim3 grid((threads + TPB80-1) / TPB80);
	dim3 block(TPB80);

	cudaMemset(d_resNonce[thr_id], 0xff, 2*sizeof(uint32_t));

	oldwhirlpool_gpu_hash_80<<<grid, block>>>(threads, startNounce, d_resNonce[thr_id], target);

	cudaMemcpy(h_resNonces, d_resNonce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (h_resNonces[0] != UINT32_MAX) h_resNonces[0] += startNounce;
	if (h_resNonces[1] != UINT32_MAX) h_resNonces[1] += startNounce;
}

__global__
__launch_bounds__(TPB64,2)
void x15_whirlpool_gpu_hash_64(uint32_t threads, uint64_t *g_hash)
{
	__shared__ uint2 sharedMemory[7][256];

	if (threadIdx.x < 256) {
		const uint2 tmp = __ldg((uint2*)&b0[threadIdx.x]);
		sharedMemory[0][threadIdx.x] = tmp;
		sharedMemory[1][threadIdx.x] = ROL8(tmp);
		sharedMemory[2][threadIdx.x] = ROL16(tmp);
		sharedMemory[3][threadIdx.x] = ROL24(tmp);
		sharedMemory[4][threadIdx.x] = SWAPUINT2(tmp);
		sharedMemory[5][threadIdx.x] = ROR24(tmp);
		sharedMemory[6][threadIdx.x] = ROR16(tmp);
	}

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads){

		uint2 hash[8], n[8], h[ 8];
		uint2 tmp[8] = {
			{0xC0EE0B30,0x672990AF},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},
			{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828},{0x28282828,0x28282828}
		};

		*(uint2x4*)&hash[ 0] = __ldg4((uint2x4*)&g_hash[(thread<<3) + 0]);
		*(uint2x4*)&hash[ 4] = __ldg4((uint2x4*)&g_hash[(thread<<3) + 4]);

		__syncthreads();

		#pragma unroll 8
		for(int i=0;i<8;i++)
			n[i]=hash[i];

		tmp[ 0]^= d_ROUND_ELT(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1);
		tmp[ 1]^= d_ROUND_ELT_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[ 2]^= d_ROUND_ELT(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[ 3]^= d_ROUND_ELT_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[ 4]^= d_ROUND_ELT(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[ 5]^= d_ROUND_ELT_LDG(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[ 6]^= d_ROUND_ELT(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[ 7]^= d_ROUND_ELT_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0);
		for (int i=1; i <10; i++){
			TRANSFER(n, tmp);
			tmp[ 0] = d_ROUND_ELT1_LDG(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, precomputed_round_key_64[(i-1)*8+0]);
			tmp[ 1] = d_ROUND_ELT1(    sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, precomputed_round_key_64[(i-1)*8+1]);
			tmp[ 2] = d_ROUND_ELT1(    sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, precomputed_round_key_64[(i-1)*8+2]);
			tmp[ 3] = d_ROUND_ELT1_LDG(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, precomputed_round_key_64[(i-1)*8+3]);
			tmp[ 4] = d_ROUND_ELT1(    sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, precomputed_round_key_64[(i-1)*8+4]);
			tmp[ 5] = d_ROUND_ELT1(    sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, precomputed_round_key_64[(i-1)*8+5]);
			tmp[ 6] = d_ROUND_ELT1(    sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, precomputed_round_key_64[(i-1)*8+6]);
			tmp[ 7] = d_ROUND_ELT1_LDG(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, precomputed_round_key_64[(i-1)*8+7]);
		}

		TRANSFER(h, tmp);
		#pragma unroll 8
		for (int i=0; i<8; i++)
			hash[ i] = h[i] = h[i] ^ hash[i];

		#pragma unroll 6
		for (int i=1; i<7; i++)
			n[i]=vectorize(0);

		n[0] = vectorize(0x80);
		n[7] = vectorize(0x2000000000000);

		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			n[i] = n[i] ^ h[i];
		}

//		#pragma unroll 10
		for (int i=0; i < 10; i++) {
			tmp[ 0] = InitVector_RC[i];
			tmp[ 0]^= d_ROUND_ELT(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1);
			tmp[ 1] = d_ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
			tmp[ 2] = d_ROUND_ELT_LDG(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
			tmp[ 3] = d_ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[ 4] = d_ROUND_ELT_LDG(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
			tmp[ 5] = d_ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[ 6] = d_ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
			tmp[ 7] = d_ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);
			TRANSFER(h, tmp);
			tmp[ 0] = d_ROUND_ELT1(sharedMemory,n, 0, 7, 6, 5, 4, 3, 2, 1, tmp[0]);
			tmp[ 1] = d_ROUND_ELT1_LDG(sharedMemory,n, 1, 0, 7, 6, 5, 4, 3, 2, tmp[1]);
			tmp[ 2] = d_ROUND_ELT1(sharedMemory,n, 2, 1, 0, 7, 6, 5, 4, 3, tmp[2]);
			tmp[ 3] = d_ROUND_ELT1(sharedMemory,n, 3, 2, 1, 0, 7, 6, 5, 4, tmp[3]);
			tmp[ 4] = d_ROUND_ELT1_LDG(sharedMemory,n, 4, 3, 2, 1, 0, 7, 6, 5, tmp[4]);
			tmp[ 5] = d_ROUND_ELT1(sharedMemory,n, 5, 4, 3, 2, 1, 0, 7, 6, tmp[5]);
			tmp[ 6] = d_ROUND_ELT1_LDG(sharedMemory,n, 6, 5, 4, 3, 2, 1, 0, 7, tmp[6]);
			tmp[ 7] = d_ROUND_ELT1(sharedMemory,n, 7, 6, 5, 4, 3, 2, 1, 0, tmp[7]);
			TRANSFER(n, tmp);
		}

		hash[0] = xor3x(hash[0], n[0], vectorize(0x80));
		hash[1] = hash[1]^ n[1];
		hash[2] = hash[2]^ n[2];
		hash[3] = hash[3]^ n[3];
		hash[4] = hash[4]^ n[4];
		hash[5] = hash[5]^ n[5];
		hash[6] = hash[6]^ n[6];
		hash[7] = xor3x(hash[7], n[7], vectorize(0x2000000000000));

		*(uint2x4*)&g_hash[(thread<<3)+ 0] = *(uint2x4*)&hash[ 0];
		*(uint2x4*)&g_hash[(thread<<3)+ 4] = *(uint2x4*)&hash[ 4];
	}
}

__host__
static void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB64-1) / TPB64);
	dim3 block(TPB64);

	x15_whirlpool_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash);
}

__host__
void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	x15_whirlpool_cpu_hash_64(thr_id, threads, d_hash);
}

