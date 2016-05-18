/**
 * Blake2-S 256 CUDA implementation
 * @author tpruvot@github March 2016
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <memory.h>

#include "miner.h"

extern "C" {
#define NATIVE_LITTLE_ENDIAN
#include <sph/blake2s.h>
}

//#define GPU_MIDSTATE
#define MIDLEN 76
#define A 64

static __thread blake2s_state ALIGN(A) s_midstate;
static __thread blake2s_state ALIGN(A) s_ctx;

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

#ifndef GPU_MIDSTATE
__constant__ uint2 d_data[10];
#else
__constant__ blake2s_state ALIGN(8) d_state[1];
#endif

/* 16 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* threads per block */
#define TPB 512

/* max count of found nonces in one call */
#define NBN 2
#if NBN > 1
static uint32_t extra_results[NBN] = { UINT32_MAX };
#endif

extern "C" void blake2s_hash(void *output, const void *input)
{
	uint8_t _ALIGN(A) hash[BLAKE2S_OUTBYTES];
	blake2s_state blake2_ctx;

	blake2s_init(&blake2_ctx, BLAKE2S_OUTBYTES);
	blake2s_update(&blake2_ctx, (uint8_t*) input, 80);
	blake2s_final(&blake2_ctx, hash, BLAKE2S_OUTBYTES);

	memcpy(output, hash, 32);
}

__host__
inline void blake2s_hash_end(uint32_t *output, const uint32_t *input)
{
	s_ctx.buflen = MIDLEN;
	memcpy(&s_ctx, &s_midstate, 32 + 16 + MIDLEN);
	blake2s_update(&s_ctx, (uint8_t*) &input[MIDLEN/4], 80-MIDLEN);
	blake2s_final(&s_ctx, (uint8_t*) output, BLAKE2S_OUTBYTES);
}

__host__
void blake2s_setBlock(uint32_t *penddata, blake2s_state *pstate)
{
#ifndef GPU_MIDSTATE
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, penddata, 80, 0, cudaMemcpyHostToDevice));
#else
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_state, pstate, sizeof(blake2s_state), 0, cudaMemcpyHostToDevice));
#endif
}

__device__ __forceinline__
uint64_t gpu_load64(void *src) {
	return *(uint64_t*)(src);
}

__device__ __forceinline__
void gpu_store32(void *dst, uint32_t dw) {
	*(uint32_t*)(dst) = dw;
}

__device__ __forceinline__
void gpu_store64(void *dst, uint64_t lw) {
	*(uint64_t*)(dst) = lw;
}

__device__ __forceinline__
void gpu_blake2s_set_lastnode(blake2s_state *S) {
	S->f[1] = ~0U;
}

__device__ __forceinline__
void gpu_blake2s_clear_lastnode(blake2s_state *S) {
	S->f[1] = 0U;
}

__device__ __forceinline__
void gpu_blake2s_increment_counter(blake2s_state *S, const uint32_t inc)
{
	S->t[0] += inc;
	S->t[1] += ( S->t[0] < inc );
}

__device__ __forceinline__
void gpu_blake2s_set_lastblock(blake2s_state *S)
{
	if (S->last_node) gpu_blake2s_set_lastnode(S);
	S->f[0] = ~0U;
}

__device__
void gpu_blake2s_compress(blake2s_state *S, const uint32_t *block)
{
	uint32_t m[16];
	uint32_t v[16];

	const uint32_t blake2s_IV[8] = {
		0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
		0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL
	};

	const uint8_t blake2s_sigma[10][16] = {
		{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
		{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
		{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
		{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
		{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
		{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
		{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
		{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
		{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
		{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
	};

	#pragma unroll
	for(int i = 0; i < 16; i++)
		m[i] = block[i];

	#pragma unroll
	for(int i = 0; i < 8; i++)
		v[i] = S->h[i];

	v[ 8] = blake2s_IV[0];
	v[ 9] = blake2s_IV[1];
	v[10] = blake2s_IV[2];
	v[11] = blake2s_IV[3];
	v[12] = S->t[0] ^ blake2s_IV[4];
	v[13] = S->t[1] ^ blake2s_IV[5];
	v[14] = S->f[0] ^ blake2s_IV[6];
	v[15] = S->f[1] ^ blake2s_IV[7];

	#define G(r,i,a,b,c,d) { \
		a += b + m[blake2s_sigma[r][2*i+0]]; \
		d = __byte_perm(d ^ a, 0, 0x1032); /* d = ROTR32(d ^ a, 16); */ \
		c = c + d; \
		b = ROTR32(b ^ c, 12); \
		a += b + m[blake2s_sigma[r][2*i+1]]; \
		d = __byte_perm(d ^ a, 0, 0x0321); /* ROTR32(d ^ a, 8); */ \
		c = c + d; \
		b = ROTR32(b ^ c, 7); \
	}

	#define ROUND(r) { \
		G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \
		G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \
		G(r,2,v[ 2],v[ 6],v[10],v[14]); \
		G(r,3,v[ 3],v[ 7],v[11],v[15]); \
		G(r,4,v[ 0],v[ 5],v[10],v[15]); \
		G(r,5,v[ 1],v[ 6],v[11],v[12]); \
		G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \
		G(r,7,v[ 3],v[ 4],v[ 9],v[14]); \
	}

	ROUND( 0 );
	ROUND( 1 );
	ROUND( 2 );
	ROUND( 3 );
	ROUND( 4 );
	ROUND( 5 );
	ROUND( 6 );
	ROUND( 7 );
	ROUND( 8 );
	ROUND( 9 );

	#pragma unroll
	for(int i = 0; i < 8; i++)
		S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];

	#undef G
	#undef ROUND
}

#if 0
/* unused but kept as reference */
__device__ __forceinline__
void gpu_blake2s_update(blake2s_state *S, const uint8_t *in, uint64_t inlen)
{
	while(inlen > 0)
	{
		const int left = S->buflen;
		size_t fill = 2 * BLAKE2S_BLOCKBYTES - left;
		if(inlen > fill)
		{
			memcpy(S->buf + left, in, fill); // Fill buffer
			S->buflen += fill;

			gpu_blake2s_increment_counter(S, BLAKE2S_BLOCKBYTES);
			gpu_blake2s_compress(S, (uint32_t*) S->buf); // Compress
			memcpy(S->buf, S->buf + BLAKE2S_BLOCKBYTES, BLAKE2S_BLOCKBYTES); // Shift buffer left
			S->buflen -= BLAKE2S_BLOCKBYTES;
			in += fill;
			inlen -= fill;
		}
		else // inlen <= fill
		{
			memcpy(S->buf + left, in, (size_t) inlen);
			S->buflen += (size_t) inlen; // Be lazy, do not compress
			in += inlen;
			inlen -= inlen;
		}
	}
}
#endif

#ifndef GPU_MIDSTATE
__device__ __forceinline__
void gpu_blake2s_fill_data(blake2s_state *S, const uint32_t nonce)
{
	uint2 *b2 = (uint2*) S->buf;
	#pragma unroll
	for (int i=0; i < 9; i++)
		b2[i] = d_data[i];
	b2[9].x = d_data[9].x;
	b2[9].y = nonce;
	S->buflen = 80;
}
#endif

__device__ __forceinline__
void gpu_blake2s_update_nonce(blake2s_state *S, const uint32_t nonce)
{
	gpu_store32(&S->buf[76], nonce);
	S->buflen = 80;
}

__device__ __forceinline__
uint2 gpu_blake2s_final(blake2s_state *S)
{
	//if (S->buflen > BLAKE2S_BLOCKBYTES)
	{
		gpu_blake2s_increment_counter(S, BLAKE2S_BLOCKBYTES);
		gpu_blake2s_compress(S, (uint32_t*) S->buf);
		S->buflen -= BLAKE2S_BLOCKBYTES;
		//memcpy(S->buf, S->buf + BLAKE2S_BLOCKBYTES, S->buflen);
	}

	gpu_blake2s_increment_counter(S, (uint32_t)S->buflen);
	gpu_blake2s_set_lastblock(S);
	//memset(&S->buf[S->buflen], 0, 2 * BLAKE2S_BLOCKBYTES - S->buflen); /* Padding */
	gpu_blake2s_compress(S, (uint32_t*) (S->buf + BLAKE2S_BLOCKBYTES));

	//#pragma unroll
	//for (int i = 0; i < 8; i++)
	//	out[i] = S->h[i];
	return make_uint2(S->h[6], S->h[7]);
}

/* init2 xors IV with input parameter block */
__device__ __forceinline__
void gpu_blake2s_init_param(blake2s_state *S, const blake2s_param *P)
{
	//blake2s_IV
	S->h[0] = 0x6A09E667UL;
	S->h[1] = 0xBB67AE85UL;
	S->h[2] = 0x3C6EF372UL;
	S->h[3] = 0xA54FF53AUL;
	S->h[4] = 0x510E527FUL;
	S->h[5] = 0x9B05688CUL;
	S->h[6] = 0x1F83D9ABUL;
	S->h[7] = 0x5BE0CD19UL;

	S->t[0] = 0; S->t[1] = 0;
	S->f[0] = 0; S->f[1] = 0;
	S->last_node = 0;

	S->buflen = 0;

	#pragma unroll
	for (int i = 8; i < sizeof(S->buf)/8; i++)
		gpu_store64(S->buf + (8*i), 0);

	uint64_t *p = (uint64_t*) P;

	/* IV XOR ParamBlock */
	#pragma unroll
	for (int i = 0; i < 4; i++)
		S->h[i] ^= gpu_load64(&p[i]);
}

// Sequential blake2s initialization
__device__ __forceinline__
void gpu_blake2s_init(blake2s_state *S, const uint8_t outlen)
{
	blake2s_param P[1];

	// if (!outlen || outlen > BLAKE2S_OUTBYTES) return;

	P->digest_length = outlen;
	P->key_length    = 0;
	P->fanout        = 1;
	P->depth         = 1;

	P->leaf_length = 0;
	gpu_store64(P->node_offset, 0);
	//P->node_depth    = 0;
	//P->inner_length  = 0;

	gpu_store64(&P->salt, 0);
	gpu_store64(&P->personal, 0);

	gpu_blake2s_init_param(S, P);
}

__device__ __forceinline__
void gpu_copystate(blake2s_state *dst, blake2s_state *src)
{
	uint64_t* d64 = (uint64_t*) dst;
	uint64_t* s64 = (uint64_t*) src;
	#pragma unroll
	for (int i=0; i < (32 + 16 + 2 * BLAKE2S_BLOCKBYTES)/8; i++)
		gpu_store64(&d64[i], s64[i]);
	dst->buflen = src->buflen;
	dst->last_node = src->last_node;
}

__global__
void blake2s_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2, const int swap)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = swap ? cuda_swab32(startNonce + thread) : startNonce + thread;
	blake2s_state ALIGN(8) blake2_ctx;

#ifndef GPU_MIDSTATE
	gpu_blake2s_init(&blake2_ctx, BLAKE2S_OUTBYTES);
	//gpu_blake2s_update(&blake2_ctx, (uint8_t*) d_data, 76);
	gpu_blake2s_fill_data(&blake2_ctx, nonce);
#else
	gpu_copystate(&blake2_ctx, &d_state[0]);
	gpu_blake2s_update_nonce(&blake2_ctx, nonce);
#endif

	uint2 h2 = gpu_blake2s_final(&blake2_ctx);
	if (h2.y <= target2.y && h2.x <= target2.x) {
#if NBN == 2
		if (resNonce[0] != UINT32_MAX)
			resNonce[1] = nonce;
		else
			resNonce[0] = nonce;
#else
		resNonce[0] = nonce;
#endif
	}
}

static __inline uint32_t swab32_if(uint32_t val, bool iftrue) {
	return iftrue ? swab32(val) : val;
}

__host__
uint32_t blake2s_hash_cuda(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint2 target2, const int swap)
{
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	blake2s_gpu_hash <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id], target2, swap);
	cudaThreadSynchronize();

	if (cudaSuccess == cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = swab32_if(h_resNonce[thr_id][0], swap);
#if NBN > 1
		for (int n=0; n < (NBN-1); n++)
			extra_results[n] = swab32_if(h_resNonce[thr_id][n+1], swap);
#endif
	}
	return result;
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_blake2s(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const int swap = 1; // to toggle nonce endian

	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 28 : 25;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark) {
		ptarget[6] = swab32(0xFFFF0);
		ptarget[7] = 0;
	}

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}

	for (int i=0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}

	// midstate
	memset(s_midstate.buf, 0, sizeof(s_midstate.buf));
	blake2s_init(&s_midstate, BLAKE2S_OUTBYTES);
	blake2s_update(&s_midstate, (uint8_t*) endiandata, MIDLEN);
	memcpy(&s_ctx, &s_midstate, sizeof(blake2s_state));

	blake2s_setBlock(endiandata, &s_midstate);

	const uint2 target = make_uint2(ptarget[6], ptarget[7]);

	do {
		uint32_t foundNonce = blake2s_hash_cuda(thr_id, throughput, pdata[19], target, swap);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (foundNonce != UINT32_MAX)
		{
			uint32_t _ALIGN(A) vhashcpu[8];

			//blake2s_hash(vhashcpu, endiandata);
			endiandata[19] = swab32_if(foundNonce, swap);
			blake2s_hash_end(vhashcpu, endiandata);

			if (vhashcpu[7] <= target.y && fulltest(vhashcpu, ptarget)) {
				work_set_target_ratio(work, vhashcpu);
				pdata[19] = work->nonces[0] = swab32_if(foundNonce, !swap);
#if NBN > 1
				if (extra_results[0] != UINT32_MAX) {
					endiandata[19] = swab32_if(extra_results[0], swap);
					blake2s_hash_end(vhashcpu, endiandata);
					if (vhashcpu[7] <= target.y && fulltest(vhashcpu, ptarget)) {
						work->nonces[1] = swab32_if(extra_results[0], !swap);
						if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio) {
							work_set_target_ratio(work, vhashcpu);
							xchg(work->nonces[1], pdata[19]);
						}
						return 2;
					}
					extra_results[0] = UINT32_MAX;
				}
#endif
				return 1;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + pdata[19]);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_blake2s(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
