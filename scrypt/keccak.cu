//
//  =============== KECCAK part on nVidia GPU ======================
//
// The keccak512 (SHA-3) is used in the PBKDF2 for scrypt-jane coins
// in place of the SHA2 based PBKDF2 used in scrypt coins.
//
// NOTE: compile this .cu module for compute_20,sm_20 with --maxrregcount=64
//

#include <map>

#include "miner.h"
#include "cuda_helper.h"

#include "keccak.h"
#include "salsa_kernel.h"

// define some error checking macros
#define DELIMITER '/'
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#undef checkCudaErrors
#define checkCudaErrors(x) \
{ \
	cudaGetLastError(); \
	x; \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess && !abort_flag) \
		applog(LOG_ERR, "GPU #%d: cudaError %d (%s) (%s line %d)\n", device_map[thr_id], err, cudaGetErrorString(err), __FILENAME__, __LINE__); \
}

// from salsa_kernel.cu
extern std::map<int, uint32_t *> context_idata[2];
extern std::map<int, uint32_t *> context_odata[2];
extern std::map<int, cudaStream_t> context_streams[2];
extern std::map<int, uint32_t *> context_hash[2];

#ifndef ROTL64
#define ROTL64(a,b) (((a) << (b)) | ((a) >> (64 - b)))
#endif

// CB
#define U32TO64_LE(p) \
	(((uint64_t)(*p)) | (((uint64_t)(*(p + 1))) << 32))

#define U64TO32_LE(p, v) \
	*p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);

static __device__ void mycpy64(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
	for (int k=0; k < 16; ++k) d[k] = s[k];
}

static __device__ void mycpy56(uint32_t *d, const uint32_t *s) {
#pragma unroll 14
	for (int k=0; k < 14; ++k) d[k] = s[k];
}

static __device__ void mycpy32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
	for (int k=0; k < 8; ++k) d[k] = s[k];
}

static __device__ void mycpy8(uint32_t *d, const uint32_t *s) {
#pragma unroll 2
	for (int k=0; k < 2; ++k) d[k] = s[k];
}

static __device__ void mycpy4(uint32_t *d, const uint32_t *s) {
	*d = *s;
}

// ---------------------------- BEGIN keccak functions ------------------------------------

#define KECCAK_HASH "Keccak-512"

typedef struct keccak_hash_state_t {
	uint64_t state[25];                        // 25*2
	uint32_t buffer[72/4];                     // 72
} keccak_hash_state;

__device__ void statecopy0(keccak_hash_state *d, keccak_hash_state *s)
{
#pragma unroll 25
	for (int i=0; i < 25; ++i)
		d->state[i] = s->state[i];
}

__device__ void statecopy8(keccak_hash_state *d, keccak_hash_state *s)
{
#pragma unroll 25
	for (int i=0; i < 25; ++i)
		d->state[i] = s->state[i];
#pragma unroll 2
	for (int i=0; i < 2; ++i)
		d->buffer[i] = s->buffer[i];
}

static const uint64_t host_keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

__constant__ uint64_t c_keccak_round_constants[24];
__constant__ uint32_t c_data[20];

__device__
void keccak_block(keccak_hash_state *S, const uint32_t *in)
{
	uint64_t *s = S->state, t[5], u[5], v, w;

	/* absorb input */
	#pragma unroll 9
	for (int i = 0; i < 72 / 8; i++, in += 2)
		s[i] ^= U32TO64_LE(in);

	for (int i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[ 1];
		s[ 1] = ROTL64(s[ 6], 44);
		s[ 6] = ROTL64(s[ 9], 20);
		s[ 9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[ 2], 62);
		s[ 2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19],  8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[ 4], 27);
		s[ 4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21],  2);
		s[21] = ROTL64(s[ 8], 55);
		s[ 8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[ 5], 36);
		s[ 5] = ROTL64(s[ 3], 28);
		s[ 3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[ 7],  6);
		s[ 7] = ROTL64(s[10],  3);
		s[10] = ROTL64(    v,  1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= c_keccak_round_constants[i];
	}
}

__device__
void keccak_hash_init(keccak_hash_state *S)
{
	#pragma unroll 25
	for (int i=0; i<25; ++i)
		S->state[i] = 0ULL;
}

// assuming there is no leftover data and exactly 72 bytes are incoming
// we can directly call into the block hashing function
__device__ void keccak_hash_update72(keccak_hash_state *S, const uint32_t *in) {
	keccak_block(S, in);
}

__device__ void keccak_hash_update8(keccak_hash_state *S, const uint32_t *in) {
	mycpy8(S->buffer, in);
}

__device__ void keccak_hash_update4_8(keccak_hash_state *S, const uint32_t *in) {
	mycpy4(S->buffer+8/4, in);
}

__device__ void keccak_hash_update4_56(keccak_hash_state *S, const uint32_t *in) {
	mycpy4(S->buffer+56/4, in);
}

__device__ void keccak_hash_update56(keccak_hash_state *S, const uint32_t *in) {
	mycpy56(S->buffer, in);
}

__device__ void keccak_hash_update64(keccak_hash_state *S, const uint32_t *in) {
	mycpy64(S->buffer, in);
}

__device__
void keccak_hash_finish8(keccak_hash_state *S, uint32_t *hash)
{
	S->buffer[8/4] = 0x01;
	#pragma unroll 15
	for (int i=8/4+1; i < 72/4; ++i) S->buffer[i] = 0;
	S->buffer[72/4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

	#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i/4]), S->state[i / 8]);
	}
}

__device__
void keccak_hash_finish12(keccak_hash_state *S, uint32_t *hash)
{
	S->buffer[12/4] = 0x01;
	#pragma unroll 14
	for (int i=12/4+1; i < 72/4; ++i) S->buffer[i] = 0;
	S->buffer[72/4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

	#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i/4]), S->state[i / 8]);
	}
}

__device__
void keccak_hash_finish60(keccak_hash_state *S, uint32_t *hash)
{
	S->buffer[60/4] = 0x01;
	#pragma unroll
	for (int i=60/4+1; i < 72/4; ++i) S->buffer[i] = 0;
	S->buffer[72/4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

	#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i/4]), S->state[i / 8]);
	}
}

__device__
void keccak_hash_finish64(keccak_hash_state *S, uint32_t *hash)
{
	S->buffer[64/4] = 0x01;
	#pragma unroll
	for (int i=64/4+1; i < 72/4; ++i) S->buffer[i] = 0;
	S->buffer[72/4 - 1] |= 0x80000000U;
	keccak_block(S, (const uint32_t*)S->buffer);

	#pragma unroll 8
	for (int i = 0; i < 64; i += 8) {
		U64TO32_LE((&hash[i/4]), S->state[i / 8]);
	}
}

// ---------------------------- END keccak functions ------------------------------------

// ---------------------------- BEGIN PBKDF2 functions ------------------------------------

typedef struct pbkdf2_hmac_state_t {
	keccak_hash_state inner, outer;
} pbkdf2_hmac_state;


__device__ void pbkdf2_hash(uint32_t *hash, const uint32_t *m)
{
	keccak_hash_state st;
	keccak_hash_init(&st);
	keccak_hash_update72(&st, m);
	keccak_hash_update8(&st, m+72/4);
	keccak_hash_finish8(&st, hash);
}

/* hmac */
__device__
void pbkdf2_hmac_init80(pbkdf2_hmac_state *st, const uint32_t *key)
{
	uint32_t pad[72/4] = { 0 };
	//#pragma unroll 18
	//for (int i = 0; i < 72/4; i++)
	//	pad[i] = 0;

	keccak_hash_init(&st->inner);
	keccak_hash_init(&st->outer);

	/* key > blocksize bytes, hash it */
	pbkdf2_hash(pad, key);

	/* inner = (key ^ 0x36) */
	/* h(inner || ...) */
	#pragma unroll 18
	for (int i = 0; i < 72/4; i++)
		pad[i] ^= 0x36363636U;
	keccak_hash_update72(&st->inner, pad);

	/* outer = (key ^ 0x5c) */
	/* h(outer || ...) */
	#pragma unroll 18
	for (int i = 0; i < 72/4; i++)
		pad[i] ^= 0x6a6a6a6aU;
	keccak_hash_update72(&st->outer, pad);
}

// assuming there is no leftover data and exactly 72 bytes are incoming
// we can directly call into the block hashing function
__device__ void pbkdf2_hmac_update72(pbkdf2_hmac_state *st, const uint32_t *m) {
	/* h(inner || m...) */
	keccak_hash_update72(&st->inner, m);
}

__device__ void pbkdf2_hmac_update8(pbkdf2_hmac_state *st, const uint32_t *m) {
	/* h(inner || m...) */
	keccak_hash_update8(&st->inner, m);
}

__device__ void pbkdf2_hmac_update4_8(pbkdf2_hmac_state *st, const uint32_t *m) {
	/* h(inner || m...) */
	keccak_hash_update4_8(&st->inner, m);
}

__device__ void pbkdf2_hmac_update4_56(pbkdf2_hmac_state *st, const uint32_t *m) {
	/* h(inner || m...) */
	keccak_hash_update4_56(&st->inner, m);
}

__device__ void pbkdf2_hmac_update56(pbkdf2_hmac_state *st, const uint32_t *m) {
	/* h(inner || m...) */
	keccak_hash_update56(&st->inner, m);
}

__device__ void pbkdf2_hmac_finish12(pbkdf2_hmac_state *st, uint32_t *mac) {
	/* h(inner || m) */
	uint32_t innerhash[16];
	keccak_hash_finish12(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	keccak_hash_update64(&st->outer, innerhash);
	keccak_hash_finish64(&st->outer, mac);
}

__device__ void pbkdf2_hmac_finish60(pbkdf2_hmac_state *st, uint32_t *mac) {
	/* h(inner || m) */
	uint32_t innerhash[16];
	keccak_hash_finish60(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	keccak_hash_update64(&st->outer, innerhash);
	keccak_hash_finish64(&st->outer, mac);
}

__device__ void pbkdf2_statecopy8(pbkdf2_hmac_state *d, pbkdf2_hmac_state *s) {
	statecopy8(&d->inner, &s->inner);
	statecopy0(&d->outer, &s->outer);
}

// ---------------------------- END PBKDF2 functions ------------------------------------

__global__ __launch_bounds__(128)
void cuda_pre_keccak512(uint32_t *g_idata, uint32_t nonce)
{
	uint32_t data[20];

	const uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	nonce   += thread;
	g_idata += thread * 32;

	#pragma unroll
	for (int i=0; i<19; i++)
		data[i] = cuda_swab32(c_data[i]);
	data[19] = cuda_swab32(nonce);

//    scrypt_pbkdf2_1((const uint8_t*)data, 80, (const uint8_t*)data, 80, (uint8_t*)g_idata, 128);

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init80(&hmac_pw, data);

	/* hmac(password, salt...) */
	pbkdf2_hmac_update72(&hmac_pw, data);
	pbkdf2_hmac_update8(&hmac_pw, data+72/4);

	pbkdf2_hmac_state work;
	uint32_t ti[16];

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	pbkdf2_statecopy8(&work, &hmac_pw);
	pbkdf2_hmac_update4_8(&work, &be);
	pbkdf2_hmac_finish12(&work, ti);
	mycpy64(g_idata, ti);

	be = 0x02000000U;//cuda_swab32(2);
	pbkdf2_statecopy8(&work, &hmac_pw);
	pbkdf2_hmac_update4_8(&work, &be);
	pbkdf2_hmac_finish12(&work, ti);
	mycpy64(g_idata+16, ti);
}


__global__ __launch_bounds__(128)
void cuda_post_keccak512(uint32_t *g_odata, uint32_t *g_hash, uint32_t nonce)
{
	uint32_t data[20];

	const uint32_t thread = (blockIdx.x * blockDim.x) + threadIdx.x;
	g_hash  += thread * 8;
	g_odata += thread * 32;
	nonce   += thread;

	#pragma unroll
	for (int i=0; i<19; i++)
		data[i] = cuda_swab32(c_data[i]);
	data[19] = cuda_swab32(nonce);

//	scrypt_pbkdf2_1((const uint8_t*)data, 80, (const uint8_t*)g_odata, 128, (uint8_t*)g_hash, 32);

	pbkdf2_hmac_state hmac_pw;

	/* hmac(password, ...) */
	pbkdf2_hmac_init80(&hmac_pw, data);

	/* hmac(password, salt...) */
	pbkdf2_hmac_update72(&hmac_pw, g_odata);
	pbkdf2_hmac_update56(&hmac_pw, g_odata+72/4);

	uint32_t ti[16];

	/* U1 = hmac(password, salt || be(i)) */
	uint32_t be = 0x01000000U;//cuda_swab32(1);
	pbkdf2_hmac_update4_56(&hmac_pw, &be);
	pbkdf2_hmac_finish60(&hmac_pw, ti);
	mycpy32(g_hash, ti);
}

//
// callable host code to initialize constants and to call kernels
//

extern "C" void prepare_keccak512(int thr_id, const uint32_t host_pdata[20])
{
	static bool init[MAX_GPUS] = { 0 };

	if (!init[thr_id])
	{
		checkCudaErrors(cudaMemcpyToSymbol(c_keccak_round_constants, host_keccak_round_constants, sizeof(host_keccak_round_constants), 0, cudaMemcpyHostToDevice));
		init[thr_id] = true;
	}
	checkCudaErrors(cudaMemcpyToSymbol(c_data, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void pre_keccak512(int thr_id, int stream, uint32_t nonce, int throughput)
{
	dim3 block(128);
	dim3 grid((throughput+127)/128);

	cuda_pre_keccak512<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_idata[stream][thr_id], nonce);
}

extern "C" void post_keccak512(int thr_id, int stream, uint32_t nonce, int throughput)
{
	dim3 block(128);
	dim3 grid((throughput+127)/128);

	cuda_post_keccak512<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_odata[stream][thr_id], context_hash[stream][thr_id], nonce);
}
