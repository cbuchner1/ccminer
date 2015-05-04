/* Based on djm code */

#include <stdint.h>

#include "miner.h"
#include "cuda_helper.h"

#include <openssl/sha.h>

static uint32_t *d_hash[MAX_GPUS] ;

extern void pluck_setBlockTarget(const void* data, const void *ptarget);
extern void pluck_cpu_init(int thr_id, uint32_t threads, uint32_t *d_outputHash);
extern uint32_t pluck_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, int order);

extern float tp_coef[MAX_GPUS];

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
//note, this is 64 bytes
static inline void xor_salsa8(uint32_t B[16], const uint32_t Bx[16])
{
#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
	uint32_t x00, x01, x02, x03, x04, x05, x06, x07, x08, x09, x10, x11, x12, x13, x14, x15;
	int i;

	x00 = (B[0] ^= Bx[0]);
	x01 = (B[1] ^= Bx[1]);
	x02 = (B[2] ^= Bx[2]);
	x03 = (B[3] ^= Bx[3]);
	x04 = (B[4] ^= Bx[4]);
	x05 = (B[5] ^= Bx[5]);
	x06 = (B[6] ^= Bx[6]);
	x07 = (B[7] ^= Bx[7]);
	x08 = (B[8] ^= Bx[8]);
	x09 = (B[9] ^= Bx[9]);
	x10 = (B[10] ^= Bx[10]);
	x11 = (B[11] ^= Bx[11]);
	x12 = (B[12] ^= Bx[12]);
	x13 = (B[13] ^= Bx[13]);
	x14 = (B[14] ^= Bx[14]);
	x15 = (B[15] ^= Bx[15]);
	for (i = 0; i < 8; i += 2) {
		/* Operate on columns. */
		x04 ^= ROTL(x00 + x12, 7);  x09 ^= ROTL(x05 + x01, 7);
		x14 ^= ROTL(x10 + x06, 7);  x03 ^= ROTL(x15 + x11, 7);

		x08 ^= ROTL(x04 + x00, 9);  x13 ^= ROTL(x09 + x05, 9);
		x02 ^= ROTL(x14 + x10, 9);  x07 ^= ROTL(x03 + x15, 9);

		x12 ^= ROTL(x08 + x04, 13);  x01 ^= ROTL(x13 + x09, 13);
		x06 ^= ROTL(x02 + x14, 13);  x11 ^= ROTL(x07 + x03, 13);

		x00 ^= ROTL(x12 + x08, 18);  x05 ^= ROTL(x01 + x13, 18);
		x10 ^= ROTL(x06 + x02, 18);  x15 ^= ROTL(x11 + x07, 18);

		/* Operate on rows. */
		x01 ^= ROTL(x00 + x03, 7);  x06 ^= ROTL(x05 + x04, 7);
		x11 ^= ROTL(x10 + x09, 7);  x12 ^= ROTL(x15 + x14, 7);

		x02 ^= ROTL(x01 + x00, 9);  x07 ^= ROTL(x06 + x05, 9);
		x08 ^= ROTL(x11 + x10, 9);  x13 ^= ROTL(x12 + x15, 9);

		x03 ^= ROTL(x02 + x01, 13);  x04 ^= ROTL(x07 + x06, 13);
		x09 ^= ROTL(x08 + x11, 13);  x14 ^= ROTL(x13 + x12, 13);

		x00 ^= ROTL(x03 + x02, 18);  x05 ^= ROTL(x04 + x07, 18);
		x10 ^= ROTL(x09 + x08, 18);  x15 ^= ROTL(x14 + x13, 18);
	}
	B[0] += x00;
	B[1] += x01;
	B[2] += x02;
	B[3] += x03;
	B[4] += x04;
	B[5] += x05;
	B[6] += x06;
	B[7] += x07;
	B[8] += x08;
	B[9] += x09;
	B[10] += x10;
	B[11] += x11;
	B[12] += x12;
	B[13] += x13;
	B[14] += x14;
	B[15] += x15;
#undef ROTL
}

static void sha256_hash(uchar *hash, const uchar *data, int len)
{
	SHA256_CTX ctx;
	SHA256_Init(&ctx);
	SHA256_Update(&ctx, data, len);
	SHA256_Final(hash, &ctx);
}

// hash exactly 64 bytes (ie, sha256 block size)
static void sha256_hash512(uint32_t *hash, const uint32_t *data)
{
	uint32_t _ALIGN(64) S[16];
	uint32_t _ALIGN(64) T[16];
	uchar _ALIGN(64) E[64] = { 0 };
	int i;

	sha256_init(S);

	for (i = 0; i < 16; i++)
		T[i] = be32dec(&data[i]);
	sha256_transform(S, T, 0);

	E[3] = 0x80;
	E[61] = 0x02; // T[15] = 8 * 64 => 0x200;
	sha256_transform(S, (uint32_t*)E, 0);

	for (i = 0; i < 8; i++)
		be32enc(&hash[i], S[i]);
}

#define BLOCK_HEADER_SIZE 80
void pluckhash(uint32_t *hash, const uint32_t *data, uchar *hashbuffer, const int N)
{
	int size = N * 1024;
	sha256_hash(hashbuffer, (uchar*)data, BLOCK_HEADER_SIZE);
	memset(&hashbuffer[32], 0, 32);

	for (int i = 64; i < size - 32; i += 32)
	{
		uint32_t _ALIGN(64) randseed[16];
		uint32_t _ALIGN(64) randbuffer[16];
		uint32_t _ALIGN(64) joint[16];
		//i-4 because we use integers for all references against this, and we don't want to go 3 bytes over the defined area
		//we could use size here, but then it's probable to use 0 as the value in most cases
		int randmax = i - 4;

		//setup randbuffer to be an array of random indexes
		memcpy(randseed, &hashbuffer[i - 64], 64);

		if (i > 128) memcpy(randbuffer, &hashbuffer[i - 128], 64);
		else memset(randbuffer, 0, 64);

		xor_salsa8((uint32_t*)randbuffer, (uint32_t*)randseed);
		memcpy(joint, &hashbuffer[i - 32], 32);

		//use the last hash value as the seed
		for (int j = 32; j < 64; j += 4)
		{
			//every other time, change to next random index
			//randmax - 32 as otherwise we go beyond memory that's already been written to
			uint32_t rand = randbuffer[(j - 32) >> 2] % (randmax - 32);
			joint[j >> 2] = *((uint32_t *)&hashbuffer[rand]);
		}

		sha256_hash512((uint32_t*)&hashbuffer[i], joint);

		//setup randbuffer to be an array of random indexes
		//use last hash value and previous hash value(post-mixing)
		memcpy(randseed, &hashbuffer[i - 32], 64);

		if (i > 128) memcpy(randbuffer, &hashbuffer[i - 128], 64);
		else memset(randbuffer, 0, 64);

		xor_salsa8((uint32_t*)randbuffer, (uint32_t*)randseed);

		//use the last hash value as the seed
		for (int j = 0; j < 32; j += 2)
		{
			uint32_t rand = randbuffer[j >> 1] % randmax;
			*((uint32_t *)(hashbuffer + rand)) = *((uint32_t *)(hashbuffer + j + randmax));
		}
	}

	memcpy(hash, hashbuffer, 32);
}

static bool init[MAX_GPUS] = { 0 };

static __thread uchar* scratchbuf = NULL;

extern "C" int scanhash_pluck(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	int opt_pluck_n = 128;

	int intensity = is_windows() ? 17 : 19; /* beware > 20 could work and create diff problems later */
	uint32_t throughput = device_intensity(thr_id, __func__, 1U << intensity);
	// divide by 128 for this algo which require a lot of memory
	throughput = throughput / 128 - 256;
	throughput = min(throughput, max_nonce - first_nonce + 1);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaMalloc(&d_hash[thr_id], opt_pluck_n * 1024 * throughput);

		if (!scratchbuf)
			scratchbuf = (uchar*) calloc(opt_pluck_n, 1024);

		pluck_cpu_init(thr_id, throughput, d_hash[thr_id]);

		CUDA_SAFE_CALL(cudaGetLastError());
		applog(LOG_INFO, "Using %d cuda threads", throughput);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	pluck_setBlockTarget(endiandata,ptarget);

	do {
		uint32_t foundNonce = pluck_cpu_hash(thr_id, throughput, pdata[19], 0);
		if (foundNonce != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			pluckhash(vhash64, endiandata, scratchbuf, opt_pluck_n);
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = foundNonce;
				return 1;
			} else {
				applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], foundNonce);
			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
