
extern "C"
{
//#include "sph/neoscrypt.h"
#include "miner.h"
}

#include <stdint.h>

// aus cpu-miner.c
extern int device_map[8];

// Speicher für Input/Output der verketteten Hashfunktionen

static uint32_t *d_hash[8] ;
 

extern void pluck_setBlockTarget(const void* data, const void *ptarget);
extern void pluck_cpu_init(int thr_id, int threads, uint32_t *d_outputHash);
extern uint32_t pluck_cpu_hash(int thr_id, int threads, uint32_t startNounce, int order);
  

extern float tp_coef[8];
extern bool opt_benchmark;

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

void sha256_hash(unsigned char *hash, const unsigned char *data, int len)
{
	uint32_t S[16], T[16];
	int i, r;

	sha256_init(S);
	for (r = len; r > -9; r -= 64) {
		if (r < 64)
			memset(T, 0, 64);
		memcpy(T, data + len - r, r > 64 ? 64 : (r < 0 ? 0 : r));
		if (r >= 0 && r < 64)
			((unsigned char *)T)[r] = 0x80;
		for (i = 0; i < 16; i++) 
			T[i] = be32dec(T + i);

		if (r < 56)
			T[15] = 8 * len;
		sha256_transform(S, T, 0);
	}
	for (i = 0; i < 8; i++)
		be32enc((uint32_t *)hash + i, S[i]);
}

void sha256_hash512(unsigned char *hash, const unsigned char *data)
{
	uint32_t S[16], T[16];
	int i;

	sha256_init(S);

	memcpy(T, data, 64);
	for (i = 0; i < 16; i++)
		T[i] = be32dec(T + i);
	sha256_transform(S, T, 0);

	memset(T, 0, 64);
	//memcpy(T, data + 64, 0);
	((unsigned char *)T)[0] = 0x80;
	for (i = 0; i < 16; i++)
		T[i] = be32dec(T + i);
	T[15] = 8 * 64;
	sha256_transform(S, T, 0);

	for (i = 0; i < 8; i++)
		be32enc((uint32_t *)hash + i, S[i]);
}

inline void pluck(uint32_t *hash, uint32_t *input)
{

	uint32_t data[20];
	
	//uint32_t midstate[8];
//	printf("coming here\n");
	const int HASH_MEMORY = 128 * 1024;
	uint8_t * scratchbuf = (uint8_t*)malloc(HASH_MEMORY);
	
	
	for (int k = 0; k<20; k++) { data[k] = input[k]; }
	
		
		uint8_t *hashbuffer = scratchbuf; //don't allocate this on stack, since it's huge.. 
		int size = HASH_MEMORY;
//        int size = 224+64;
		memset(hashbuffer, 0, 64);
		
//		for (int k = 0; k<10; k++) {
//			printf("cpu init data %d %08x %08x\n", k, ((uint32_t*)(data))[2 * k], ((uint32_t*)(data))[2 * k + 1]);}
		sha256_hash(&hashbuffer[0], (uint8_t*)data, 80);
//		for (int k = 0; k<8; k++) { printf("cpu hash %d %08x \n", k, ((uint32_t*)hashbuffer)[k]); }

		for (int i = 64; i < size - 32; i += 32)
		{
			//i-4 because we use integers for all references against this, and we don't want to go 3 bytes over the defined area
			int randmax = i - 4; //we could use size here, but then it's probable to use 0 as the value in most cases
			uint32_t joint[16];
			uint32_t randbuffer[16];

			uint32_t randseed[16];
			memcpy(randseed, &hashbuffer[i - 64], 64);
			if (i>128)
			{
				memcpy(randbuffer, &hashbuffer[i - 128], 64);
			}
			else
			{
				memset(&randbuffer, 0, 64);
			}

			xor_salsa8(randbuffer, randseed);
			
			memcpy(joint, &hashbuffer[i - 32], 32);
			//use the last hash value as the seed
			for (int j = 32; j < 64; j += 4)
			{
				uint32_t rand = randbuffer[(j - 32) / 4] % (randmax - 32); //randmax - 32 as otherwise we go beyond memory that's already been written to
				joint[j / 4] = *((uint32_t*)&hashbuffer[rand]);
			}
			sha256_hash512(&hashbuffer[i], (uint8_t*)joint);
//			for (int k = 0; k<8; k++) { printf("sha hashbuffer %d %08x\n", k, ((uint32_t*)(hashbuffer+i))[k]); }
			memcpy(randseed, &hashbuffer[i - 32], 64); //use last hash value and previous hash value(post-mixing)
			if (i>128)
			{
				memcpy(randbuffer, &hashbuffer[i - 128], 64);
			}
			else
			{
				memset(randbuffer, 0, 64);
			}
			xor_salsa8(randbuffer, randseed);
			for (int j = 0; j < 32; j += 2)
			{
				uint32_t rand = randbuffer[j / 2] % randmax;
				*((uint32_t*)&hashbuffer[rand]) = *((uint32_t*)&hashbuffer[j + i - 4]);
			}
		}

//		for (int k = 0; k<8; k++) { printf("cpu final hash %d %08x\n", k, ((uint32_t*)hashbuffer)[k]); }

		//note: off-by-one error is likely here...     
/*
		for (int i = size - 64 - 1; i >= 64; i -= 64)
		{
			sha256_hash512(&hashbuffer[i - 64], &hashbuffer[i]);
		}

		for (int k = 0; k<8; k++) { printf("cpu after of by one final hash %d %08x\n", k, ((uint32_t*)hashbuffer)[k]); }
*/
		memcpy((unsigned char*)hash, hashbuffer, 32);
}

extern "C" int scanhash_pluck(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];
	if (tp_coef[thr_id]<0) { tp_coef[thr_id]=2.45; }
	const int throughput = (uint32_t)((float)(32*1*64*tp_coef[thr_id]));
	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]); 
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// Konstanten kopieren, Speicher belegen
		cudaMalloc(&d_hash[thr_id], 32 * 1024 * sizeof(uint32_t) * throughput);


		pluck_cpu_init(thr_id, throughput,d_hash[thr_id]);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];

		for (int k = 0; k < 20; k++) 
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);
 
	pluck_setBlockTarget(endiandata,ptarget);

	do {
		int order = 0;
		uint32_t foundNonce = pluck_cpu_hash(thr_id, throughput, pdata[19], order++);
//		foundNonce = pdata[19];
		if  (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];

//             be32enc(&endiandata[19], foundNonce);
//             pluck(vhash64,endiandata);
//			 printf("target %08x vhash64 %08x", ptarget[7], vhash64[7]);
//			if ( vhash64[7] <= ptarget[7]) { // && fulltest(vhash64, ptarget)) {
				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
//			} else {
//				*hashes_done = foundNonce - first_nonce + 1; // keeps hashrate calculation happy
//				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
//			}
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
