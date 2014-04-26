#include "uint256.h"
#include "sph/sph_groestl.h"

#include "cpuminer-config.h"
#include "miner.h"

#include <string.h>
#include <stdint.h>
#include "cuda_groestlcoin.h"
#include <openssl/sha.h>

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

void sha256func(unsigned char *hash, const unsigned char *data, int len)
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
	/*
	memcpy(S + 8, sha256d_hash1 + 8, 32);
	sha256_init(T);
	sha256_transform(T, S, 0);
	*/
	for (i = 0; i < 8; i++)
		be32enc((uint32_t *)hash + i, T[i]);
}

static void groestlhash(void *state, const void *input)
{
	// Tryout GPU-groestl

    sph_groestl512_context     ctx_groestl[2];
    static unsigned char pblank[1];
	int ii;
    uint32_t mask = 8;
    uint32_t zero = 0;


	//these uint512 in the c++ source of the client are backed by an array of uint32
    uint32_t hashA[16], hashB[16];	


    sph_groestl512_init(&ctx_groestl[0]);
    sph_groestl512 (&ctx_groestl[0], input, 80); //6
    sph_groestl512_close(&ctx_groestl[0], hashA); //7	

	sph_groestl512_init(&ctx_groestl[1]);
	sph_groestl512 (&ctx_groestl[1], hashA, 64); //6
    sph_groestl512_close(&ctx_groestl[1], hashB); //7

	memcpy(state, hashB, 32);
}



extern "C" int scanhash_groestlcoin(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{	
	uint32_t start_nonce = pdata[19]++;
	const uint32_t Htarg = ptarget[7];
	const uint32_t throughPut = 4096 * 128;
	//const uint32_t throughPut = 1;
	int i;
	uint32_t *outputHash = (uint32_t*)malloc(throughPut * 16 * sizeof(uint32_t));

	// init
	static bool init[8] = { false, false, false, false, false, false, false, false };
	if(!init[thr_id])
	{
		groestlcoin_cpu_init(thr_id, throughPut);
		init[thr_id] = true;
	}
	
	// Endian Drehung ist notwendig
	//char testdata[] = {"\x70\x00\x00\x00\x5d\x38\x5b\xa1\x14\xd0\x79\x97\x0b\x29\xa9\x41\x8f\xd0\x54\x9e\x7d\x68\xa9\x5c\x7f\x16\x86\x21\xa3\x14\x20\x10\x00\x00\x00\x00\x57\x85\x86\xd1\x49\xfd\x07\xb2\x2f\x3a\x8a\x34\x7c\x51\x6d\xe7\x05\x2f\x03\x4d\x2b\x76\xff\x68\xe0\xd6\xec\xff\x9b\x77\xa4\x54\x89\xe3\xfd\x51\x17\x32\x01\x1d\xf0\x73\x10\x00"};
	//pdata = (uint32_t*)testdata;
	uint32_t endiandata[32];
	for (int kk=0; kk < 32; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
	groestlcoin_cpu_setBlock(thr_id, endiandata, (void*)ptarget);
	
	do {
		// GPU
		uint32_t foundNounce = 0xFFFFFFFF;

		groestlcoin_cpu_hash(thr_id, throughPut, pdata[19], outputHash, &foundNounce);

		/*
		{
			for(i=0;i<throughPut;i++)
			{
				uint32_t tmpHash[8];
				endiandata[19] = SWAP32(pdata[19]);
				groestlhash(tmpHash, endiandata);
				
				int ii;
				printf("result GPU: ");
				for (ii=0; ii < 32; ii++)
				{
					printf ("%.2x",((uint8_t*)&outputHash[8*i])[ii]);
				};
				printf ("\n");	
		

				groestlhash(tmpHash, endiandata);
				printf("result CPU: ");
				for (ii=0; ii < 32; ii++)
				{
					printf ("%.2x",((uint8_t*)tmpHash)[ii]);
				};
				
				
			}
			exit(0);
		}		
		*/
		if(foundNounce < 0xffffffff)
		{
			uint32_t tmpHash[8];
			endiandata[19] = SWAP32(foundNounce);
			groestlhash(tmpHash, endiandata);
			if (tmpHash[7] <= Htarg && 
					fulltest(tmpHash, ptarget)) {
						pdata[19] = foundNounce;
						*hashes_done = foundNounce - start_nonce;
						free(outputHash);
				return true;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNounce);
			}

			foundNounce = 0xffffffff;
			/*
			int ii;
			printf("result GPU: ");
			for (ii=0; ii < 32; ii++)
			{
				printf ("%.2x",((uint8_t*)&outputHash[0])[ii]);
			};
			printf ("\n");	
			printf("result CPU: ");
			for (ii=0; ii < 32; ii++)
			{
				printf ("%.2x",((uint8_t*)tmpHash)[ii]);
			};
			printf ("\n");	
			*/
		}

		if (pdata[19] + throughPut < pdata[19])
			pdata[19] = max_nonce;
		else pdata[19] += throughPut;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = pdata[19] - start_nonce;
	free(outputHash);
	return 0;
}

