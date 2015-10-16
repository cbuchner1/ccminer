#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "sph/sph_fugue.h"

#include "miner.h"

#include "cuda_fugue256.h"

extern "C" void my_fugue256_init(void *cc);
extern "C" void my_fugue256(void *cc, const void *data, size_t len);
extern "C" void my_fugue256_close(void *cc, void *dst);
extern "C" void my_fugue256_addbits_and_close(void *cc, unsigned ub, unsigned n, void *dst);

// vorbereitete Kontexte nach den ersten 80 Bytes
// sph_fugue256_context  ctx_fugue_const[MAX_GPUS];

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

void fugue256_hash(unsigned char* output, const unsigned char* input, int len)
{
	sph_fugue256_context ctx;

	sph_fugue256_init(&ctx);
	sph_fugue256(&ctx, input, len);
	sph_fugue256_close(&ctx, (void *)output);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_fugue256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t start_nonce = pdata[19]++;
	int intensity = (device_sm[device_map[thr_id]] > 500) ? 22 : 19;
	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << intensity); // 256*256*8
	if (init[thr_id]) throughput = min(throughput, max_nonce - start_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0xf;

	// init
	if(!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);

		fugue256_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	// Endian
	for (int kk=0; kk < 20; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
	fugue256_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		// GPU
		uint32_t foundNounce = UINT32_MAX;
		fugue256_cpu_hash(thr_id, throughput, pdata[19], NULL, &foundNounce);

		if (foundNounce < UINT32_MAX)
		{
			uint32_t vhash[8];
			sph_fugue256_context ctx_fugue;
			endiandata[19] = SWAP32(foundNounce);

			sph_fugue256_init(&ctx_fugue);
			sph_fugue256 (&ctx_fugue, endiandata, 80);
			sph_fugue256_close(&ctx_fugue, &vhash);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget))
			{
				work_set_target_ratio(work, vhash);
				pdata[19] = foundNounce;
				*hashes_done = foundNounce - start_nonce + 1;
				return 1;
			} else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNounce);
			}
		}

		if ((uint64_t) pdata[19] + throughput > (uint64_t) max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - start_nonce + 1;
	return 0;
}

// cleanup
void free_fugue256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	fugue256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
