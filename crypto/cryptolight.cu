
#include "cryptolight.h"

extern char *device_config[MAX_GPUS]; // -l 32x16

static __thread uint32_t cn_blocks  = 32;
static __thread uint32_t cn_threads = 16;

static uint32_t *d_long_state[MAX_GPUS];
static uint64_t *d_ctx_state[MAX_GPUS];
static uint32_t *d_ctx_key1[MAX_GPUS];
static uint32_t *d_ctx_key2[MAX_GPUS];
static uint32_t *d_ctx_text[MAX_GPUS];
static uint32_t *d_ctx_a[MAX_GPUS];
static uint32_t *d_ctx_b[MAX_GPUS];

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_cryptolight(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int res = 0;
	uint32_t throughput = 0;

	uint32_t *ptarget = work->target;
	uint8_t *pdata = (uint8_t*) work->data;
	uint32_t *nonceptr = (uint32_t*) (&pdata[39]);
	const uint32_t first_nonce = *nonceptr;
	uint32_t nonce = first_nonce;

	if(opt_benchmark) {
		ptarget[7] = 0x00ff;
	}

	if(!init[thr_id])
	{
		if (device_config[thr_id]) {
			sscanf(device_config[thr_id], "%ux%u", &cn_blocks, &cn_threads);
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			gpulog(LOG_INFO, thr_id, "Using %u x %u kernel launch config, %u threads",
				cn_blocks, cn_threads, throughput);
		} else {
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			if (throughput != cn_blocks*cn_threads && cn_threads) {
				cn_blocks = throughput / cn_threads;
				throughput = cn_threads * cn_blocks;
			}
			gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u threads (%ux%u)",
				throughput2intensity(throughput), throughput, cn_blocks, cn_threads);
		}

		if(sizeof(size_t) == 4 && throughput > UINT32_MAX / MEMORY) {
			gpulog(LOG_ERR, thr_id, "THE 32bit VERSION CAN'T ALLOCATE MORE THAN 4GB OF MEMORY!");
			gpulog(LOG_ERR, thr_id, "PLEASE REDUCE THE NUMBER OF THREADS OR BLOCKS");
			exit(1);
		}

		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		const size_t alloc = MEMORY * throughput;
		cryptonight_extra_cpu_init(thr_id, throughput);

		cudaMalloc(&d_long_state[thr_id], alloc);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_state[thr_id], 26 * sizeof(uint64_t) * throughput);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_key1[thr_id], 40 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_key2[thr_id], 40 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_text[thr_id], 32 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_a[thr_id], 4 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		cudaMalloc(&d_ctx_b[thr_id], 4 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);

		init[thr_id] = true;
	}

	throughput = cn_blocks*cn_threads;

	do
	{
		const uint32_t Htarg = ptarget[7];
		uint32_t resNonces[2] = { UINT32_MAX, UINT32_MAX };

		cryptonight_extra_cpu_setData(thr_id, pdata, ptarget);
		cryptonight_extra_cpu_prepare(thr_id, throughput, nonce, d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id], d_ctx_key1[thr_id], d_ctx_key2[thr_id]);
		cryptolight_core_cpu_hash(thr_id, cn_blocks, cn_threads, d_long_state[thr_id], d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id], d_ctx_key1[thr_id], d_ctx_key2[thr_id]);
		cryptonight_extra_cpu_final(thr_id, throughput, nonce, resNonces, d_ctx_state[thr_id]);

		*hashes_done = nonce - first_nonce + throughput;

		if(resNonces[0] != UINT32_MAX)
		{
			uint32_t vhash[8];
			uint32_t tempdata[19];
			uint32_t *tempnonceptr = (uint32_t*)(((char*)tempdata) + 39);
			memcpy(tempdata, pdata, 76);
			*tempnonceptr = resNonces[0];
			cryptolight_hash(vhash, tempdata, 76);
			if(vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				res = 1;
				work->nonces[0] = resNonces[0];
				work_set_target_ratio(work, vhash);
				// second nonce
				if(resNonces[1] != UINT32_MAX)
				{
					*tempnonceptr = resNonces[1];
					cryptolight_hash(vhash, tempdata, 76);
					if(vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
						res++;
						work->nonces[1] = resNonces[1];
					} else if (vhash[7] > Htarg) {
						gpu_increment_reject(thr_id);
					}
				}
				goto done;
			} else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for nonce %08x does not validate on CPU!", resNonces[0]);
			}
		}

		if ((uint64_t) throughput + nonce >= max_nonce - 127) {
			nonce = max_nonce;
			break;
		}

		nonce += throughput;
		gpulog(LOG_DEBUG, thr_id, "nonce %08x", nonce);

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + nonce);

done:
	gpulog(LOG_DEBUG, thr_id, "nonce %08x exit", nonce);
	work->valid_nonces = res;
	*nonceptr = nonce;
	return res;
}

void free_cryptolight(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaFree(d_long_state[thr_id]);
	cudaFree(d_ctx_state[thr_id]);
	cudaFree(d_ctx_key1[thr_id]);
	cudaFree(d_ctx_key2[thr_id]);
	cudaFree(d_ctx_text[thr_id]);
	cudaFree(d_ctx_a[thr_id]);
	cudaFree(d_ctx_b[thr_id]);

	cryptonight_extra_cpu_free(thr_id);

	cudaDeviceSynchronize();

	init[thr_id] = false;
}
