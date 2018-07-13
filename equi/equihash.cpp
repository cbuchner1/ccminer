/**
 * Equihash solver interface for ccminer (compatible with linux and windows)
 * Solver taken from nheqminer, by djeZo (and NiceHash)
 * tpruvot - 2017 (GPL v3)
 */
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <stdexcept>
#include <vector>

#include <sph/sph_sha2.h>

#include "eqcuda.hpp"
#include "equihash.h" // equi_verify()

#include <miner.h>

// All solutions (BLOCK_HEADER_LEN + SOLSIZE_LEN + SOL_LEN) sha256d should be under the target
extern "C" void equi_hash(const void* input, void* output, int len)
{
	uint8_t _ALIGN(64) hash0[32], hash1[32];

	sph_sha256_context ctx_sha256;

	sph_sha256_init(&ctx_sha256);
	sph_sha256(&ctx_sha256, input, len);
	sph_sha256_close(&ctx_sha256, hash0);
	sph_sha256(&ctx_sha256, hash0, 32);
	sph_sha256_close(&ctx_sha256, hash1);

	memcpy(output, hash1, 32);
}

// input here is 140 for the header and 1344 for the solution (equi.cpp)
extern "C" int equi_verify_sol(void * const hdr, void * const sol)
{
	bool res = equi_verify((uint8_t*) hdr, (uint8_t*) sol);

	//applog_hex((void*)hdr, 140);
	//applog_hex((void*)sol, 1344);

	return res ? 1 : 0;
}

#include <cuda_helper.h>

//#define EQNONCE_OFFSET 30 /* 27:34 */
#define NONCE_OFT EQNONCE_OFFSET

static bool init[MAX_GPUS] = { 0 };
static int valid_sols[MAX_GPUS] = { 0 };
static uint8_t _ALIGN(64) data_sols[MAX_GPUS][MAXREALSOLS][1536] = { 0 }; // 140+3+1344 required
static eq_cuda_context_interface* solvers[MAX_GPUS] = { NULL };

static void CompressArray(const unsigned char* in, size_t in_len,
	unsigned char* out, size_t out_len, size_t bit_len, size_t byte_pad)
{
	assert(bit_len >= 8);
	assert(8 * sizeof(uint32_t) >= 7 + bit_len);

	size_t in_width = (bit_len + 7) / 8 + byte_pad;
	assert(out_len == bit_len*in_len / (8 * in_width));

	uint32_t bit_len_mask = (1UL << bit_len) - 1;

	// The acc_bits least-significant bits of acc_value represent a bit sequence
	// in big-endian order.
	size_t acc_bits = 0;
	uint32_t acc_value = 0;

	size_t j = 0;
	for (size_t i = 0; i < out_len; i++) {
		// When we have fewer than 8 bits left in the accumulator, read the next
		// input element.
		if (acc_bits < 8) {
			acc_value = acc_value << bit_len;
			for (size_t x = byte_pad; x < in_width; x++) {
				acc_value = acc_value | (
					(
					// Apply bit_len_mask across byte boundaries
					in[j + x] & ((bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF)
					) << (8 * (in_width - x - 1))); // Big-endian
			}
			j += in_width;
			acc_bits += bit_len;
		}

		acc_bits -= 8;
		out[i] = (acc_value >> acc_bits) & 0xFF;
	}
}

#ifndef htobe32
#define htobe32(x) swab32(x)
#endif

static void EhIndexToArray(const u32 i, unsigned char* arr)
{
	u32 bei = htobe32(i);
	memcpy(arr, &bei, sizeof(u32));
}

static std::vector<unsigned char> GetMinimalFromIndices(std::vector<u32> indices, size_t cBitLen)
{
	assert(((cBitLen + 1) + 7) / 8 <= sizeof(u32));
	size_t lenIndices = indices.size()*sizeof(u32);
	size_t minLen = (cBitLen + 1)*lenIndices / (8 * sizeof(u32));
	size_t bytePad = sizeof(u32) - ((cBitLen + 1) + 7) / 8;
	std::vector<unsigned char> array(lenIndices);
	for (size_t i = 0; i < indices.size(); i++) {
		EhIndexToArray(indices[i], array.data() + (i*sizeof(u32)));
	}
	std::vector<unsigned char> ret(minLen);
	CompressArray(array.data(), lenIndices, ret.data(), minLen, cBitLen + 1, bytePad);
	return ret;
}

// solver callbacks
static void cb_solution(int thr_id, const std::vector<uint32_t>& solutions, size_t cbitlen, const unsigned char *compressed_sol)
{
	std::vector<unsigned char> nSolution;
	if (!compressed_sol) {
		nSolution = GetMinimalFromIndices(solutions, cbitlen);
	} else {
		gpulog(LOG_INFO, thr_id, "compressed_sol");
		nSolution = std::vector<unsigned char>(1344);
		for (size_t i = 0; i < cbitlen; i++)
			nSolution[i] = compressed_sol[i];
	}
	int nsol = valid_sols[thr_id];
	if (nsol < 0) nsol = 0;
	if(nSolution.size() == 1344) {
		// todo, only store solution data here...
		le32enc(&data_sols[thr_id][nsol][140], 0x000540fd); // sol sz header
		memcpy(&data_sols[thr_id][nsol][143], nSolution.data(), 1344);
		valid_sols[thr_id] = nsol + 1;
	}
}
static void cb_hashdone(int thr_id) {
	if (!valid_sols[thr_id]) valid_sols[thr_id] = -1;
}
static bool cb_cancel(int thr_id) {
	if (work_restart[thr_id].restart)
		valid_sols[thr_id] = -1;
	return work_restart[thr_id].restart;
}

extern "C" int scanhash_equihash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[35];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[NONCE_OFT];
	uint32_t nonce_increment = rand() & 0xFF; // nonce randomizer
	struct timeval tv_start, tv_end, diff;
	double secs, solps;
	uint32_t soluce_count = 0;

	if (opt_benchmark)
		ptarget[7] = 0xfffff;

	if (!init[thr_id]) {
		try {
			int mode = 1;
			switch (mode) {
			case 1:
				solvers[thr_id] = new eq_cuda_context<CONFIG_MODE_1>(thr_id, device_map[thr_id]);
				break;
#ifdef CONFIG_MODE_2
			case 2:
				solvers[thr_id] = new eq_cuda_context<CONFIG_MODE_2>(thr_id, device_map[thr_id]);
				break;
#endif
#ifdef CONFIG_MODE_3
			case 3:
				solvers[thr_id] = new eq_cuda_context<CONFIG_MODE_3>(thr_id, device_map[thr_id]);
				break;
#endif
			default:
				proper_exit(EXIT_CODE_SW_INIT_ERROR);
				return -1;
			}
			size_t memSz = solvers[thr_id]->equi_mem_sz / (1024*1024);
			gpus_intensity[thr_id] = (uint32_t) solvers[thr_id]->throughput;
			api_set_throughput(thr_id, gpus_intensity[thr_id]);
			gpulog(LOG_DEBUG, thr_id, "Allocated %u MB of context memory", (u32) memSz);
			cuda_get_arch(thr_id);
			init[thr_id] = true;
		} catch (const std::exception & e) {
			CUDA_LOG_ERROR();
			gpulog(LOG_ERR, thr_id, "init: %s", e.what());
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
	}

	gettimeofday(&tv_start, NULL);
	memcpy(endiandata, pdata, 140);
	work->valid_nonces = 0;

	do {

		try {

			valid_sols[thr_id] = 0;
			solvers[thr_id]->solve(
				(const char *) endiandata, (unsigned int) (140 - 32),
				(const char *) &endiandata[27], (unsigned int) 32,
				&cb_cancel, &cb_solution, &cb_hashdone
			);

			*hashes_done = soluce_count;

		} catch (const std::exception & e) {
			gpulog(LOG_WARNING, thr_id, "solver: %s", e.what());
			free_equihash(thr_id);
			sleep(1);
			return -1;
		}

		if (valid_sols[thr_id] > 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			uint8_t _ALIGN(64) full_data[140+3+1344] = { 0 };
			uint8_t* sol_data = &full_data[140];

			soluce_count += valid_sols[thr_id];

			for (int nsol=0; nsol < valid_sols[thr_id]; nsol++)
			{
				memcpy(full_data, endiandata, 140);
				memcpy(sol_data, &data_sols[thr_id][nsol][140], 1347);
				equi_hash(full_data, vhash, 140+3+1344);

				if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
				{
					bool valid = equi_verify_sol(endiandata, &sol_data[3]);
					if (valid && work->valid_nonces < MAX_NONCES) {
						work->valid_nonces++;
						memcpy(work->data, endiandata, 140);
						equi_store_work_solution(work, vhash, sol_data);
						work->nonces[work->valid_nonces-1] = endiandata[NONCE_OFT];
						pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
						//applog_hex(vhash, 32);
						//applog_hex(&work->data[27], 32);
						goto out; // second solution storage not handled..
					}
				}
				if (work->valid_nonces == MAX_NONCES) goto out;
			}
			if (work->valid_nonces)
				goto out;

			valid_sols[thr_id] = 0;
		}

		endiandata[NONCE_OFT] += nonce_increment;

	} while (!work_restart[thr_id].restart);

out:
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&diff, &tv_end, &tv_start);
	secs = (1.0 * diff.tv_sec) + (0.000001 * diff.tv_usec);
	solps = (double)soluce_count / secs;
	gpulog(LOG_DEBUG, thr_id, "%d solutions in %.2f s (%.2f Sol/s)", soluce_count, secs, solps);

	// H/s
	*hashes_done = soluce_count;

	pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;

	return work->valid_nonces;
}

// cleanup
void free_equihash(int thr_id)
{
	if (!init[thr_id])
		return;

	// assume config 1 was used... interface destructor seems bad
	eq_cuda_context<CONFIG_MODE_1>* ptr = dynamic_cast<eq_cuda_context<CONFIG_MODE_1>*>(solvers[thr_id]);
	ptr->freemem();
	ptr = NULL;

	solvers[thr_id] = NULL;

	init[thr_id] = false;
}

// mmm... viva c++ junk
void eq_cuda_context_interface::solve(const char *tequihash_header, unsigned int tequihash_header_len,
	const char* nonce, unsigned int nonce_len,
	fn_cancel cancelf, fn_solution solutionf, fn_hashdone hashdonef) { }
