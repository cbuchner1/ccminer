/*
 * Genoil's CUDA mining kernel for Ethereum
 * based on Tim Hughes' opencl kernel.
 * thanks to tpruvot,djm34,sp,cbuchner for things i took from ccminer.
 */

#define SHUFFLE_MIN_VER 9350


#include "miner.h"
#include "cuda_helper.h"
#include "ethash_cu_miner_kernel.h"

#include <stdio.h>
#include <unistd.h>
#include <memory.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "vector_types.h"

#include "rotl64.cuh"
#include "keccak.cuh"

//#include "ethash_cu_miner_kernel_globals.h"
__constant__ uint32_t d_dag_size;
__constant__ uint32_t d_max_outputs;

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16) /* 8 */

#define FNV_PRIME	0x01000193

#define SWAP64(v) \
  ((ROTL64L(v,  8) & 0x000000FF000000FF) | \
   (ROTL64L(v, 24) & 0x0000FF000000FF00) | \
   (ROTL64H(v, 40) & 0x00FF000000FF0000) | \
   (ROTL64H(v, 56) & 0xFF000000FF000000))

#undef SWAP64
#define SWAP64(v) cuda_swab64(v)

#define PACK64(result, lo, hi) asm("mov.b64 %0, {%1,%2};//pack64"  : "=l"(result) : "r"(lo), "r"(hi));
#define UNPACK64(lo, hi, input) asm("mov.b64 {%0, %1}, %2;//unpack64" : "=r"(lo),"=r"(hi) : "l"(input));

#define copy(dst, src, count) for (uint32_t i = 0; i < count; i++) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

#define fnv(x,y) ((x) * FNV_PRIME ^(y))

__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

__device__ uint32_t fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}


__device__ void init_hash(hash32_t const* header, uint64_t nonce, hash64_t &init)
{
	// sha3_512(header .. nonce)
	uint64_t state[25] = { 0 };

	//copy(state, header->uint64s, 4);
	#pragma unroll
	for (int i = 0; i < 4; i++) state[i] = header->uint64s[i];

	state[4] = nonce;
	state[5] = 0x0000000000000001;
	//state[6] = 0;
	//state[7] = 0;
	state[8] = 0x8000000000000000;
	//for (uint32_t i = 9; i < 25; i++) {
	//	state[i] = 0;
	//}

	keccak_f1600_block(state, 8);

	//copy(init.uint64s, state, 8);
	#pragma unroll
	for (int i = 0; i < 8; i++) init.uint64s[i] = state[i];
}

__device__ uint32_t inner_loop(uint4 mix, uint32_t thread_id, uint32_t* share, hash128_t const* g_dag)
{
	// share init0
	if (thread_id == 0)
		*share = mix.x;

	uint32_t init0 = *share;

	uint32_t a = 0;

	do
	{

		bool update_share = thread_id == ((a >> 2) & (THREADS_PER_HASH-1));

		//#pragma unroll 4
		for (uint32_t i = 0; i < 4; i++)
		{

			if (update_share)
			{
				uint32_t m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a + i), m[i]) % d_dag_size;
			}
			__threadfence_block();

#if __CUDA_ARCH__ >= 350
			mix = fnv4(mix, __ldg(&g_dag[*share].uint4s[thread_id]));
#else
			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
#endif

		}

	} while ((a += 4) != ACCESSES);

	return fnv_reduce(mix);
}

__device__ void final_hash(hash64_t const* init, hash32_t const* mix, hash32_t &hash)
{
	uint64_t state[25] = { 0 };

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->uint64s, 8);
	copy(state + 8, mix->uint64s, 4);
	state[12] = 0x0000000000000001;
	//#pragma unroll
	//for (uint32_t i = 13; i < 16; i++)
	//{
		//state[i] = 0;
	//}
	state[16] = 0x8000000000000000;
	//#pragma unroll
	//for (uint32_t i = 17; i < 25; i++)
	//{
		//state[i] = 0;
	//}

	keccak_f1600_block(state,1);

	// copy out
	copy(hash.uint64s, state, 4);
}

typedef union
{
	hash64_t init;
	hash32_t mix;
} compute_hash_share;

#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
__device__ uint64_t compute_hash_shuffle(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce
	)
{
	// sha3_512(header .. nonce)
	uint64_t state[25];

	copy(state, g_header->uint64s, 4);
	state[4] = nonce;
	state[5] = 0x0000000000000001ULL;
	for (uint32_t i = 6; i < 25; i++)
	{
		state[i] = 0;
	}
	state[8] = 0x8000000000000000ULL;
	keccak_f1600_block(state, 8);

	// Threads work together in this phase in groups of 8.
	const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
	const int start_lane = threadIdx.x & ~(THREADS_PER_HASH - 1);
	const int mix_idx = (thread_id & 3);

	uint4 mix;

	uint32_t shuffle[16];
	//uint32_t * init = (uint32_t *)state;

	uint32_t init[16];
	UNPACK64(init[0], init[1], state[0]);
	UNPACK64(init[2], init[3], state[1]);
	UNPACK64(init[4], init[5], state[2]);
	UNPACK64(init[6], init[7], state[3]);
	UNPACK64(init[8], init[9], state[4]);
	UNPACK64(init[10], init[11], state[5]);
	UNPACK64(init[12], init[13], state[6]);
	UNPACK64(init[14], init[15], state[7]);

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{

		// share init among threads
		for (int j = 0; j < 16; j++)
			shuffle[j] = __shfl(init[j], start_lane + i);

		// ugly but avoids local reads/writes
		if (mix_idx == 0) {
			mix = make_uint4(shuffle[0], shuffle[1], shuffle[2], shuffle[3]);
		}
		else if (mix_idx == 1) {
			mix = make_uint4(shuffle[4], shuffle[5], shuffle[6], shuffle[7]);
		}
		else if (mix_idx == 2) {
			mix = make_uint4(shuffle[8], shuffle[9], shuffle[10], shuffle[11]);
		}
		else {
			mix = make_uint4(shuffle[12], shuffle[13], shuffle[14], shuffle[15]);
		}

		uint32_t init0 = __shfl(shuffle[0], start_lane);


		for (uint32_t a = 0; a < ACCESSES; a+=4)
		{
			int t = ((a >> 2) & (THREADS_PER_HASH - 1));

			for (uint32_t b = 0; b < 4; b++)
			{
				if (thread_id == t)
				{
					shuffle[0] = fnv(init0 ^ (a + b), ((uint32_t *)&mix)[b]) % d_dag_size;;
				}
				shuffle[0] = __shfl(shuffle[0], start_lane + t);

				mix = fnv4(mix, g_dag[shuffle[0]].uint4s[thread_id]);
			}
		}

		uint32_t thread_mix = fnv_reduce(mix);

		// update mix accross threads

		for (int j = 0; j < 8; j++)
			shuffle[j] = __shfl(thread_mix, start_lane + j);

		if (i == thread_id) {

			//move mix into state:
			PACK64(state[8],  shuffle[0], shuffle[1]);
			PACK64(state[9],  shuffle[2], shuffle[3]);
			PACK64(state[10], shuffle[4], shuffle[5]);
			PACK64(state[11], shuffle[6], shuffle[7]);
		}

	}

	// keccak_256(keccak_512(header..nonce) .. mix);
	state[12] = 0x0000000000000001ULL;
	for (uint32_t i = 13; i < 25; i++)
	{
		state[i] = 0ULL;
	}
	state[16] = 0x8000000000000000;
	keccak_f1600_block(state, 1);

	return state[0];
}
#endif

__device__ void compute_hash(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t nonce,
	hash32_t &out
	)
{
	extern __shared__  compute_hash_share share[];

	// Compute one init hash per work item.
	hash64_t init;
	init_hash(g_header, nonce, init);

	// Threads work together in this phase in groups of 8.
	uint32_t const thread_id = threadIdx.x & (THREADS_PER_HASH-1);
	uint32_t const hash_id   = threadIdx.x >> 3;

	hash32_t mix;

	for (int i = 0; i < THREADS_PER_HASH; i++)
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;

		uint4 thread_init = share[hash_id].init.uint4s[thread_id & 3];

		uint32_t thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uint32s, g_dag);

		share[hash_id].mix.uint32s[thread_id] = thread_mix;


		if (i == thread_id)
			mix = share[hash_id].mix;
	}

	final_hash(&init, &mix, out);
}

__global__ void
__launch_bounds__(128, 7)
ethash_search(
	uint32_t* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
	)
{

	uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;

#if 0 // _CUDA_ARCH__ >= SHUFFLE_MIN_VER
	uint64_t hash = compute_hash_shuffle(g_header, g_dag, start_nonce + gid);
	if (SWAP64(hash) < target)
	{
		atomicInc(g_output, d_max_outputs);
		g_output[g_output[0]] = gid;
	}
#else
	hash32_t hash;
	compute_hash(g_header, g_dag, start_nonce + gid, hash);
	if (SWAP64(hash.uint64s[0]) <= target)
	{
		atomicInc(g_output,d_max_outputs);
		g_output[g_output[0]] = gid;
	}
#endif


}

void run_ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce
)
{
}

void run_ethash_search(
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	uint32_t* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	uint64_t start_nonce,
	uint64_t target
)
{
#if __CUDA_ARCH__ >= SHUFFLE_MIN_VER
	ethash_search <<<blocks, threads, 0, stream>>>(g_output, g_header, g_dag, start_nonce, target);
#else
	ethash_search <<<blocks, threads, (sizeof(compute_hash_share) * threads) / THREADS_PER_HASH, stream>>>(g_output, g_header, g_dag, start_nonce, target);
#endif
}

cudaError set_constants(
	uint32_t * dag_size,
	uint32_t * max_outputs
	)
{
	cudaError result;
	result = cudaMemcpyToSymbol(d_dag_size, dag_size, sizeof(uint32_t));
	result = cudaMemcpyToSymbol(d_max_outputs, max_outputs, sizeof(uint32_t));
	return result;
}

static unsigned m_search_batch_size = 262144;
static unsigned m_workgroup_size = 64;

static __thread hash128_t * m_dag_ptr;
static __thread hash32_t * m_header;

//static void* m_hash_buf[MAX_GPUS] = { 0 };
static uint32_t* m_search_buf[MAX_GPUS] = { 0 };
static cudaStream_t m_streams[MAX_GPUS] = { 0 };

static uint64_t dagSize = 0;
static int fdDAG = -1;
static uint8_t const *dag = NULL;

uint32_t const c_zero = 0;

#define ETHASH_MIX_BYTES 128
#define MIX_WORDS (ETHASH_MIX_BYTES/4)


uint8_t* alloc_dag_file(uint32_t rev, uint32_t* seedhash)
{
	uint8_t* buffer = NULL;
	char fn[256];
	struct stat stat_buf;
	int rc;

	sprintf(fn, "%s/.ethash/full-R%u-%08x%08x", getenv("HOME"), rev, seedhash[7], seedhash[6]);

	rc = stat(fn, &stat_buf);
	fdDAG = open(fn, O_RDONLY);
	if (rc != 0) {
		applog(LOG_ERR, "Unable to open DAG file %s (%d)!", fn, rc);
		return NULL;
	}

    dagSize = stat_buf.st_size - 8;

	#ifdef __linux__
		buffer = (uint8_t*) (mmap(NULL, dagSize, PROT_READ, MAP_FILE | MAP_SHARED | MAP_POPULATE , fdDAG, 0));
	#else
		buffer = (uint8_t*) (mmap(NULL, dagSize, PROT_READ, MAP_FILE | MAP_SHARED), fdDAG, 0));
	#endif
	if (buffer == MAP_FAILED) {
		applog(LOG_ERR, "Unable to map DAG file %s!", fn);
		return NULL;
	}

	if (buffer[0] != 0xfe) {
		applog(LOG_ERR, "DAG file header is bad %s!", fn);
		close(fdDAG);
		return NULL;
	}

	return (buffer+8);
}

void close_dag_file(uint8_t const *dag)
{
	uint8_t* buffer = dag ? (uint8_t*) dag - 8 : NULL;
	munmap(buffer, dagSize);
	dag = NULL;
	close(fdDAG);
	fdDAG = -1;
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_ether(int thr_id, struct work *work, uint64_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pseed = &work->data[8];

	const uint64_t first_nonce = (((uint64_t)pdata[20]) << 32) + pdata[19];
	uint64_t nonce = first_nonce;

	uint32_t throughput =  device_intensity(thr_id, __func__, 1U << 18);
	throughput = min(throughput, max_nonce - first_nonce);

	if (!dagSize) {
		dag = alloc_dag_file(23, pseed);
		applog(LOG_NOTICE, "DAG size=%ld", (long int) dagSize);
	}

	//0000.0000 0050.4c39c 047378ad262a995be73cc504643443728e1ca415bae9208
	uint64_t target64 = ptarget[7];
	target64 = (target64 << 32) + ptarget[6];

	if (!init[thr_id])
	{
		uint32_t dag_size = (unsigned)(dagSize / ETHASH_MIX_BYTES);
		uint32_t max_outputs=63;

		cudaSetDevice(device_map[thr_id]);

		// create buffer for header256
		cudaMalloc(&m_header, 32);

		// create mining buffer
		cudaMalloc(&m_search_buf[thr_id], 8 * sizeof(uint32_t));
		//cudaMalloc(&m_hash_buf[thr_id], c_hash_batch_size * 32);

		//cudaStreamCreate(&m_streams[thr_id]);
		m_streams[thr_id] = 0;

		// create buffer for dag
		CUDA_SAFE_CALL(cudaMalloc(&m_dag_ptr, dagSize));
		CUDA_SAFE_CALL(cudaMemcpy(m_dag_ptr, dag, dagSize, cudaMemcpyHostToDevice));

		CUDA_SAFE_CALL(set_constants(&dag_size, &max_outputs));

		init[thr_id] = true;
	}
	else if (dag) {
		close_dag_file(dag);
	}

	CUDA_SAFE_CALL(cudaMemcpy(m_header, work->data, 32, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(m_search_buf[thr_id], &c_zero, 4, cudaMemcpyHostToDevice));

	uint32_t results[8] = { 0 };
	do {
		results[0] = 0;
		CUDA_SAFE_CALL(cudaMemset(m_search_buf[thr_id], 0, sizeof(uint32_t)));

		run_ethash_search(m_search_batch_size / m_workgroup_size, m_workgroup_size, m_streams[thr_id], m_search_buf[thr_id], m_header, m_dag_ptr, nonce, target64);

		CUDA_SAFE_CALL(cudaThreadSynchronize());
		CUDA_SAFE_CALL(cudaMemcpyAsync(results, m_search_buf[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost, m_streams[thr_id]));

		unsigned num_found = results[0];
		if (num_found) {
			nonce = nonce + results[1];
			pdata[19] = (uint32_t) nonce;
			pdata[20] = (uint32_t) (nonce >> 32);
			return 1;
		}

		nonce += throughput;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = nonce - first_nonce + 1;
	return 0;
}
