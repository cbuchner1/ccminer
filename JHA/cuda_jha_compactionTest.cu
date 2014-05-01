#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_30_intrinsics.h"

#include <stdio.h>
#include <memory.h>
#include <stdint.h>

// aus cpu-miner.c
extern int device_map[8];

// diese Struktur wird in der Init Funktion angefordert
static cudaDeviceProp props[8];

static uint32_t *d_tempBranch1Nonces[8];
static uint32_t *d_tempBranch2Nonces[8];
static size_t *d_numValid[8];
static size_t *h_numValid[8];

static uint32_t *d_partSum1[8], *d_partSum2[8]; // 2x partielle summen
static uint32_t *d_validTemp1[8], *d_validTemp2[8];

// Zwischenspeicher
static uint32_t *d_tempBranchAllNonces[8];

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);



// Setup-Funktionen
__host__ void jackpot_compactTest_cpu_init(int thr_id, int threads)
{
	cudaGetDeviceProperties(&props[thr_id], device_map[thr_id]);

	// wir brauchen auch Speicherplatz auf dem Device
	cudaMalloc(&d_tempBranchAllNonces[thr_id], sizeof(uint32_t) * threads);
	cudaMalloc(&d_tempBranch1Nonces[thr_id], sizeof(uint32_t) * threads);
	cudaMalloc(&d_tempBranch2Nonces[thr_id], sizeof(uint32_t) * threads);
	cudaMalloc(&d_numValid[thr_id], 2*sizeof(size_t));
	cudaMallocHost(&h_numValid[thr_id], 2*sizeof(size_t));

	uint32_t s1;
	s1 = threads / 256;

	cudaMalloc(&d_partSum1[thr_id], sizeof(uint32_t) * s1); // BLOCKSIZE (Threads/Block)
	cudaMalloc(&d_partSum2[thr_id], sizeof(uint32_t) * s1); // BLOCKSIZE (Threads/Block)

	cudaMalloc(&d_validTemp1[thr_id], sizeof(uint32_t) * threads); // BLOCKSIZE (Threads/Block)
	cudaMalloc(&d_validTemp2[thr_id], sizeof(uint32_t) * threads); // BLOCKSIZE (Threads/Block)
}

// Die Testfunktion (zum Erstellen der TestMap)
__global__ void jackpot_compactTest_gpu_TEST_64(int threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_noncesFull,
												uint32_t *d_nonces1, uint32_t *d_nonces2,
												uint32_t *d_validT1, uint32_t *d_validT2)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		uint32_t nounce = startNounce + thread;
		uint32_t *inpHash = &inpHashes[16 * thread];

		uint32_t tmp = inpHash[0] & 0x01;
		uint32_t val1 = (tmp == 1);
		uint32_t val2 = (tmp == 0);

		d_nonces1[thread] = val1;
		d_validT1[thread] = val1;
		d_nonces2[thread] = val2;
		d_validT2[thread] = val2;
		d_noncesFull[thread] = nounce;
	}
}

// Die Summenfunktion (vom NVIDIA SDK)
__global__ void jackpot_compactTest_gpu_SCAN(uint32_t *data, int width, uint32_t *partial_sums=NULL)
{
	extern __shared__ uint32_t sums[];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	//int lane_id = id % warpSize;
	int lane_id = id % width;
	// determine a warp_id within a block
	 //int warp_id = threadIdx.x / warpSize;
	int warp_id = threadIdx.x / width;

	// Below is the basic structure of using a shfl instruction
	// for a scan.
	// Record "value" as a variable - we accumulate it along the way
	uint32_t value = data[id];

	// Now accumulate in log steps up the chain
	// compute sums, with another thread's value who is
	// distance delta away (i).  Note
	// those threads where the thread 'i' away would have
	// been out of bounds of the warp are unaffected.  This
	// creates the scan sum.
#pragma unroll

	for (int i=1; i<=width; i*=2)
	{
		uint32_t n = __shfl_up((int)value, i, width);

		if (lane_id >= i) value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp

	// write the sum of the warp to smem
	//if (threadIdx.x % warpSize == warpSize-1)
	if (threadIdx.x % width == width-1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	if (warp_id == 0)
	{
		uint32_t warp_sum = sums[lane_id];

		for (int i=1; i<=width; i*=2)
		{
			uint32_t n = __shfl_up((int)warp_sum, i, width);

		if (lane_id >= i) warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighbouring warp's sum and add it to threads value
	uint32_t blockSum = 0;

	if (warp_id > 0)
	{
		blockSum = sums[warp_id-1];
	}

	value += blockSum;

	// Now write out our result
	data[id] = value;

	// last thread has sum, write write out the block's sum
	if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
	{
		partial_sums[blockIdx.x] = value;
	}
}

// Uniform add: add partial sums array
__global__ void jackpot_compactTest_gpu_ADD(uint32_t *data, uint32_t *partial_sums, int len)
{
	__shared__ uint32_t buf;
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (id > len) return;

	if (threadIdx.x == 0)
	{
		buf = partial_sums[blockIdx.x];
	}

	__syncthreads();
	data[id] += buf;
}

// Der Scatter
__global__ void jackpot_compactTest_gpu_SCATTER(uint32_t *data, uint32_t *valid, uint32_t *sum, uint32_t *outp)
{
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	if( valid[id] )
	{
		int idx = sum[id];
		if(idx > 0)
			outp[idx-1] = data[id];
	}
}

////// ACHTUNG: Diese funktion geht aktuell nur mit threads > 65536 (Am besten 256 * 1024 oder 256*2048)
__host__ void jackpot_compactTest_cpu_dualCompaction(int thr_id, int threads, size_t *nrm,
													 uint32_t *d_nonces1, uint32_t *d_nonces2)
{
	// threadsPerBlock ausrechnen
	int blockSize = 256;
	int thr1 = threads / blockSize;
	int thr2 = threads / (blockSize*blockSize);

	// 1
	jackpot_compactTest_gpu_SCAN<<<thr1,blockSize, 8*sizeof(uint32_t)>>>(d_tempBranch1Nonces[thr_id], 32, d_partSum1[thr_id]);
	jackpot_compactTest_gpu_SCAN<<<thr2,blockSize, 8*sizeof(uint32_t)>>>(d_partSum1[thr_id], 32, d_partSum2[thr_id]);
	jackpot_compactTest_gpu_SCAN<<<1, thr2, 8*sizeof(uint32_t)>>>(d_partSum2[thr_id], (thr2>32) ? 32 : thr2);
	cudaStreamSynchronize(NULL);
	cudaMemcpy(&nrm[0], &(d_partSum2[thr_id])[thr2-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	jackpot_compactTest_gpu_ADD<<<thr2-1, blockSize>>>(d_partSum1[thr_id]+blockSize, d_partSum2[thr_id], blockSize*thr2);
	jackpot_compactTest_gpu_ADD<<<thr1-1, blockSize>>>(d_tempBranch1Nonces[thr_id]+blockSize, d_partSum1[thr_id], threads);

	// 2
	jackpot_compactTest_gpu_SCAN<<<thr1,blockSize, 8*sizeof(uint32_t)>>>(d_tempBranch2Nonces[thr_id], 32, d_partSum1[thr_id]);
	jackpot_compactTest_gpu_SCAN<<<thr2,blockSize, 8*sizeof(uint32_t)>>>(d_partSum1[thr_id], 32, d_partSum2[thr_id]);
	jackpot_compactTest_gpu_SCAN<<<1, thr2, 8*sizeof(uint32_t)>>>(d_partSum2[thr_id], (thr2>32) ? 32 : thr2);
	cudaStreamSynchronize(NULL);
	cudaMemcpy(&nrm[1], &(d_partSum2[thr_id])[thr2-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);	
	jackpot_compactTest_gpu_ADD<<<thr2-1, blockSize>>>(d_partSum1[thr_id]+blockSize, d_partSum2[thr_id], blockSize*thr2);
	jackpot_compactTest_gpu_ADD<<<thr1-1, blockSize>>>(d_tempBranch2Nonces[thr_id]+blockSize, d_partSum1[thr_id], threads);
	
	// Hier ist noch eine Besonderheit: in d_tempBranch1Nonces sind die element von 1...nrm1 die Interessanten
	// Schritt 3: Scatter
	jackpot_compactTest_gpu_SCATTER<<<thr1,blockSize,0>>>(d_tempBranchAllNonces[thr_id], d_validTemp1[thr_id], d_tempBranch1Nonces[thr_id], d_nonces1);
	jackpot_compactTest_gpu_SCATTER<<<thr1,blockSize,0>>>(d_tempBranchAllNonces[thr_id], d_validTemp2[thr_id], d_tempBranch2Nonces[thr_id], d_nonces2);
	cudaStreamSynchronize(NULL);
}

__host__ void jackpot_compactTest_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *inpHashes, 
											uint32_t *d_nonces1, size_t *nrm1,
											uint32_t *d_nonces2, size_t *nrm2,
											int order)
{
	// Compute 3.x und 5.x Geräte am besten mit 768 Threads ansteuern,
	// alle anderen mit 512 Threads.
	//int threadsperblock = (props[thr_id].major >= 3) ? 768 : 512;
	int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

//	fprintf(stderr, "threads=%d, %d blocks, %d threads per block, %d bytes shared\n", threads, grid.x, block.x, shared_size);

	// Schritt 1: Prüfen der Bedingung und Speicherung in d_tempBranch1/2Nonces
	jackpot_compactTest_gpu_TEST_64<<<grid, block, shared_size>>>(threads, startNounce, inpHashes, d_tempBranchAllNonces[thr_id],
		d_tempBranch1Nonces[thr_id], d_tempBranch2Nonces[thr_id],
		d_validTemp1[thr_id], d_validTemp2[thr_id]);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	jackpot_compactTest_cpu_dualCompaction(thr_id, threads,
		h_numValid[thr_id], d_nonces1, d_nonces2);

	cudaStreamSynchronize(NULL); // Das original braucht zwar etwas CPU-Last, ist an dieser Stelle aber evtl besser
	*nrm1 = h_numValid[thr_id][0];
	*nrm2 = h_numValid[thr_id][1];
}
