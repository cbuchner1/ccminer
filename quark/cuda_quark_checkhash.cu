#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

// Folgende Definitionen später durch header ersetzen
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// das Hash Target gegen das wir testen sollen
__constant__ uint32_t pTarget[8];

uint32_t *d_resNounce[8];
uint32_t *h_resNounce[8];

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__global__ void quark_check_gpu_hash_64(int threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *inpHash = &g_hash[16 * hashPosition];

		uint32_t hash[8];
#pragma unroll 8
		for (int i=0; i < 8; i++)
			hash[i] = inpHash[i];

		// kopiere Ergebnis
		int i, position = -1;
		bool rc = true;

#pragma unroll 8
		for (i = 7; i >= 0; i--) {
			if (hash[i] > pTarget[i]) {
				if(position < i) {
					position = i;
					rc = false;
				}
	 		}
	 		if (hash[i] < pTarget[i]) {
				if(position < i) {
					position = i;
					rc = true;
				}
	 		}
		}

		if(rc == true)
			if(resNounce[0] > nounce)
				resNounce[0] = nounce;
	}
}

// Setup-Funktionen
__host__ void quark_check_cpu_init(int thr_id, int threads)
{
    cudaMallocHost(&h_resNounce[thr_id], 1*sizeof(uint32_t));
    cudaMalloc(&d_resNounce[thr_id], 1*sizeof(uint32_t));
}

// Target Difficulty setzen
__host__ void quark_check_cpu_setTarget(const void *ptarget)
{
	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( pTarget, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__ uint32_t quark_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order)
{
	uint32_t result = 0xffffffff;
	cudaMemset(d_resNounce[thr_id], 0xff, sizeof(uint32_t));

	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	quark_check_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, d_nonceVector, d_inputHash, d_resNounce[thr_id]);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);

	// Ergebnis zum Host kopieren (in page locked memory, damits schneller geht)
	cudaMemcpy(h_resNounce[thr_id], d_resNounce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// cudaMemcpy() ist asynchron!
	cudaThreadSynchronize();
	result = *h_resNounce[thr_id];

	return result;
}
