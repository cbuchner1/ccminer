#ifndef _CUDA_FUGUE256_H
#define _CUDA_FUGUE256_H

void fugue256_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce);
void fugue256_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void fugue256_cpu_init(int thr_id, uint32_t threads);
void fugue256_cpu_free(int thr_id);

#endif
