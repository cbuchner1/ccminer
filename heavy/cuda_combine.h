#ifndef _CUDA_COMBINE_H
#define _CUDA_COMBINE_H

void combine_cpu_init(int thr_id, int threads);
void combine_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint32_t *hash);

#endif
