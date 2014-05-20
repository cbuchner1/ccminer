#ifndef _CUDA_BLAKE512_H
#define _CUDA_BLAKE512_H

void blake512_cpu_init(int thr_id, int threads);
void blake512_cpu_setBlock(void *pdata, int len);
void blake512_cpu_hash(int thr_id, int threads, uint32_t startNounce);
#endif
