#ifndef _CUDA_GROESTL512_H
#define _CUDA_GROESTL512_H

void groestl512_cpu_init(int thr_id, int threads);
void groestl512_cpu_copyHeftyHash(int thr_id, int threads, void *heftyHashes, int copy);
void groestl512_cpu_setBlock(void *data, int len);
void groestl512_cpu_hash(int thr_id, int threads, uint32_t startNounce);

#endif