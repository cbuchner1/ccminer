#ifndef _CUDA_KECCAK512_H
#define _CUDA_KECCAK512_H

void keccak512_cpu_init(int thr_id, int threads);
void keccak512_cpu_setBlock(void *data, int len);
void keccak512_cpu_copyHeftyHash(int thr_id, int threads, void *heftyHashes, int copy);
void keccak512_cpu_hash(int thr_id, int threads, uint32_t startNounce);

#endif
