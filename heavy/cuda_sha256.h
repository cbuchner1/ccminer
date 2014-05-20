#ifndef _CUDA_SHA256_H
#define _CUDA_SHA256_H

void sha256_cpu_init(int thr_id, int threads);
void sha256_cpu_setBlock(void *data, int len);
void sha256_cpu_hash(int thr_id, int threads, int startNounce);
void sha256_cpu_copyHeftyHash(int thr_id, int threads, void *heftyHashes, int copy);
#endif
