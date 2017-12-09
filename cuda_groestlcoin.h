#ifndef _CUDA_GROESTLCOIN_H
#define _CUDA_GROESTLCOIN_H

void groestlcoin_cpu_init(int thr_id, uint32_t threads);
void groestlcoin_cpu_free(int thr_id);
void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void groestlcoin_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonce);

#endif