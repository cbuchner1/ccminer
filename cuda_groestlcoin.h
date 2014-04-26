#ifndef _CUDA_GROESTLCOIN_H
#define _CUDA_GROESTLCOIN_H

void groestlcoin_cpu_init(int thr_id, int threads);
void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void groestlcoin_cpu_hash(int thr_id, int threads, uint32_t startNounce, void *outputHashes, uint32_t *nounce);

#endif