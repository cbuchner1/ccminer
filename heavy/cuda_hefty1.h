#ifndef _CUDA_HEFTY1_H
#define _CUDA_HEFTY1_H

void hefty_cpu_hash(int thr_id, int threads, int startNounce);
void hefty_cpu_setBlock(int thr_id, int threads, void *data, int len);
void hefty_cpu_init(int thr_id, int threads);

#endif