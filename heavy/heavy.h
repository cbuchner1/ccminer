#ifndef _CUDA_HEAVY_H
#define _CUDA_HEAVY_H

void blake512_cpu_init(int thr_id, uint32_t threads);
void blake512_cpu_setBlock(void *pdata, int len);
void blake512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce);
void blake512_cpu_free(int thr_id);

void groestl512_cpu_init(int thr_id, uint32_t threads);
void groestl512_cpu_copyHeftyHash(int thr_id, uint32_t threads, void *heftyHashes, int copy);
void groestl512_cpu_setBlock(void *data, int len);
void groestl512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce);
void groestl512_cpu_free(int thr_id);

void hefty_cpu_hash(int thr_id, uint32_t threads, int startNounce);
void hefty_cpu_setBlock(int thr_id, uint32_t threads, void *data, int len);
void hefty_cpu_init(int thr_id, uint32_t threads);
void hefty_cpu_free(int thr_id);

void keccak512_cpu_init(int thr_id, uint32_t threads);
void keccak512_cpu_setBlock(void *data, int len);
void keccak512_cpu_copyHeftyHash(int thr_id, uint32_t threads, void *heftyHashes, int copy);
void keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce);
void keccak512_cpu_free(int thr_id);

void sha256_cpu_init(int thr_id, uint32_t threads);
void sha256_cpu_setBlock(void *data, int len);
void sha256_cpu_hash(int thr_id, uint32_t threads, int startNounce);
void sha256_cpu_copyHeftyHash(int thr_id, uint32_t threads, void *heftyHashes, int copy);
void sha256_cpu_free(int thr_id);

void combine_cpu_init(int thr_id, uint32_t threads);
void combine_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *hash);
void combine_cpu_free(int thr_id);

#endif
