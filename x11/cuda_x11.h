#include "quark/cuda_quark.h"

extern void x11_luffaCubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffaCubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern int  x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x11_simd512_cpu_free(int thr_id);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

