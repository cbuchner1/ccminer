#ifndef CUDA_VECTOR_UINT2x4_H
#define CUDA_VECTOR_UINT2x4_H

///////////////////////////////////////////////////////////////////////////////////
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

#include "cuda_helper.h"

typedef struct __align__(16) uint2x4 {
	uint2 x, y, z, w;
} uint2x4;


static __inline__ __device__ uint2x4 make_uint2x4(uint2 s0, uint2 s1, uint2 s2, uint2 s3)
{
	uint2x4 t;
	t.x = s0; t.y = s1; t.z = s2; t.w = s3;
	return t;
}

static __forceinline__ __device__  uint2x4 operator^ (const uint2x4 &a, const uint2x4 &b) {
	return make_uint2x4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

static __forceinline__ __device__  uint2x4 operator+ (const uint2x4 &a, const uint2x4 &b) {
	return make_uint2x4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/////////////////////////

static __forceinline__ __device__ void operator^= (uint2x4 &a, const uint2x4 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (uint2x4 &a, const uint2x4 &b) { a = a + b; }

#if __CUDA_ARCH__ >= 320

static __device__ __inline__ uint2x4 __ldg4(const uint2x4 *ptr)
{
	uint2x4 ret;
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"    : "=r"(ret.x.x), "=r"(ret.x.y), "=r"(ret.y.x), "=r"(ret.y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(ret.z.x), "=r"(ret.z.y), "=r"(ret.w.x), "=r"(ret.w.y) : __LDG_PTR(ptr));
	return ret;
}

static __device__ __inline__ void ldg4(const uint2x4 *ptr, uint2x4 *ret)
{
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"     : "=r"(ret[0].x.x), "=r"(ret[0].x.y), "=r"(ret[0].y.x), "=r"(ret[0].y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];"  : "=r"(ret[0].z.x), "=r"(ret[0].z.y), "=r"(ret[0].w.x), "=r"(ret[0].w.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+32];"  : "=r"(ret[1].x.x), "=r"(ret[1].x.y), "=r"(ret[1].y.x), "=r"(ret[1].y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+48];"  : "=r"(ret[1].z.x), "=r"(ret[1].z.y), "=r"(ret[1].w.x), "=r"(ret[1].w.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+64];"  : "=r"(ret[2].x.x), "=r"(ret[2].x.y), "=r"(ret[2].y.x), "=r"(ret[2].y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+80];"  : "=r"(ret[2].z.x), "=r"(ret[2].z.y), "=r"(ret[2].w.x), "=r"(ret[2].w.y) : __LDG_PTR(ptr));
}
#elif !defined(__ldg4)
#define __ldg4(x) (*(x))
#define ldg4(ptr, ret) { *(ret) = (*(ptr)); }
#endif

#endif // H
