#include "cuda_helper.h"

/* Macros for uint2 operations (used by skein) */

__device__ __forceinline__
uint2 ROR8(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x4321);
	result.y = __byte_perm(a.y, a.x, 0x4321);
	return result;
}

__device__ __forceinline__
uint2 ROL24(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x0765);
	result.y = __byte_perm(a.y, a.x, 0x0765);
	return result;
}

static __device__ __forceinline__ uint2 operator+ (const uint2 a, const uint32_t b)
{
#if 0 && defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm(
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
	: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b), "r"(0));
	return result;
#else
	return vectorize(devectorize(a) + b);
#endif
}

/* whirlpool ones */
#ifdef __CUDA_ARCH__
__device__ __forceinline__
uint2 ROL16(const uint2 a) {
	uint2 result;
	result.x = __byte_perm(a.x, a.y, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x1076);
	return result;
}
#else
#define ROL16(a) make_uint2(a.x, a.y) /* bad, just to define it */
#endif

