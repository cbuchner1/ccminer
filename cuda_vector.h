#ifndef CUDA_VECTOR_H
#define CUDA_VECTOR_H


///////////////////////////////////////////////////////////////////////////////////
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

#include "cuda_helper.h"

//typedef __device_builtin__ struct ulong16 ulong16;

typedef struct __align__(32) uint8
{
	unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
} uint8;

typedef struct __align__(64) uint16
{
	union {
		struct {unsigned int  s0, s1, s2, s3, s4, s5, s6, s7;};
		uint8 lo;
	};
	union {
		struct {unsigned int s8, s9, sa, sb, sc, sd, se, sf;};
		uint8 hi;
	};
} uint16;


static __inline__ __host__ __device__ uint16 make_uint16(
	unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7,
	unsigned int s8, unsigned int s9, unsigned int sa, unsigned int sb, unsigned int sc, unsigned int sd, unsigned int se, unsigned int sf)
{
	uint16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.sa = sa; t.sb = sb; t.sc = sc; t.sd = sd; t.se = se; t.sf = sf;
	return t;
}

static __inline__ __host__ __device__ uint16 make_uint16(const uint8 &a, const uint8 &b)
{
	uint16 t; t.lo=a; t.hi=b; return t;
}

static __inline__ __host__ __device__ uint8 make_uint8(
	unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7)
{
	uint8 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}


static __forceinline__ __device__ uchar4 operator^ (uchar4 a, uchar4 b) { return make_uchar4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ uchar4 operator+ (uchar4 a, uchar4 b) { return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }


static __forceinline__ __device__ uint4 operator^ (uint4 a, uint4 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ uint4 operator+ (uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }


static __forceinline__ __device__ ulonglong4 operator^ (ulonglong4 a, ulonglong4 b) { return make_ulonglong4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ ulonglong4 operator+ (ulonglong4 a, ulonglong4 b) { return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __forceinline__ __device__ ulonglong2 operator^ (ulonglong2 a, ulonglong2 b) { return make_ulonglong2(a.x ^ b.x, a.y ^ b.y); }


static __forceinline__ __device__  __host__ uint8 operator^ (const uint8 &a, const uint8 &b) { return make_uint8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7); }

static __forceinline__ __device__  __host__ uint8 operator+ (const uint8 &a, const uint8 &b) { return make_uint8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7); }

static __forceinline__ __device__ __host__ uint16 operator^ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
		a.s8 ^ b.s8, a.s9 ^ b.s9, a.sa ^ b.sa, a.sb ^ b.sb, a.sc ^ b.sc, a.sd ^ b.sd, a.se ^ b.se, a.sf ^ b.sf);
}

static __forceinline__ __device__  __host__ uint16 operator+ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
		a.s8 + b.s8, a.s9 + b.s9, a.sa + b.sa, a.sb + b.sb, a.sc + b.sc, a.sd + b.sd, a.se + b.se, a.sf + b.sf);
}

static __forceinline__ __device__ void operator^= (uint4 &a, uint4 b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (uchar4 &a, uchar4 b) { a = a ^ b; }
static __forceinline__ __device__  __host__ void operator^= (uint8 &a, const uint8 &b) { a = a ^ b; }
static __forceinline__ __device__  __host__ void operator^= (uint16 &a, const uint16 &b) { a = a ^ b; }


static __forceinline__ __device__ void operator^= (ulonglong4 &a, const ulonglong4 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (ulonglong2 &a, const ulonglong2 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (uint4 &a, uint4 b) { a = a + b; }
static __forceinline__ __device__ void operator+= (uchar4 &a, uchar4 b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator+= (uint8 &a, const uint8 &b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator+= (uint16 &a, const uint16 &b) { a = a + b; }

#if __CUDA_ARCH__ < 320

#define rotateL ROTL32
#define rotateR ROTR32

#else

static __forceinline__ __device__ uint32_t rotateL(uint32_t vec4, uint32_t shift)
{
	uint32_t ret;
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(vec4), "r"(vec4), "r"(shift));
	return ret;
}

static __forceinline__ __device__ uint32_t rotateR(uint32_t vec4, uint32_t shift)
{
	uint32_t ret;
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(vec4), "r"(vec4), "r"(shift));
	return ret;
}

static __device__ __inline__ uint8 __ldg8(const uint8_t *ptr)
{
	uint8 test;
	asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(test.s0), "=r"(test.s1), "=r"(test.s2), "=r"(test.s3) : __LDG_PTR(ptr));
	asm volatile ("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4+16];" : "=r"(test.s4), "=r"(test.s5), "=r"(test.s6), "=r"(test.s7) : __LDG_PTR(ptr));
	return (test);
}

static __device__ __inline__ uint32_t __ldgtoint(const uint8_t *ptr)
{
	uint32_t test;
	asm volatile ("ld.global.nc.u32 {%0},[%1];" : "=r"(test) : __LDG_PTR(ptr));
	return (test);
}

static __device__ __inline__ uint32_t __ldgtoint64(const uint8_t *ptr)
{
	uint64_t test;
	asm volatile ("ld.global.nc.u64 {%0},[%1];" : "=l"(test) : __LDG_PTR(ptr));
	return (test);
}


static __device__ __inline__ uint32_t __ldgtoint_unaligned(const uint8_t *ptr)
{
	uint32_t test;
	asm volatile ("{\n\t"
		".reg .u8 a,b,c,d; \n\t"
	"ld.global.nc.u8 a,[%1]; \n\t"
	"ld.global.nc.u8 b,[%1+1]; \n\t"
	"ld.global.nc.u8 c,[%1+2]; \n\t"
	"ld.global.nc.u8 d,[%1+3]; \n\t"
	"mov.b32 %0,{a,b,c,d}; }\n\t"
		: "=r"(test) : __LDG_PTR(ptr));
	return (test);
}

static __device__ __inline__ uint64_t __ldgtoint64_unaligned(const uint8_t *ptr)
{
	uint64_t test;
	asm volatile ("{\n\t"
		".reg .u8 a,b,c,d,e,f,g,h; \n\t"
		".reg .u32 i,j; \n\t"
		"ld.global.nc.u8 a,[%1]; \n\t"
		"ld.global.nc.u8 b,[%1+1]; \n\t"
		"ld.global.nc.u8 c,[%1+2]; \n\t"
		"ld.global.nc.u8 d,[%1+3]; \n\t"
		"ld.global.nc.u8 e,[%1+4]; \n\t"
		"ld.global.nc.u8 f,[%1+5]; \n\t"
		"ld.global.nc.u8 g,[%1+6]; \n\t"
		"ld.global.nc.u8 h,[%1+7]; \n\t"
		 "mov.b32 i,{a,b,c,d}; \n\t"
         "mov.b32 j,{e,f,g,h}; \n\t"
		 "mov.b64 %0,{i,j}; }\n\t"
		: "=l"(test) : __LDG_PTR(ptr));
	return (test);
}


static __device__ __inline__ uint64_t __ldgtoint64_trunc(const uint8_t *ptr)
{
	uint32_t zero = 0;
	uint64_t test;
	asm volatile ("{\n\t"
		".reg .u8 a,b,c,d; \n\t"
		".reg .u32 i; \n\t"
		"ld.global.nc.u8 a,[%1]; \n\t"
		"ld.global.nc.u8 b,[%1+1]; \n\t"
		"ld.global.nc.u8 c,[%1+2]; \n\t"
		"ld.global.nc.u8 d,[%1+3]; \n\t"
		"mov.b32 i,{a,b,c,d}; \n\t"
		"mov.b64 %0,{i,%1}; }\n\t"
		: "=l"(test) : __LDG_PTR(ptr), "r"(zero));
	return (test);
}



static __device__ __inline__ uint32_t __ldgtoint_unaligned2(const uint8_t *ptr)
{
	uint32_t test;
	asm("{\n\t"
		".reg .u8 e,b,c,d; \n\t"
		"ld.global.nc.u8 e,[%1]; \n\t"
		"ld.global.nc.u8 b,[%1+1]; \n\t"
		"ld.global.nc.u8 c,[%1+2]; \n\t"
		"ld.global.nc.u8 d,[%1+3]; \n\t"
		"mov.b32 %0,{e,b,c,d}; }\n\t"
		: "=r"(test) : __LDG_PTR(ptr));
	return (test);
}

#endif


static __forceinline__ __device__ uint8 swapvec(const uint8 *buf)
{
	uint8 vec;
	vec.s0 = cuda_swab32(buf[0].s0);
	vec.s1 = cuda_swab32(buf[0].s1);
	vec.s2 = cuda_swab32(buf[0].s2);
	vec.s3 = cuda_swab32(buf[0].s3);
	vec.s4 = cuda_swab32(buf[0].s4);
	vec.s5 = cuda_swab32(buf[0].s5);
	vec.s6 = cuda_swab32(buf[0].s6);
	vec.s7 = cuda_swab32(buf[0].s7);
	return vec;
}

static __forceinline__ __device__ uint16 swapvec(const uint16 *buf)
{
	uint16 vec;
	vec.s0 = cuda_swab32(buf[0].s0);
	vec.s1 = cuda_swab32(buf[0].s1);
	vec.s2 = cuda_swab32(buf[0].s2);
	vec.s3 = cuda_swab32(buf[0].s3);
	vec.s4 = cuda_swab32(buf[0].s4);
	vec.s5 = cuda_swab32(buf[0].s5);
	vec.s6 = cuda_swab32(buf[0].s6);
	vec.s7 = cuda_swab32(buf[0].s7);
	vec.s8 = cuda_swab32(buf[0].s8);
	vec.s9 = cuda_swab32(buf[0].s9);
	vec.sa = cuda_swab32(buf[0].sa);
	vec.sb = cuda_swab32(buf[0].sb);
	vec.sc = cuda_swab32(buf[0].sc);
	vec.sd = cuda_swab32(buf[0].sd);
	vec.se = cuda_swab32(buf[0].se);
	vec.sf = cuda_swab32(buf[0].sf);
	return vec;
}
#endif // #ifndef CUDA_VECTOR_H
