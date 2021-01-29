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

typedef struct __align__(64) ulonglong2to8
{
ulonglong2 l0,l1,l2,l3;
} ulonglong2to8;

typedef struct __align__(128) ulonglong8to16
{
	ulonglong2to8 lo, hi;
} ulonglong8to16;

typedef struct __align__(256) ulonglong16to32
{
	ulonglong8to16 lo, hi;
} ulonglong16to32;

typedef struct __align__(512) ulonglong32to64
{
	ulonglong16to32 lo, hi;
} ulonglong32to64;



typedef struct __align__(1024) ulonglonglong
{
	ulonglong8to16 s0,s1,s2,s3,s4,s5,s6,s7;
} ulonglonglong;




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

typedef struct __align__(128) uint32
{

		uint16 lo,hi;
} uint32;



struct __align__(128) ulong8
{
	ulonglong4 s0, s1, s2, s3;
};
typedef __device_builtin__ struct ulong8 ulong8;


typedef struct  __align__(256) ulonglong16
{
	ulonglong2 s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sa, sb, sc, sd, se, sf;
} ulonglong16;

typedef struct  __align__(32) uint48
{
	uint4 s0, s1;

} uint48;

typedef struct  __align__(64) uint816
{
	uint48 s0, s1;

} uint816;

typedef struct  __align__(128) uint1632
{
	uint816 s0, s1;

} uint1632;

typedef struct  __align__(256) uintx64
{
	uint1632 s0, s1;

} uintx64;

typedef struct  __align__(512) uintx128
{
	uintx64 s0, s1;

} uintx128;

typedef struct  __align__(1024) uintx256
{
	uintx128 s0, s1;

} uintx256;



typedef struct __align__(256) uint4x16
{
	uint4 s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
} uint4x16;

static __inline__ __device__ ulonglong2to8 make_ulonglong2to8(ulonglong2 s0, ulonglong2 s1, ulonglong2 s2, ulonglong2 s3)
{
ulonglong2to8 t; t.l0=s0; t.l1=s1; t.l2=s2; t.l3=s3;
return t;
}

static __inline__ __device__ ulonglong8to16 make_ulonglong8to16(const ulonglong2to8 &s0, const ulonglong2to8 &s1)
{
	ulonglong8to16 t; t.lo = s0; t.hi = s1;
	return t;
}

static __inline__ __device__ ulonglong16to32 make_ulonglong16to32(const ulonglong8to16 &s0, const ulonglong8to16 &s1)
{
	ulonglong16to32 t; t.lo = s0; t.hi = s1;
	return t;
}

static __inline__ __device__ ulonglong32to64 make_ulonglong32to64(const ulonglong16to32 &s0, const ulonglong16to32 &s1)
{
	ulonglong32to64 t; t.lo = s0; t.hi = s1;
	return t;
}


static __inline__ __host__ __device__ ulonglonglong make_ulonglonglong(
	const ulonglong8to16 &s0, const ulonglong8to16 &s1, const ulonglong8to16 &s2, const ulonglong8to16 &s3,
	const ulonglong8to16 &s4, const ulonglong8to16 &s5, const ulonglong8to16 &s6, const ulonglong8to16 &s7)
{
	ulonglonglong t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}


static __inline__ __device__ uint48 make_uint48(uint4 s0, uint4 s1)
{
	uint48 t; t.s0 = s0; t.s1 = s1;
	return t;
}

static __inline__ __device__ uint816 make_uint816(const uint48 &s0, const uint48 &s1)
{
	uint816 t; t.s0 = s0; t.s1 = s1;
	return t;
}

static __inline__ __device__ uint1632 make_uint1632(const uint816 &s0, const uint816 &s1)
{
	uint1632 t; t.s0 = s0; t.s1 = s1;
	return t;
}

static __inline__ __device__ uintx64 make_uintx64(const uint1632 &s0, const uint1632 &s1)
{
	uintx64 t; t.s0 = s0; t.s1 = s1;
	return t;
}

static __inline__ __device__ uintx128 make_uintx128(const uintx64 &s0, const uintx64 &s1)
{
	uintx128 t; t.s0 = s0; t.s1 = s1;
	return t;
}

static __inline__ __device__ uintx256 make_uintx256(const uintx128 &s0, const uintx128 &s1)
{
	uintx256 t; t.s0 = s0; t.s1 = s1;
	return t;
}


static __inline__ __device__ uintx256 make_uintx64(const uintx128 &s0, const uintx128 &s1)
{
	uintx256 t; t.s0 = s0; t.s1 = s1;
	return t;
}


static __inline__ __host__ __device__ uint4x16 make_uint4x16(
	uint4 s0, uint4 s1, uint4 s2, uint4 s3, uint4 s4, uint4 s5, uint4 s6, uint4 s7,
	uint4 s8, uint4 s9, uint4 sa, uint4 sb, uint4 sc, uint4 sd, uint4 se, uint4 sf)
{
	uint4x16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.s10 = sa; t.s11 = sb; t.s12 = sc; t.s13 = sd; t.s14 = se; t.s15 = sf;
	return t;
}




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

static __inline__ __host__ __device__ uint32 make_uint32(const uint16 &a, const uint16 &b)
{
	uint32 t; t.lo = a; t.hi = b; return t;
}


static __inline__ __host__ __device__ uint8 make_uint8(
	unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7)
{
	uint8 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}


static __inline__ __host__ __device__ ulonglong16 make_ulonglong16(const ulonglong2 &s0, const ulonglong2 &s1,
	const ulonglong2 &s2, const ulonglong2 &s3, const ulonglong2 &s4, const ulonglong2 &s5, const ulonglong2 &s6, const ulonglong2 &s7,
	const ulonglong2 &s8, const ulonglong2 &s9,
	const ulonglong2 &sa, const ulonglong2 &sb, const ulonglong2 &sc, const ulonglong2 &sd, const ulonglong2 &se, const ulonglong2 &sf
) {
	ulonglong16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.sa = sa; t.sb = sb; t.sc = sc; t.sd = sd; t.se = se; t.sf = sf;
	return t;
}



static __inline__ __host__ __device__ ulong8 make_ulong8(
	ulonglong4 s0, ulonglong4 s1, ulonglong4 s2, ulonglong4 s3)
{
	ulong8 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3;// t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}


static __forceinline__ __device__ uchar4 operator^ (uchar4 a, uchar4 b) { return make_uchar4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ uchar4 operator+ (uchar4 a, uchar4 b) { return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }





static __forceinline__ __device__ uint4 operator^ (uint4 a, uint4 b) { return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ uint4 operator+ (uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }


static __forceinline__ __device__ ulonglong4 operator^ (ulonglong4 a, ulonglong4 b) { return make_ulonglong4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ ulonglong4 operator+ (ulonglong4 a, ulonglong4 b) { return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __forceinline__ __device__ ulonglong2 operator^ (ulonglong2 a, ulonglong2 b) { return make_ulonglong2(a.x ^ b.x, a.y ^ b.y); }
static __forceinline__ __device__ ulonglong2 operator+ (ulonglong2 a, ulonglong2 b) { return make_ulonglong2(a.x + b.x, a.y + b.y); }

static __forceinline__ __device__ ulong8 operator^ (const ulong8 &a, const ulong8 &b) {
	return make_ulong8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3);
} //, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7); }

static __forceinline__ __device__ ulong8 operator+ (const ulong8 &a, const ulong8 &b) {
	return make_ulong8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3);
} //, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7); }


static __forceinline__ __device__  __host__ uint8 operator^ (const uint8 &a, const uint8 &b) { return make_uint8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7); }

static __forceinline__ __device__  __host__ uint8 operator+ (const uint8 &a, const uint8 &b) { return make_uint8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7); }

////////////// mess++ //////

static __forceinline__ __device__  uint48 operator^ (const uint48 &a, const uint48 &b) {
	return make_uint48(a.s0 ^ b.s0, a.s1 ^ b.s1);
}

static __forceinline__ __device__  uint816 operator^ (const uint816 &a, const uint816 &b) {
	return make_uint816(a.s0 ^ b.s0, a.s1 ^ b.s1);
}

static __forceinline__ __device__ uint1632 operator^ (const uint1632 &a, const uint1632 &b) {
	return make_uint1632(a.s0 ^ b.s0, a.s1 ^ b.s1);
}


static __forceinline__ __device__  uintx64 operator^ (const uintx64 &a, const uintx64 &b) {
	return make_uintx64(a.s0 ^ b.s0, a.s1 ^ b.s1);
}

static __forceinline__ __device__  uintx128 operator^ (const uintx128 &a, const uintx128 &b) {
	return make_uintx128(a.s0 ^ b.s0, a.s1 ^ b.s1);
}

static __forceinline__ __device__  uintx256 operator^ (const uintx256 &a, const uintx256 &b) {
	return make_uintx256(a.s0 ^ b.s0, a.s1 ^ b.s1);
}

/////////////////////////

static __forceinline__ __device__ __host__ uint16 operator^ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
		a.s8 ^ b.s8, a.s9 ^ b.s9, a.sa ^ b.sa, a.sb ^ b.sb, a.sc ^ b.sc, a.sd ^ b.sd, a.se ^ b.se, a.sf ^ b.sf);
}

static __forceinline__ __device__  __host__ uint16 operator+ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
		a.s8 + b.s8, a.s9 + b.s9, a.sa + b.sa, a.sb + b.sb, a.sc + b.sc, a.sd + b.sd, a.se + b.se, a.sf + b.sf);
}

static __forceinline__ __device__  uint32 operator^ (const uint32 &a, const uint32 &b) {
	return make_uint32(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__  uint32 operator+ (const uint32 &a, const uint32 &b) {
	return make_uint32(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__ ulonglong16 operator^ (const ulonglong16 &a, const ulonglong16 &b) {
	return make_ulonglong16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
		a.s8 ^ b.s8, a.s9 ^ b.s9, a.sa ^ b.sa, a.sb ^ b.sb, a.sc ^ b.sc, a.sd ^ b.sd, a.se ^ b.se, a.sf ^ b.sf
);
}

static __forceinline__ __device__ ulonglong16 operator+ (const ulonglong16 &a, const ulonglong16 &b) {
	return make_ulonglong16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
		a.s8 + b.s8, a.s9 + b.s9, a.sa + b.sa, a.sb + b.sb, a.sc + b.sc, a.sd + b.sd, a.se + b.se, a.sf + b.sf
);
}

static __forceinline__ __device__ void operator^= (ulong8 &a, const ulong8 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (uintx64 &a, const uintx64 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator^= (uintx128 &a, const uintx128 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (uintx256 &a, const uintx256 &b) { a = a ^ b; }


static __forceinline__ __device__ void operator^= (uint816 &a, const uint816 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator^= (uint48 &a, const uint48 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator^= (uint32 &a, const uint32 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (uint32 &a, const uint32 &b) { a = a + b; }


static __forceinline__ __device__ void operator^= (uint4 &a, uint4 b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (uchar4 &a, uchar4 b) { a = a ^ b; }
static __forceinline__ __device__  __host__ void operator^= (uint8 &a, const uint8 &b) { a = a ^ b; }
static __forceinline__ __device__  __host__ void operator^= (uint16 &a, const uint16 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator^= (ulonglong16 &a, const ulonglong16 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (ulonglong4 &a, const ulonglong4 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (ulonglong2 &a, const ulonglong2 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (ulonglong2 &a, const ulonglong2 &b) { a = a + b; }

static __forceinline__ __device__
ulonglong2to8 operator^ (const ulonglong2to8 &a, const ulonglong2to8 &b)
{
	return make_ulonglong2to8(a.l0 ^ b.l0, a.l1 ^ b.l1, a.l2 ^ b.l2, a.l3 ^ b.l3);
}
static __forceinline__ __device__
ulonglong2to8 operator+ (const ulonglong2to8 &a, const ulonglong2to8 &b)
{
	return make_ulonglong2to8(a.l0 + b.l0, a.l1 + b.l1, a.l2 + b.l2, a.l3 + b.l3);
}


static __forceinline__ __device__
ulonglong8to16 operator^ (const ulonglong8to16 &a, const ulonglong8to16 &b)
{
	return make_ulonglong8to16(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__
ulonglong8to16 operator+ (const ulonglong8to16 &a, const ulonglong8to16 &b)
{
	return make_ulonglong8to16(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__
ulonglong16to32 operator^ (const ulonglong16to32 &a, const ulonglong16to32 &b)
{
	return make_ulonglong16to32(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__
ulonglong16to32 operator+ (const ulonglong16to32 &a, const ulonglong16to32 &b)
{
	return make_ulonglong16to32(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__
ulonglong32to64 operator^ (const ulonglong32to64 &a, const ulonglong32to64 &b)
{
	return make_ulonglong32to64(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__
ulonglong32to64 operator+ (const ulonglong32to64 &a, const ulonglong32to64 &b)
{
	return make_ulonglong32to64(a.lo + b.lo, a.hi + b.hi);
}


static __forceinline__ __device__ ulonglonglong operator^ (const ulonglonglong &a, const ulonglonglong &b) {
	return make_ulonglonglong(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7);
}

static __forceinline__ __device__ ulonglonglong operator+ (const ulonglonglong &a, const ulonglonglong &b) {
	return make_ulonglonglong(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7);
}


static __forceinline__ __device__ void operator^= (ulonglong2to8 &a, const ulonglong2to8 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (uint4 &a, uint4 b) { a = a + b; }
static __forceinline__ __device__ void operator+= (uchar4 &a, uchar4 b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator+= (uint8 &a, const uint8 &b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator+= (uint16 &a, const uint16 &b) { a = a + b; }
static __forceinline__ __device__ void operator+= (ulong8 &a, const ulong8 &b) { a = a + b; }
static __forceinline__ __device__ void operator+= (ulonglong16 &a, const ulonglong16 &b) { a = a + b; }
static __forceinline__ __device__ void operator+= (ulonglong8to16 &a, const ulonglong8to16 &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglong8to16 &a, const ulonglong8to16 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (ulonglong16to32 &a, const ulonglong16to32 &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglong16to32 &a, const ulonglong16to32 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (ulonglong32to64 &a, const ulonglong32to64 &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglong32to64 &a, const ulonglong32to64 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (ulonglonglong &a, const ulonglonglong &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglonglong &a, const ulonglonglong &b) { a = a ^ b; }

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

#endif

#if __CUDA_ARCH__ < 320

// right shift a 64-bytes integer (256-bits) by 0 8 16 24 bits
// require a uint32_t[9] ret array
// note: djm neoscrypt implementation is near the limits of gpu capabilities
//       and weird behaviors can happen when tuning device functions code...
__device__ static void shift256R(uint32_t* ret, const uint8 &vec4, uint32_t shift)
{
	uint8_t *v = (uint8_t*) &vec4.s0;
	uint8_t *r = (uint8_t*) ret;
	uint8_t bytes = (uint8_t) (shift >> 3);
	ret[0] = 0;
	for (uint8_t i=bytes; i<32; i++)
		r[i] = v[i-bytes];
	ret[8] = vec4.s7 >> (32 - shift); // shuffled part required
}

#else

// same for SM 3.5+, really faster ?
__device__ static void shift256R(uint32_t* ret, const uint8 &vec4, uint32_t shift)
{
	uint32_t truc = 0, truc2 = cuda_swab32(vec4.s7), truc3 = 0;
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc3), "r"(truc2), "r"(shift));
	ret[8] = cuda_swab32(truc);
	truc3 = cuda_swab32(vec4.s6);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc2), "r"(truc3), "r"(shift));
	ret[7] = cuda_swab32(truc);
	truc2 = cuda_swab32(vec4.s5);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc3), "r"(truc2), "r"(shift));
	ret[6] = cuda_swab32(truc);
	truc3 = cuda_swab32(vec4.s4);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc2), "r"(truc3), "r"(shift));
	ret[5] = cuda_swab32(truc);
	truc2 = cuda_swab32(vec4.s3);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc3), "r"(truc2), "r"(shift));
	ret[4] = cuda_swab32(truc);
	truc3 = cuda_swab32(vec4.s2);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc2), "r"(truc3), "r"(shift));
	ret[3] = cuda_swab32(truc);
	truc2 = cuda_swab32(vec4.s1);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc3), "r"(truc2), "r"(shift));
	ret[2] = cuda_swab32(truc);
	truc3 = cuda_swab32(vec4.s0);
	asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(truc) : "r"(truc2), "r"(truc3), "r"(shift));
	ret[1] = cuda_swab32(truc);
	asm("shr.b32        %0, %1, %2;" : "=r"(truc) : "r"(truc3), "r"(shift));
	ret[0] = cuda_swab32(truc);
}
#endif

#if __CUDA_ARCH__ < 320

// copy 256 bytes
static __device__ __inline__ uintx64 ldg256(const uint4 *ptr)
{
	uintx64 ret;
	uint32_t *dst = (uint32_t*) &ret.s0;
	uint32_t *src = (uint32_t*) &ptr[0].x;
	for (int i=0; i < (256 / sizeof(uint32_t)); i++) {
		dst[i] = src[i];
	}
	return ret;
}

#else

// complicated way to copy 256 bytes ;)
static __device__ __inline__ uintx64 ldg256(const uint4 *ptr)
{
	uintx64 ret;
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.s0.s0.s0.s0.x), "=r"(ret.s0.s0.s0.s0.y), "=r"(ret.s0.s0.s0.s0.z), "=r"(ret.s0.s0.s0.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];"  : "=r"(ret.s0.s0.s0.s1.x), "=r"(ret.s0.s0.s0.s1.y), "=r"(ret.s0.s0.s0.s1.z), "=r"(ret.s0.s0.s0.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+32];"  : "=r"(ret.s0.s0.s1.s0.x), "=r"(ret.s0.s0.s1.s0.y), "=r"(ret.s0.s0.s1.s0.z), "=r"(ret.s0.s0.s1.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+48];"  : "=r"(ret.s0.s0.s1.s1.x), "=r"(ret.s0.s0.s1.s1.y), "=r"(ret.s0.s0.s1.s1.z), "=r"(ret.s0.s0.s1.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+64];"  : "=r"(ret.s0.s1.s0.s0.x), "=r"(ret.s0.s1.s0.s0.y), "=r"(ret.s0.s1.s0.s0.z), "=r"(ret.s0.s1.s0.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+80];"  : "=r"(ret.s0.s1.s0.s1.x), "=r"(ret.s0.s1.s0.s1.y), "=r"(ret.s0.s1.s0.s1.z), "=r"(ret.s0.s1.s0.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+96];"  : "=r"(ret.s0.s1.s1.s0.x), "=r"(ret.s0.s1.s1.s0.y), "=r"(ret.s0.s1.s1.s0.z), "=r"(ret.s0.s1.s1.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+112];"  : "=r"(ret.s0.s1.s1.s1.x), "=r"(ret.s0.s1.s1.s1.y), "=r"(ret.s0.s1.s1.s1.z), "=r"(ret.s0.s1.s1.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+128];"  : "=r"(ret.s1.s0.s0.s0.x), "=r"(ret.s1.s0.s0.s0.y), "=r"(ret.s1.s0.s0.s0.z), "=r"(ret.s1.s0.s0.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+144];"  : "=r"(ret.s1.s0.s0.s1.x), "=r"(ret.s1.s0.s0.s1.y), "=r"(ret.s1.s0.s0.s1.z), "=r"(ret.s1.s0.s0.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+160];"  : "=r"(ret.s1.s0.s1.s0.x), "=r"(ret.s1.s0.s1.s0.y), "=r"(ret.s1.s0.s1.s0.z), "=r"(ret.s1.s0.s1.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+176];"  : "=r"(ret.s1.s0.s1.s1.x), "=r"(ret.s1.s0.s1.s1.y), "=r"(ret.s1.s0.s1.s1.z), "=r"(ret.s1.s0.s1.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+192];"  : "=r"(ret.s1.s1.s0.s0.x), "=r"(ret.s1.s1.s0.s0.y), "=r"(ret.s1.s1.s0.s0.z), "=r"(ret.s1.s1.s0.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+208];"  : "=r"(ret.s1.s1.s0.s1.x), "=r"(ret.s1.s1.s0.s1.y), "=r"(ret.s1.s1.s0.s1.z), "=r"(ret.s1.s1.s0.s1.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+224];"  : "=r"(ret.s1.s1.s1.s0.x), "=r"(ret.s1.s1.s1.s0.y), "=r"(ret.s1.s1.s1.s0.z), "=r"(ret.s1.s1.s1.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+240];"  : "=r"(ret.s1.s1.s1.s1.x), "=r"(ret.s1.s1.s1.s1.y), "=r"(ret.s1.s1.s1.s1.z), "=r"(ret.s1.s1.s1.s1.w) : __LDG_PTR(ptr));
	return ret;
}
#endif

#endif // #ifndef CUDA_VECTOR_H
