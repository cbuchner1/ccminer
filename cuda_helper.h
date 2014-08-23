#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

static __device__ unsigned long long oMAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
#if __CUDA_ARCH__ >= 130
    return __double_as_longlong(__hiloint2double(HI, LO));
#else
	return (unsigned long long)LO | (((unsigned long long)HI) << 32);
#endif
}

static __device__ unsigned long long MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
uint64_t result;
asm("{\n\t"
	"mov.b64 %0,{%1,%2}; \n\t"
		"}"
		: "=l"(result) : "r"(LO) , "r"(HI));
return result;
}
static __device__ uint32_t HIWORD(uint64_t x)
{
uint32_t result;
asm("{\n\t"
	".reg .u32 xl; \n\t"
	"mov.b64 {xl,%0},%1; \n\t"
		"}"
		: "=r"(result) : "l"(x));
return result;
}
static __device__ uint32_t LOWORD(uint64_t x)
{
uint32_t result;
asm("{\n\t"
	".reg .u32 xh; \n\t"
	"mov.b64 {%0,xh},%1; \n\t"
		"}"
		: "=r"(result) : "l"(x));
return result;
}
// das Hi Word aus einem 64 Bit Typen extrahieren
static __device__ uint32_t oHIWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2hiint(__longlong_as_double(x));
#else
	return (uint32_t)(x >> 32);
#endif
}

#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define SPH_ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
    #define SPH_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
    // Kepler (Compute 3.5)
    #define SPH_ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
    #define SPH_ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

// das Hi Word in einem 64 Bit Typen ersetzen
static __device__ uint64_t oREPLACE_HIWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32ULL);
}

static __device__ uint64_t REPLACE_HIWORD(uint64_t x, uint32_t y) {
	asm("{\n\t"
		" .reg .u32 tl,th; \n\t"
		"mov.b64 {tl,th},%0; \n\t"
		"mov.b64 %0,{tl,%1}; \n\t"
		"}"
		: "+l"(x) : "r"(y) );
return x;
}

static __device__ uint64_t REPLACE_LOWORD(uint64_t x, uint32_t y) {
        asm("{\n\t"
		" .reg .u32 tl,th; \n\t"
		"mov.b64 {tl,th},%0; \n\t"
		"mov.b64 %0,{%1,th}; \n\t"
		"}"
		: "+l"(x) : "r"(y) );
return x;
}

// das Lo Word aus einem 64 Bit Typen extrahieren
static __device__ uint32_t oLOWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2loint(__longlong_as_double(x));
#else
	return (uint32_t)(x & 0xFFFFFFFFULL);
#endif
}

// das Lo Word in einem 64 Bit Typen ersetzen
static __device__ uint64_t oREPLACE_LOWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFF00000000ULL) | ((uint64_t)y);
}

// Endian Drehung für 32 Bit Typen
static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}

static __device__ uint64_t swap2ll(uint32_t lo, uint32_t hi)
{
return(MAKE_ULONGLONG(cuda_swab32(lo),cuda_swab32(hi)));
}

static __device__ uint64_t swap32toll(uint8_t* lo,uint8_t* hi)
{
uint64_t result;
asm("{\n\t"
		".reg .b32 l,h; \n\t"
		"mov.b32 l,{%4,%3,%2,%1}; \n\t"		
		"mov.b32 h,{%8,%7,%6,%5}; \n\t"
		"mov.b64 %0,{l,h}; \n\t"
		"}"
		: "=l"(result) : "r"(lo)  ,
		                 "r"(hi));
return result;
}
// Endian Drehung für 64 Bit Typen
static __device__ uint64_t cuda_swab64(uint64_t x) {
    return MAKE_ULONGLONG(cuda_swab32(HIWORD(x)), cuda_swab32(LOWORD(x)));
}
static __device__ uint64_t cuda_swab32ll(uint64_t x) {
    return MAKE_ULONGLONG(cuda_swab32(LOWORD(x)), cuda_swab32(HIWORD(x)));
}





// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t oROTR64(const uint64_t value, const int offset) {
    uint2 result;
    if(offset < 32) {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    } else {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    }
    return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
#define oROTR64(x, n)        (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t oROTL64(const uint64_t value, const int offset) {
    uint2 result;
    if(offset >= 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    } else {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    }
    return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
#define oROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTR64(const uint64_t value, const int offset) {
    uint64_t result;
    if(offset < 32) {
		asm("{\n\t"
		" .reg .u32 tl,th,vl,vh; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"shf.r.wrap.b32 vl,tl,th,%2; \n\t"
		"shf.r.wrap.b32 vh,th,tl,%2; \n\t"
		"mov.b64 %0,{vl,vh}; \n\t"
		"}"
		: "=l"(result) : "l"(value) , "r"(offset));
    } else {
		asm("{\n\t"
		" .reg .u32 tl,th,vl,vh; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"shf.r.wrap.b32 vh,tl,th,%2; \n\t"
		"shf.r.wrap.b32 vl,th,tl,%2; \n\t"
		"mov.b64 %0,{vl,vh}; \n\t"
		"}"
		: "=l"(result) : "l"(value) , "r"(offset));
    }
    return  result;
}
#else
#define ROTR64(x, n)        (((x) >> (n)) | ((x) << (64 - (n))))
#endif

#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTL64(const uint64_t value, const int offset) {
    uint64_t result;
    if(offset >= 32) {
		asm("{\n\t"
		" .reg .u32 tl,th,vl,vh; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"shf.l.wrap.b32 vl,tl,th,%2; \n\t"
		"shf.l.wrap.b32 vh,th,tl,%2; \n\t"
		"mov.b64 %0,{vl,vh}; \n\t"
		"}"
		: "=l"(result) : "l"(value) , "r"(offset));
    } else {
		asm("{\n\t"
		" .reg .u32 tl,th,vl,vh; \n\t"
		"mov.b64 {tl,th},%1; \n\t"
		"shf.l.wrap.b32 vh,tl,th,%2; \n\t"
		"shf.l.wrap.b32 vl,th,tl,%2; \n\t"
		"mov.b64 %0,{vl,vh}; \n\t"
		"}"
		: "=l"(result) : "l"(value) , "r"(offset));
    }
    return  result;
}
#else
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

/*
__device__ __forceinline__
uint64_t rotr_t64(uint64_t x, uint32_t n)
{
    uint64_t result;
    asm("{\n\t"
        ".reg .b64 lhs;\n\t"
        ".reg .b64 rhs;\n\t"
        ".reg .u32 amt2;\n\t"
        "shr.b64 lhs, %1, %2;\n\t"
        "sub.u32 amt2, 64, %2;\n\t"
        "shl.b64 rhs, %1, amt2;\n\t"
        "add.u64 %0, lhs, rhs;\n\t"
        "}\n\t"
    : "=l"(result) : "l"(x), "r"(n));
    return result;
}

// 64-bit ROTATE LEFT
__device__ __forceinline__
uint64_t rotl_t64(uint64_t x, uint32_t n)
{
    uint64_t result;
    asm("{\n\t"
        ".reg .b64 lhs;\n\t"
        ".reg .b64 rhs;\n\t"
        ".reg .u32 amt2;\n\t"
        "shl.b64 lhs, %1, %2;\n\t"
        "sub.u32 amt2, 64, %2;\n\t"
        "shr.b64 rhs, %1, amt2;\n\t"
        "add.u64 %0, lhs, rhs;\n\t"
        "}\n\t"
    : "=l"(result) : "l"(x), "r"(n));
    return result;
}
*/

__forceinline__ __device__ uint64_t xor1(uint64_t a, uint64_t b) {
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(a) ,"l"(b));
	return result;
}
__forceinline__ __device__ uint32_t xor1b(uint32_t a, uint32_t b) {
	uint32_t result;
	asm("xor.b32 %0, %1, %2;" : "=r"(result) : "r"(a) ,"r"(b));
	return result;
}

__forceinline__ __device__ uint64_t xor3(uint64_t a, uint64_t b, uint64_t c) {
	uint64_t result;
	asm("{\n\t"
		" .reg .u64 t1;\n\t"
		"xor.b64 t1, %2, %3;\n\t"
		"xor.b64 %0, %1, t1;\n\t" 
		"}"
		: "=l"(result) : "l"(a) ,"l"(b),"l"(c));
	return result;
}

__forceinline__ __device__ uint32_t xor3b(uint32_t a, uint32_t b, uint32_t c) {
	uint32_t result;
	asm("{\n\t"
		" .reg .u32 t1;\n\t"
		"xor.b32 t1, %2, %3;\n\t"
		"xor.b32 %0, %1, t1;\n\t" 
		"}"
		: "=r"(result) : "r"(a) ,"r"(b),"r"(c));
	return result;
}
__forceinline__ __device__ uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
	uint64_t result;
	asm("{\n\t"
		" .reg .u64 t1,t2,t3;\n\t"
		"xor.b64 t1, %1, %2;\n\t"
		"xor.b64 t2, %3, %4;\n\t"
		"xor.b64 t3, t1, t2;\n\t"
		"xor.b64 %0, t3,%5;\n\t"
		"}"
		: "=l"(result) : "l"(a) ,"l"(b), "l"(c), "l"(d) ,"l"(e));
	return result;
}


/*
__forceinline__ __device__ uint64_t xor4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
	uint64_t result;
	asm("{\n\t"
		" .reg .u64 m,n;\n\t"
		"xor.b64 m, %3, %4;\n\t"
		"xor.b64 n, %2, m;\n\t"
		"xor.b64 %0, %1, n;\n\t"
		"}\n\t"
		: "=l"(result) :"l"(a), "l"(b), "l"(c), "l"(d));
	return result;
}
*/
__forceinline__ __device__ uint64_t xor8(uint64_t a, uint64_t b, uint64_t c, uint64_t d,uint64_t e,uint64_t f,uint64_t g, uint64_t h) {
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(g) ,"l"(h));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(f));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(d));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}

__forceinline__ __device__ uint32_t xandx(uint32_t a, uint32_t b, uint32_t c)
{
	uint32_t result;
	asm("{\n\t"
		".reg .u32 m,n;\n\t"
		"xor.b32 m, %2,%3;\n\t"
		"and.b32 n, m,%1;\n\t"
		"xor.b32 %0, n,%3;\n\t"
		"}\n\t"
		: "=r"(result) : "r"(a), "r"(b), "r"(c));
	return result;

}
__forceinline__ __device__ uint64_t xandx64(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n;\n\t"
		"xor.b64 m, %2,%3;\n\t"
		"and.b64 n, m,%1;\n\t"
		"xor.b64 %0, n,%3;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;

}

__forceinline__ __device__ uint64_t xornot64(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n;\n\t"
		"not.b64 m,%2; \n\t"
		"or.b64 n, %1,m;\n\t"
		"xor.b64 %0, n,%3;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;

}

__forceinline__ __device__ void chi(uint64_t &s0, uint64_t &s1, uint64_t &s2, uint64_t &s3, uint64_t &s4)
{
	asm("{\n\t"
		".reg .u64 m0,m1,m2,m3,m4;\n\t"
		".reg .u64 z0,z1,z2,z3,z4;\n\t"
		"not.b64 m0,%0; \n\t"
		"not.b64 m1,%1; \n\t"
		"not.b64 m2,%2; \n\t"
		"not.b64 m3,%3; \n\t"
		"not.b64 m4,%4; \n\t"
		"and.b64 z1,m1,%2;\n\t"
		"and.b64 z2,m2,%3;\n\t"
		"and.b64 z3,m3,%4;\n\t"
		"and.b64 z4,m4,%0;\n\t"
		"and.b64 z0,m0,%1;\n\t"
		"xor.b64 %0,%0,z1;\n\t"
		"xor.b64 %1,%1,z2;\n\t"
		"xor.b64 %2,%2,z3;\n\t"
		"xor.b64 %3,%3,z4;\n\t"
		"xor.b64 %4,%4,z0;\n\t"		
		"}\n\t"
		: "+l"(s0),"+l"(s1),"+l"(s2),"+l"(s3),"+l"(s4));
}
__forceinline__ __device__ uint64_t xornt64(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n;\n\t"
		"not.b64 m,%3; \n\t"
		"or.b64 n, %2,m;\n\t"
		"xor.b64 %0, %1,n;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;

}
__forceinline__ __device__ uint64_t sph_t64(uint64_t x)
{
uint64_t result;
asm("{\n\t"
    "and.b64 %0,%1,0xFFFFFFFFFFFFFFFF;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x));
	return result;
}
__forceinline__ __device__ uint32_t sph_t32(uint32_t x)
{
uint32_t result;
asm("{\n\t"
    "and.b32 %0,%1,0xFFFFFFFF;\n\t"
    "}\n\t"
	: "=r"(result) : "r"(x));
	return result;
}

__forceinline__ __device__ uint64_t andor(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n,o;\n\t"
		"and.b64 m,  %1, %2;\n\t"
		" or.b64 n,  %1, %2;\n\t"
		"and.b64 o,   n, %3;\n\t"
		" or.b64 %0,  m, o ;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;

}
__forceinline__ __device__ uint32_t andor32(uint32_t a, uint32_t b, uint32_t c)
{
	uint32_t result;
	asm("{\n\t"
		".reg .u32 m,n,o;\n\t"
		"and.b32 m,  %1, %2;\n\t"
		" or.b32 n,  %1, %2;\n\t"
		"and.b32 o,   n, %3;\n\t"
		" or.b32 %0,  m, o ;\n\t"
		"}\n\t"
		: "=r"(result) : "r"(a), "r"(b), "r"(c));
	return result;

}
__forceinline__ __device__ uint64_t shr_t64(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	"shr.b64 %0,%1,%2;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint64_t shl_t64(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	"shl.b64 %0,%1,%2;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint32_t shr_t32(uint32_t x,uint32_t n)
{
uint32_t result;
asm("{\n\t"
	"shr.b32 %0,%1,%2;\n\t"
    "}\n\t"
	: "=r"(result) : "r"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint32_t shl_t32(uint32_t x,uint32_t n)
{
uint32_t result;
asm("{\n\t"
	"shl.b32 %0,%1,%2;\n\t"
    "}\n\t"
	: "=r"(result) : "r"(x), "r"(n));
	return result;
}
__forceinline__ __device__ void and64(uint64_t &d,uint64_t a,uint64_t b)
{
asm("and.b64 %0,%1,%2;" : "=l"(d) : "l"(a), "l"(b));
}

__forceinline__ __device__ void sbox(uint32_t &a, uint32_t &b,uint32_t &c,uint32_t &d)
{
uint32_t t; 
t = a;
asm("and.b32 %0,%0,%1;" : "+r"(a) : "r"(c));
asm("xor.b32 %0,%0,%1;" : "+r"(a) : "r"(d));
asm("xor.b32 %0,%0,%1;" : "+r"(c) : "r"(b));
asm("xor.b32 %0,%0,%1;" : "+r"(c) : "r"(a));
asm( "or.b32 %0,%0,%1;" : "+r"(d) : "r"(t));
asm("xor.b32 %0,%0,%1;" : "+r"(d) : "r"(b));
asm("xor.b32 %0,%0,%1;" : "+r"(t) : "r"(c));
b=d;
asm( "or.b32 %0,%0,%1;" : "+r"(d) : "r"(t));
asm("xor.b32 %0,%0,%1;" : "+r"(d) : "r"(a));
asm("and.b32 %0,%0,%1;" : "+r"(a) : "r"(b));
asm("xor.b32 %0,%0,%1;" : "+r"(t) : "r"(a));
asm("xor.b32 %0,%0,%1;" : "+r"(b) : "r"(d));
asm("xor.b32 %0,%0,%1;" : "+r"(b) : "r"(t));
a=c;
c=b;
b=d;
asm("not.b32 %0,%1;" : "=r"(d) : "r"(t));
//asm("xor.b32 %0,%0,0xFFFFFFFF;" : "+r"(d));
}
/*
__forceinline__ __device__ uint64_t byte64(uint64_t x,uint32_t n) 
{
uint64_t result;
unsigned res;
asm("shr.b64 %0,%1,%2;" : "=l"(result) : "l"(x),"r"(8*n)); 
res= (unsigned) result;
asm("and.b64 %0,%0,0x00000000000000FF;" : "+l"(res));
return res;
}
*/
__forceinline__ __device__ uint32_t byte(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	".reg .u64 m;\n\t"
	"shr.b64 m,%1,%2;\n\t"
	"and.b64 %0,m,0x00000000000000FF;\n\t"
	    "}\n\t" 
    : "=l"(result) : "l"(x) , "r"(8*n));
	return (uint32_t) result;
//asm("shr.b64 %0,%1,%2;" : "=l"(result) : "l"(x) , "r"(8*n));
//asm("and.b64 %0,%0,0x00000000000000FF;" : "+l"(result));
//	return (uint32_t) result;
}
/*
__forceinline__ __device__ uint64_t* mult128(uint64_t a, uint64_t b)
{
uint64_t c[2];
asm("mul.hi.u64 %0,%1,%2" : "=l"(c[1]) : "l"(a) , "l"(b));
asm("mul.lo.u64 %0,%1,%2" : "=l"(c[0]) : "l"(a) , "l"(b));
return c;
}
*/
__forceinline__ __device__ uint64_t mult128hi(uint64_t a, uint64_t b)
{
uint64_t c;
asm("mul.hi.u64 %0,%1,%2" : "=l"(c) : "l"(a) , "l"(b));
return c;
}
__forceinline__ __device__ uint64_t muladd128hi(uint64_t a, uint64_t b,uint64_t c,uint64_t e)
{
uint64_t d;
asm("{\n\t"
	".reg .u64 m;\n\t"
    "add.u64 m,%3,%4;\n\t" 
    "mad.hi.u64 %0,%1,%2,m;\n\t"     
	"}\n\t"
	: "=l"(d) : "l"(a), "l"(b), "l"(c), "l"(e));
return d;
}

__forceinline__ __device__ uint64_t muladd128lo(uint64_t a, uint64_t b,uint64_t c,uint64_t e)
{
uint64_t d;
asm("{\n\t"
	".reg .u64 m;\n\t"
    "add.u64 m,3%,%4;\n\t" 
    "mad.lo.u64 %0,%1,%2,m;\n\t"     
	"}\n\t"
	: "=l"(d) : "l"(a), "l"(b), "l"(c), "l"(e));
return d;
}


__forceinline__ __device__ uint64_t mult128lo(uint64_t a, uint64_t b)
{
uint64_t c;
asm("mul.lo.u64 %0,%1,%2" : "=l"(c) : "l"(a) , "l"(b));
return c;
}

__forceinline__ __device__ void muladd128(uint64_t &u,uint64_t &v,uint64_t a, uint64_t b,uint64_t &c,uint64_t &e)
{
asm("{\n\t"
	".reg .b64 abl,abh; \n\t"
	".reg .b32 abll,ablh,abhl,abhh,x1,x2,x3,x4; \n\t"
	".reg .b32 cl,ch,el,eh; \n\t"
	"mul.lo.u64 abl,%2,%3; \n\t"    
	"mul.hi.u64 abh,%2,%3; \n\t" 
	"mov.b64 {abll,ablh},abl; \n\t" 
	"mov.b64 {abhl,abhh},abh; \n\t" 
	"mov.b64 {cl,ch},%4; \n\t" 
	"mov.b64 {el,eh},%5; \n\t" 
    "add.cc.u32 x1,cl,el; \n\t"
	"addc.cc.u32 x2,ch,eh; \n\t" 
	"addc.u32 x3,0,0; \n\t" 
	"add.cc.u32 x1,x1,abll; \n\t"
	"addc.cc.u32 x2,x2,ablh; \n\t"  
	"addc.cc.u32 x3,x3,abhl; \n\t" 
	"addc.u32 x4,abhh,0; \n\t"
	"mov.b64 %1,{x1,x2}; \n\t"
	"mov.b64 %0,{x3,x4}; \n\t"
	"}\n\t"
	: "=l"(u), "=l"(v) : "l"(a) , "l"(b) , "l"(c) , "l"(e));
}
__forceinline__ __device__ uint64_t mul(uint64_t a,uint64_t b)
{
uint64_t result;
asm("{\n\t"
	"mul.lo.u64 %0,%1,%2; \n\t"    
     "}\n\t"
	: "=l"(result) : "l"(a) , "l"(b));
return result;
}

/*
__device__  void bigmul(void *wa, uint64_t* am, uint64_t* bm, int sizea, int sizeb)
{

uint64_t* w = (uint64_t*)wa;
//printf("coming here bigmul core routine %08x %08x  %d  %d \n",am[0],bm[0],sizea,sizeb);
#pragma unroll
for (int i=0;i<sizea+sizeb;i++) {w[i]=0;}
#pragma unroll
for (int i=0;i<sizeb;i++) {
	uint64_t c=0;
	uint64_t u=0,v=0;
    #pragma unroll
	for (int j=0;j<sizea;j++) {
    muladd128(u,v,am[j],bm[i],w[i+j],c);	
    w[i+j]=v;
    c=u;
	}
   w[i+sizea]=u;
 }

}
*/
#endif // #ifndef CUDA_HELPER_H
