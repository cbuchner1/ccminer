#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#ifdef __INTELLISENSE__
#define __launch_bounds__(x)
#define __byte_perm(x,y,z)
#endif

static __device__ void LOHI(uint32_t &lo, uint32_t &hi, uint64_t x)
{
	asm("{\n\t"
		"mov.b64 {%0,%1},%2; \n\t"
		"}"
		: "=r"(lo), "=r"(hi) : "l"(x));
}

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
asm volatile ("{\n\t"
	"mov.b64 %0,{%1,%2}; \n\t"
		"}"
		: "=l"(result) : "r"(LO) , "r"(HI));
return result;
}
static __device__ uint32_t HIWORD(uint64_t x)
{
uint32_t result;
asm volatile ("{\n\t"
	".reg .u32 xl; \n\t"
	"mov.b64 {xl,%0},%1; \n\t"
		"}"
		: "=r"(result) : "l"(x));
return result;
}
static __device__ uint32_t LOWORD(uint64_t x)
{
uint32_t result;
asm volatile ("{\n\t"
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
    #define SPH_ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
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
	asm volatile("{\n\t"
		" .reg .u32 tl,th; \n\t"
		"mov.b64 {tl,th},%0; \n\t"
		"mov.b64 %0,{tl,%1}; \n\t"
		"}"
		: "+l"(x) : "r"(y) );
return x;
}


static __device__ uint64_t REPLACE_LOWORD(uint64_t x, uint32_t y) {
        asm volatile ("{\n\t"
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
	return __byte_perm(x, 0, 0x0123);
}

static __device__ uint64_t swap2ll(uint32_t lo, uint32_t hi)
{
return(MAKE_ULONGLONG(cuda_swab32(lo),cuda_swab32(hi)));
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

// Wolf0 Rotate
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTR64(const uint64_t x, const int y)
{
	uint64_t res;
		
	asm("{\n\t"
			".reg .u32 tl,th,vl,vh;\n\t"
			".reg .pred p;\n\t"
			"mov.b64 {tl,th}, %1;\n\t"
			"shf.r.wrap.b32 vl, tl, th, %2;\n\t"
			"shf.r.wrap.b32 vh, th, tl, %2;\n\t"
			"setp.lt.u32 p, %2, 32;\n\t"
			"@p mov.b64 %0, {vl,vh};\n\t"
			"@!p mov.b64 %0, {vh,vl};\n\t"
			"}" : "=l"(res) : "l"(x) , "r"(y));
	
	return res;
}
#else
#define ROTR64(x, n)        (((x) >> (n)) | ((x) << (64 - (n))))
#endif

#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTL64(const uint64_t x, const int y)
{
	uint64_t res;
		
	asm("{\n\t"
			".reg .u32 tl,th,vl,vh;\n\t"
			".reg .pred p;\n\t"
			"mov.b64 {tl,th}, %1;\n\t"
			"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
			"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
			"setp.lt.u32 p, %2, 32;\n\t"
			"@!p mov.b64 %0, {vl,vh};\n\t"
			"@p mov.b64 %0, {vh,vl};\n\t"
			"}" : "=l"(res) : "l"(x) , "r"(y));
	
	return res;
}
#else
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

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



__forceinline__ __device__ uint64_t xor8(uint64_t a, uint64_t b, uint64_t c, uint64_t d,uint64_t e,uint64_t f,uint64_t g, uint64_t h) {
	uint64_t result;
	asm volatile ("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(g) ,"l"(h));
	asm volatile ("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(f));
	asm volatile ("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(e));
	asm volatile ("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(d));
	asm volatile ("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm volatile ("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm volatile ("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
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
}




__forceinline__ __device__ void muladd128(uint64_t &u,uint64_t &v,uint64_t a, uint64_t b,uint64_t &c,uint64_t &e)
{

	asm("{\n\t"
		".reg .b32 al,ah,bl,bh; \n\t"
		".reg .b32 x1,x2,x3,x4; \n\t"
		".reg .b32 cl,ch,el,eh; \n\t"
		"mov.b64 {al,ah},%2; \n\t"
		"mov.b64 {bl,bh},%3; \n\t"
		"mov.b64 {cl,ch},%4; \n\t"
		"mov.b64 {el,eh},%5; \n\t"
		"add.cc.u32 x1,cl,el; \n\t"
		"addc.cc.u32 x2,ch,eh; \n\t"
		"addc.u32 x3,0,0; \n\t"
		"mad.lo.cc.u32 x1,bl,al,x1; \n\t"
		"madc.hi.cc.u32 x2,bl,al,x2; \n\t"
		"addc.u32      x3,x3,0;         \n\t"
		"mad.lo.cc.u32    x2,bh,al,x2; \n\t"
		"madc.hi.cc.u32   x3,bh,al,x3;    \n\t"
		"addc.u32         x4,0,0;         \n\t"
		"mad.lo.cc.u32  x2,bl,ah,x2;  \n\t"
		"madc.hi.cc.u32 x3,bl,ah,x3;  \n\t"
		"addc.u32       x4,x4,0;         \n\t"
		"mad.lo.cc.u32  x3,bh,ah,x3;   \n\t"
		"madc.hi.u32    x4,bh,ah,x4;   \n\t"
		"mov.b64 %1,{x1,x2}; \n\t"
		"mov.b64 %0,{x3,x4}; \n\t"
		"}\n\t"
		: "=l"(u), "=l"(v) : "l"(a), "l"(b), "l"(c), "l"(e));

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

__device__ __forceinline__ uint64_t shfl(uint64_t x, int lane)
{
uint32_t lo,hi;
asm volatile("mov.b64 {%0,%1},%2;" : "=r"(lo), "=r"(hi) : "l"(x));
lo = __shfl(lo, lane);
hi = __shfl(hi, lane);
asm volatile("mov.b64 %0,{%1,%2};" : "=l"(x) : "r"(lo) , "r"(hi));
return x;
}


///uint2 method

#if  __CUDA_ARCH__ >= 350 
__inline__ __device__ uint2 ROR2(const uint2 a, const int offset) {
	uint2 result;
	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
		
	}
	return result;
}
#else
__inline__ __device__ uint2 ROR2(const uint2 v, const int a) {
		uint2 result;
        int n = 64 -a; //lazy
		if (n <= 32) {
			result.y = ((v.y << (n)) | (v.x >> (32 - n)));
			result.x = ((v.x << (n)) | (v.y >> (32 - n)));
		}
		else {
			result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
			result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
		}
		return result;
	}
#endif


#if  __CUDA_ARCH__ >= 350 
__inline__ __device__ uint2 ROL2(const uint2 a, const int offset) {
	uint2 result;
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));        
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
return result;
}
#else
__inline__ __device__ uint2 ROL2(const uint2 v, const int n) {
		uint2 result;
		if (n == 32) {result.x = v.y;result.y=v.x;}
		if (n < 32) {
			result.y = ((v.y << (n)) | (v.x >> (32 - n)));
			result.x = ((v.x << (n)) | (v.y >> (32 - n)));
		}
		else {
			result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
			result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
		}
		return result;
	}
#endif

static __forceinline__ __device__ uint64_t devectorize(uint2 v) { return MAKE_ULONGLONG(v.x, v.y); }
static __forceinline__ __device__ uint2 vectorize(uint64_t v) {
	uint2 result;
	LOHI(result.x, result.y, v);
	return result;
}

static __forceinline__ __device__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __forceinline__ __device__ uint2 operator& (uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
static __forceinline__ __device__ uint2 operator| (uint2 a, uint2 b) { return make_uint2(a.x | b.x, a.y | b.y); }
static __forceinline__ __device__ uint2 operator~ (uint2 a) { return make_uint2(~a.x, ~a.y); }
static __forceinline__ __device__ void operator^= (uint2 &a, uint2 b) { a = a ^ b; }
static __forceinline__ __device__ uint2 operator+ (uint2 a, uint2 b)
{
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}
static __forceinline__ __device__ void operator+= (uint2 &a, uint2 b) { a = a + b; }

static __forceinline__ __device__ uint2 operator* (uint2 a, uint2 b)
{ //basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b)) 
	//(what does uint64 "*" operator) 
	uint2 result;
	asm("{\n\t"
		"mul.lo.u32        %0,%2,%4;  \n\t"
		"mul.hi.u32        %1,%2,%4;  \n\t"
		"mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
		"madc.lo.u32      %1,%3,%5,%1; \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}
#if  __CUDA_ARCH__ >= 350 
static __forceinline__ __device__ uint2 shiftl2(uint2 a, int offset)
{
	uint2 result;
	if (offset<32) {
		asm("{\n\t"
			"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
			"shl.b32 %0,%2,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else {
		asm("{\n\t"
			"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
			"shl.b32 %0,%2,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
}
static __forceinline__ __device__ uint2 shiftr2(uint2 a, int offset)
{
	uint2 result;
	if (offset<32) {
		asm("{\n\t"
			"shf.r.clamp.b32 %0,%2,%3,%4; \n\t"
			"shr.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else {
		asm("{\n\t"
			"shf.l.clamp.b32 %0,%2,%3,%4; \n\t"
			"shl.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
}
#else 
static __forceinline__ __device__ uint2 shiftl2(uint2 a, int offset)
{
	uint2 result;
	asm("{\n\t"
		".reg .b64 u,v; \n\t"
		"mov.b64 v,{%2,%3}; \n\t"
		"shl.b64 u,v,%4; \n\t"
		"mov.b64 {%0,%1},v;  \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}
static __forceinline__ __device__ uint2 shiftr2(uint2 a, int offset)
{
	uint2 result;
	asm("{\n\t"
		".reg .b64 u,v; \n\t"
		"mov.b64 v,{%2,%3}; \n\t"
		"shr.b64 u,v,%4; \n\t"
		"mov.b64 {%0,%1},v;  \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}
#endif
///////////////////////////////////////////////////////////////////////////////////


#endif // #ifndef CUDA_HELPER_H
