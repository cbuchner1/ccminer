#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

static __device__ unsigned long long MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
#if __CUDA_ARCH__ >= 130
    return __double_as_longlong(__hiloint2double(HI, LO));
#else
	return (unsigned long long)LO | (((unsigned long long)HI) << 32);
#endif
}

// das Hi Word aus einem 64 Bit Typen extrahieren
static __device__ uint32_t HIWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2hiint(__longlong_as_double(x));
#else
	return (uint32_t)(x >> 32);
#endif
}

// das Hi Word in einem 64 Bit Typen ersetzen
static __device__ uint64_t REPLACE_HIWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32ULL);
}

// das Lo Word aus einem 64 Bit Typen extrahieren
static __device__ uint32_t LOWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2loint(__longlong_as_double(x));
#else
	return (uint32_t)(x & 0xFFFFFFFFULL);
#endif
}

// das Lo Word in einem 64 Bit Typen ersetzen
static __device__ uint64_t REPLACE_LOWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFF00000000ULL) | ((uint64_t)y);
}

// Endian Drehung für 32 Bit Typen
static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, x, 0x0123);
}

// Endian Drehung für 64 Bit Typen
static __device__ uint64_t cuda_swab64(uint64_t x) {
    return MAKE_ULONGLONG(cuda_swab32(HIWORD(x)), cuda_swab32(LOWORD(x)));
}






// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTR64(const uint64_t value, const int offset) {
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
#define ROTR64(x, n)        (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTL64(const uint64_t value, const int offset) {
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
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

__forceinline__ __device__ uint64_t xor1(uint64_t a, uint64_t b) {
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(a) ,"l"(b));
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

__forceinline__ __device__ uint64_t xandx(uint64_t a, uint64_t b, uint64_t c)
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
__forceinline__ __device__ uint64_t sph_t64(uint64_t x)
{
uint64_t result;
asm("{\n\t"
    "and.b64 %0,%1,0xFFFFFFFFFFFFFFFF;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x));
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
__forceinline__ __device__ uint64_t shr_t64(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	".reg .u64 m;\n\t"
	"shr.b64 m,%1,%2;\n\t"
    "and.b64 %0,m,0xFFFFFFFFFFFFFFFF;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}
__forceinline__ __device__ uint64_t shl_t64(uint64_t x,uint32_t n)
{
uint64_t result;
asm("{\n\t"
	".reg .u64 m;\n\t"
	"shl.b64 m,%1,%2;\n\t"
    "and.b64 %0,m,0xFFFFFFFFFFFFFFFF;\n\t"
    "}\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}


#endif // #ifndef CUDA_HELPER_H
