#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#endif

#include <stdint.h>

// common functions
extern void cuda_check_cpu_init(int thr_id, int threads);
extern void cuda_check_cpu_setTarget(const void *ptarget);
extern uint32_t cuda_check_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

extern __device__ __device_builtin__ void __syncthreads(void);

#ifndef __CUDA_ARCH__
// define blockDim and threadIdx for host
extern const dim3 blockDim;
extern const uint3 threadIdx;
#endif

#ifndef SPH_C32
#define SPH_C32(x) ((uint32_t)(x ## U))
#endif

#ifndef SPH_C64
#define SPH_C64(x) ((uint64_t)(x ## ULL))
#endif

#define SPH_T32(x) ((x) & SPH_C32(0xFFFFFFFF))

#if __CUDA_ARCH__ < 320
// Kepler (Compute 3.0)
#define ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#else
// Kepler (Compute 3.5, 5.0)
#define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#endif

__device__ __forceinline__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
#if __CUDA_ARCH__ >= 130
	return __double_as_longlong(__hiloint2double(HI, LO));
#else
	return (uint64_t)LO | (((uint64_t)HI) << 32);
#endif
}

// das Hi Word in einem 64 Bit Typen ersetzen
__device__ __forceinline__ uint64_t REPLACE_HIWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32ULL);
}

// das Lo Word in einem 64 Bit Typen ersetzen
__device__ __forceinline__ uint64_t REPLACE_LOWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFF00000000ULL) | ((uint64_t)y);
}

// Endian Drehung für 32 Bit Typen
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
	/* device */
	return __byte_perm(x, x, 0x0123);
}
#else
	/* host */
	#define cuda_swab32(x) \
	((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | \
		(((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

// das Lo Word aus einem 64 Bit Typen extrahieren
__device__ __forceinline__ uint32_t _LOWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2loint(__longlong_as_double(x));
#else
	return (uint32_t)(x & 0xFFFFFFFFULL);
#endif
}

// das Hi Word aus einem 64 Bit Typen extrahieren
__device__ __forceinline__ uint32_t _HIWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2hiint(__longlong_as_double(x));
#else
	return (uint32_t)(x >> 32);
#endif
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint64_t cuda_swab64(uint64_t x)
{
	// Input:       77665544 33221100
	// Output:      00112233 44556677
	uint64_t result = __byte_perm((uint32_t) x, 0, 0x0123);
	return (result << 32) | __byte_perm(_HIWORD(x), 0, 0x0123);
}
#else
	/* host */
	#define cuda_swab64(x) \
		((uint64_t)((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
			(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
			(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
			(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
			(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
			(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
			(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
			(((uint64_t)(x) & 0x00000000000000ffULL) << 56)))
#endif

/*********************************************************************/
// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
		         __FILE__, __LINE__, cudaGetErrorString(err) );       \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)

/*********************************************************************/
#ifdef _WIN64
#define USE_XOR_ASM_OPTS 0
#else
#define USE_XOR_ASM_OPTS 1
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor1(uint64_t a, uint64_t b)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
	return result;
}
#else
#define xor1(a,b) (a ^ b)
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor3(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("xor.b64 %0, %2, %3;\n\t"
	    "xor.b64 %0, %0, %1;\n\t"
		/* output : input registers */
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}
#else
#define xor3(a,b,c) (a ^ b ^ c)
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor8(uint64_t a, uint64_t b, uint64_t c, uint64_t d,uint64_t e,uint64_t f,uint64_t g, uint64_t h)
{
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
#else
#define xor8(a,b,c,d,e,f,g,h) (a^b^c^d^e^f^g^h)
#endif

// device asm for x17
__device__ __forceinline__
uint64_t xandx(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 n;\n\t"
		"xor.b64 %0, %2, %3;\n\t"
		"and.b64 n, %0, %1;\n\t"
		"xor.b64 %0, n, %3;"
	"}\n"
	: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

// device asm for x17
__device__ __forceinline__
uint64_t sph_t64(uint64_t x)
{
	uint64_t result;
	asm("{\n\t"
		"and.b64 %0,%1,0xFFFFFFFFFFFFFFFF;\n\t"
	"}\n"
	: "=l"(result) : "l"(x));
	return result;
}

// device asm for x17
__device__ __forceinline__
uint64_t andor(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("{\n\t"
		".reg .u64 m,n;\n\t"
		"and.b64 m,  %1, %2;\n\t"
		" or.b64 n,  %1, %2;\n\t"
		"and.b64 %0, n,  %3;\n\t"
		" or.b64 %0, %0, m ;\n\t"
	"}\n"
	: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}

// device asm for x17
__device__ __forceinline__
uint64_t shr_t64(uint64_t x, uint32_t n)
{
	uint64_t result;
	asm("shr.b64 %0,%1,%2;\n\t"
		"and.b64 %0,%0,0xFFFFFFFFFFFFFFFF;\n\t" /* useful ? */
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}

// device asm for ?
__device__ __forceinline__
uint64_t shl_t64(uint64_t x, uint32_t n)
{
	uint64_t result;
	asm("shl.b64 %0,%1,%2;\n\t"
		"and.b64 %0,%0,0xFFFFFFFFFFFFFFFF;\n\t" /* useful ? */
	: "=l"(result) : "l"(x), "r"(n));
	return result;
}

#ifndef USE_ROT_ASM_OPT
#define USE_ROT_ASM_OPT 1
#endif

// 64-bit ROTATE RIGHT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
/* complicated sm >= 3.5 one (with Funnel Shifter beschleunigt), to bench */
__device__ __forceinline__
uint64_t ROTR64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTR64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shr.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shl.b64 %0, %1, roff;\n\t"
		"add.u64 %0, %0, lhs;\n\t"
	"}\n"
	: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#else
/* host */
#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
__device__ __forceinline__
uint64_t ROTL64(const uint64_t value, const int offset) {
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
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{\n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shl.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shr.b64 %0, %1, roff;\n\t"
		"add.u64 %0, lhs, %0;\n\t"
	"}\n"
	: "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#elif __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 3
__device__
uint64_t ROTL64(const uint64_t x, const int offset)
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
		"@p  mov.b64 %0, {vh,vl};\n\t"
		"}"
		: "=l"(res) : "l"(x) , "r"(offset)
	);
	return res;
}
#else
/* host */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif

__device__ __forceinline__
uint64_t SWAPDWORDS(uint64_t value)
{
#if __CUDA_ARCH__ >= 320
	uint2 temp;
	asm("mov.b64 {%0, %1}, %2; ": "=r"(temp.x), "=r"(temp.y) : "l"(value));
	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(temp.y), "r"(temp.x));
	return value;
#else
	return ROTL64(value, 32);
#endif
}

#endif // #ifndef CUDA_HELPER_H
