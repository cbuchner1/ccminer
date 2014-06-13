#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>

#define USE_SHUFFLE 0

// Folgende Definitionen später durch header ersetzen
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// die Message it Padding zur Berechnung auf der GPU
__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

__constant__ uint8_t c_sigma[16][16];

const uint8_t host_sigma[16][16] =
{
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
  {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
  {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
  { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

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

__device__ __forceinline__ uint64_t SWAP64(uint64_t x)
{
	// Input:	77665544 33221100
	// Output:	00112233 44556677
	uint64_t temp[2];
	temp[0] = __byte_perm(HIWORD(x), 0, 0x0123);
	temp[1] = __byte_perm(LOWORD(x), 0, 0x0123);

	return temp[0] | (temp[1]<<32);
}

__constant__ uint64_t c_u512[16];

const uint64_t host_u512[16] =
{
  0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 
  0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
  0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 
  0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
  0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL, 
  0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
  0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL, 
  0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};


// diese 64 Bit Rotates werden unter Compute 3.5 (und besser) mit dem Funnel Shifter beschleunigt
#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTR(const uint64_t value, const int offset) {
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
#define ROTR(x, n)        (((x) >> (n)) | ((x) << (64 - (n))))
#endif

#define G(a,b,c,d,e)          \
    v[a] += (m[sigma[i][e]] ^ u512[sigma[i][e+1]]) + v[b];\
    v[d] = ROTR( v[d] ^ v[a],32);        \
    v[c] += v[d];           \
    v[b] = ROTR( v[b] ^ v[c],25);        \
    v[a] += (m[sigma[i][e+1]] ^ u512[sigma[i][e]])+v[b];  \
    v[d] = ROTR( v[d] ^ v[a],16);        \
    v[c] += v[d];           \
    v[b] = ROTR( v[b] ^ v[c],11);


__device__ void quark_blake512_compress( uint64_t *h, const uint64_t *block, const uint8_t ((*sigma)[16]), const uint64_t *u512, const int bits )
{
    uint64_t v[16], m[16], i;

#pragma unroll 16
    for( i = 0; i < 16; ++i )
    {
        m[i] = SWAP64(block[i]);
    }

#pragma unroll 8
    for( i = 0; i < 8; ++i )  v[i] = h[i];

    v[ 8] = u512[0];
    v[ 9] = u512[1];
    v[10] = u512[2];
    v[11] = u512[3];
    v[12] = u512[4];
    v[13] = u512[5];
    v[14] = u512[6];
    v[15] = u512[7];

    v[12] ^= bits;
    v[13] ^= bits;

//#pragma unroll 16
    for( i = 0; i < 16; ++i )
    {
        /* column step */
        G( 0, 4, 8, 12, 0 );
        G( 1, 5, 9, 13, 2 );
        G( 2, 6, 10, 14, 4 );
        G( 3, 7, 11, 15, 6 );
        /* diagonal step */
        G( 0, 5, 10, 15, 8 );
        G( 1, 6, 11, 12, 10 );
        G( 2, 7, 8, 13, 12 );
        G( 3, 4, 9, 14, 14 );
    }

#pragma unroll 16
    for( i = 0; i < 16; ++i )  h[i % 8] ^= v[i];
}

// Endian Drehung für 32 Bit Typen

static __device__ uint32_t cuda_swab32(uint32_t x)
{
	return __byte_perm(x, 0, 0x0123);
}

/*
// Endian Drehung für 64 Bit Typen
static __device__ uint64_t cuda_swab64(uint64_t x) {
    uint32_t h = (x >> 32);
    uint32_t l = (x & 0xFFFFFFFFULL);
    return (((uint64_t)cuda_swab32(l)) << 32) | ((uint64_t)cuda_swab32(h));
}
*/

static __constant__ uint64_t d_constMem[8];
static const uint64_t h_constMem[8] = {
	0x6a09e667f3bcc908ULL,
	0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL,
	0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL,
	0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL,
	0x5be0cd19137e2179ULL };

// Hash-Padding
static __constant__ uint64_t d_constHashPadding[8];
static const uint64_t h_constHashPadding[8] = {
	0x0000000000000080ull,
	0,
	0,
	0,
	0,
	0x0100000000000000ull,
	0,
	0x0002000000000000ull };

__global__ __launch_bounds__(256, 2) void quark_blake512_gpu_hash_64(int threads, uint32_t startNounce, uint32_t *g_nonceVector, uint64_t *g_hash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);

#if USE_SHUFFLE
	const int warpID = threadIdx.x & 0x0F; // 16 warps
	const int warpBlockID = (thread + 15)>>4; // aufrunden auf volle Warp-Blöcke
	const int maxHashPosition = thread<<3;
#endif

#if USE_SHUFFLE
	if (warpBlockID < ( (threads+15)>>4 ))
#else
	if (thread < threads)
#endif
	{
		// bestimme den aktuellen Zähler
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		//uint64_t *inpHash = &g_hash[8 * hashPosition];
		uint64_t *inpHash = &g_hash[hashPosition<<3];

		// State vorbereiten
		uint64_t h[8];
		/*
		h[0] = 0x6a09e667f3bcc908ULL;
		h[1] = 0xbb67ae8584caa73bULL;
		h[2] = 0x3c6ef372fe94f82bULL;
		h[3] = 0xa54ff53a5f1d36f1ULL;
		h[4] = 0x510e527fade682d1ULL;
		h[5] = 0x9b05688c2b3e6c1fULL;
		h[6] = 0x1f83d9abfb41bd6bULL;
		h[7] = 0x5be0cd19137e2179ULL;
		*/
#pragma unroll 8
		for(int i=0;i<8;i++)
			h[i] = d_constMem[i];

		// 128 Byte für die Message
		uint64_t buf[16];

		// Message für die erste Runde in Register holen
#pragma unroll 8
		for (int i=0; i < 8; ++i) buf[i] = inpHash[i];

		/*
		buf[ 8] = 0x0000000000000080ull;
		buf[ 9] = 0;
		buf[10] = 0;
		buf[11] = 0;
		buf[12] = 0;
		buf[13] = 0x0100000000000000ull;
		buf[14] = 0;
		buf[15] = 0x0002000000000000ull;
		*/
#pragma unroll 8
		for(int i=0;i<8;i++)
			buf[i+8] = d_constHashPadding[i];

		// die einzige Hashing-Runde
		quark_blake512_compress( h, buf, c_sigma, c_u512, 512 );

		// Hash rauslassen
#if __CUDA_ARCH__ >= 130
		// ausschliesslich 32 bit Operationen sofern die SM1.3 double intrinsics verfügbar sind
		uint32_t *outHash = (uint32_t*)&g_hash[8 * hashPosition];
#pragma unroll 8
		for (int i=0; i < 8; ++i) {
			outHash[2*i+0] = cuda_swab32( HIWORD(h[i]) );
			outHash[2*i+1] = cuda_swab32( LOWORD(h[i]) );
		}
#else
		// in dieser Version passieren auch ein paar 64 Bit Shifts
		uint64_t *outHash = &g_hash[8 * hashPosition];
#pragma unroll 8
		for (int i=0; i < 8; ++i)
		{
			//outHash[i] = cuda_swab64( h[i] );
			outHash[i] = SWAP64(h[i]);
		}
#endif
	}
}

__global__ void quark_blake512_gpu_hash_80(int threads, uint32_t startNounce, void *outputHash)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		uint32_t nounce = startNounce + thread;

		// State vorbereiten
		uint64_t h[8];
		/*
		h[0] = 0x6a09e667f3bcc908ULL;
		h[1] = 0xbb67ae8584caa73bULL;
		h[2] = 0x3c6ef372fe94f82bULL;
		h[3] = 0xa54ff53a5f1d36f1ULL;
		h[4] = 0x510e527fade682d1ULL;
		h[5] = 0x9b05688c2b3e6c1fULL;
		h[6] = 0x1f83d9abfb41bd6bULL;
		h[7] = 0x5be0cd19137e2179ULL;
		*/
#pragma unroll 8
		for(int i=0;i<8;i++)
			h[i] = d_constMem[i];
		// 128 Byte für die Message
		uint64_t buf[16];

		// Message für die erste Runde in Register holen
#pragma unroll 16
		for (int i=0; i < 16; ++i) buf[i] = c_PaddedMessage80[i];

		// die Nounce durch die thread-spezifische ersetzen
		buf[9] = REPLACE_HIWORD(buf[9], cuda_swab32(nounce));

		// die einzige Hashing-Runde
		quark_blake512_compress( h, buf, c_sigma, c_u512, 640 );

		// Hash rauslassen
#if __CUDA_ARCH__ >= 130
		// ausschliesslich 32 bit Operationen sofern die SM1.3 double intrinsics verfügbar sind
		uint32_t *outHash = (uint32_t *)outputHash + 16 * thread;
#pragma unroll 8
		for (int i=0; i < 8; ++i) {
			outHash[2*i+0] = cuda_swab32( HIWORD(h[i]) );
			outHash[2*i+1] = cuda_swab32( LOWORD(h[i]) );
		}
#else
		// in dieser Version passieren auch ein paar 64 Bit Shifts
		uint64_t *outHash = (uint64_t *)outputHash + 8 * thread;
#pragma unroll 8
		for (int i=0; i < 8; ++i)
		{
			//outHash[i] = cuda_swab64( h[i] );
			outHash[i] = SWAP64(h[i]);
		}
#endif
	}
}


// ---------------------------- END CUDA quark_blake512 functions ------------------------------------

// Setup-Funktionen
__host__ void quark_blake512_cpu_init(int thr_id, int threads)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol( c_sigma,
						host_sigma,
						sizeof(host_sigma),
						0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( c_u512,
						host_u512,
						sizeof(host_u512),
						0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( d_constMem,
						h_constMem,
						sizeof(h_constMem),
						0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( d_constHashPadding,
						h_constHashPadding,
						sizeof(h_constHashPadding),
						0, cudaMemcpyHostToDevice);
}

// Blake512 für 80 Byte grosse Eingangsdaten
__host__ void quark_blake512_cpu_setBlock_80(void *pdata)
{
	// Message mit Padding bereitstellen
	// lediglich die korrekte Nonce ist noch ab Byte 76 einzusetzen.
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);
	PaddedMessage[80] = 0x80;
	PaddedMessage[111] = 1;
	PaddedMessage[126] = 0x02;
	PaddedMessage[127] = 0x80;

	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_blake512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	quark_blake512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, d_nonceVector, (uint64_t*)d_outputHash);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void quark_blake512_cpu_hash_80(int thr_id, int threads, uint32_t startNounce, uint32_t *d_outputHash, int order)
{
	const int threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	quark_blake512_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash);

	// Strategisches Sleep Kommando zur Senkung der CPU Last
	MyStreamSynchronize(NULL, order, thr_id);
}
