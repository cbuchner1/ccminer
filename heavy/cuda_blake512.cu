#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// globaler Speicher für alle HeftyHashes aller Threads
extern uint32_t *heavy_heftyHashes[MAX_GPUS];
extern uint32_t *heavy_nonceVector[MAX_GPUS];

// globaler Speicher für unsere Ergebnisse
uint32_t *d_hash5output[MAX_GPUS];

// die Message (112 bzw. 116 Bytes) mit Padding zur Berechnung auf der GPU
__constant__ uint64_t c_PaddedMessage[16]; // padded message (80/84+32 bytes + padding)


// ---------------------------- BEGIN CUDA blake512 functions ------------------------------------

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

/* in cuda_helper */
#define SWAP32(x) cuda_swab32(x)
#define SWAP64(x) cuda_swab64(x)

__constant__ uint64_t c_SecondRound[15];

const uint64_t host_SecondRound[15] =
{
  0,0,0,0,0,0,0,0,0,0,0,0,0,SWAP64(1),0
};

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


#define G(a,b,c,d,e)          \
    v[a] += (m[sigma[i][e]] ^ u512[sigma[i][e+1]]) + v[b];\
    v[d] = SWAPDWORDS( v[d] ^ v[a]);        \
    v[c] += v[d];           \
    v[b] = ROTR64( v[b] ^ v[c],25);        \
    v[a] += (m[sigma[i][e+1]] ^ u512[sigma[i][e]])+v[b];  \
    v[d] = ROTR64( v[d] ^ v[a],16);        \
    v[c] += v[d];           \
    v[b] = ROTR64( v[b] ^ v[c],11);

template <int BLOCKSIZE> __device__ void blake512_compress( uint64_t *h, const uint64_t *block, int nullt, const uint8_t ((*sigma)[16]), const uint64_t *u512 )
{
    uint64_t v[16], m[16], i;

#pragma unroll 16
    for( i = 0; i < 16; ++i )  m[i] = cuda_swab64(block[i]);

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

    /* don't xor t when the block is only padding */
    if ( !nullt ) {
        v[12] ^= 8*(BLOCKSIZE+32);
        v[13] ^= 8*(BLOCKSIZE+32);
    }

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

template <int BLOCKSIZE> __global__ void blake512_gpu_hash(uint32_t threads, uint32_t startNounce, void *outputHash, uint32_t *heftyHashes, uint32_t *nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		//uint32_t nounce = startNounce + thread;
		uint32_t nounce = nonceVector[thread];

		// Index-Position des Hashes in den Hash Puffern bestimmen (Hefty1 und outputHash)
		uint32_t hashPosition = nounce - startNounce;

		// State vorbereiten
		uint64_t h[8];
		h[0] = 0x6a09e667f3bcc908ULL;
		h[1] = 0xbb67ae8584caa73bULL;
		h[2] = 0x3c6ef372fe94f82bULL;
		h[3] = 0xa54ff53a5f1d36f1ULL;
		h[4] = 0x510e527fade682d1ULL;
		h[5] = 0x9b05688c2b3e6c1fULL;
		h[6] = 0x1f83d9abfb41bd6bULL;
		h[7] = 0x5be0cd19137e2179ULL;

		// 128 Byte für die Message
		uint64_t buf[16];

		// Message für die erste Runde in Register holen
#pragma unroll 16
		for (int i=0; i < 16; ++i) buf[i] = c_PaddedMessage[i];

		// die Nounce durch die thread-spezifische ersetzen
		buf[9] = REPLACE_HIDWORD(buf[9], nounce);

		uint32_t *hefty = heftyHashes + 8 * hashPosition;
		if (BLOCKSIZE == 84) {
			// den thread-spezifischen Hefty1 hash einsetzen
			// aufwändig, weil das nicht mit uint64_t Wörtern aligned ist.
			buf[10] = REPLACE_HIDWORD(buf[10], hefty[0]);
			buf[11] = REPLACE_LODWORD(buf[11], hefty[1]);
			buf[11] = REPLACE_HIDWORD(buf[11], hefty[2]);
			buf[12] = REPLACE_LODWORD(buf[12], hefty[3]);
			buf[12] = REPLACE_HIDWORD(buf[12], hefty[4]);
			buf[13] = REPLACE_LODWORD(buf[13], hefty[5]);
			buf[13] = REPLACE_HIDWORD(buf[13], hefty[6]);
			buf[14] = REPLACE_LODWORD(buf[14], hefty[7]);
		}
		else if (BLOCKSIZE == 80) {
			buf[10] = MAKE_ULONGLONG(hefty[0], hefty[1]);
			buf[11] = MAKE_ULONGLONG(hefty[2], hefty[3]);
			buf[12] = MAKE_ULONGLONG(hefty[4], hefty[5]);
			buf[13] = MAKE_ULONGLONG(hefty[6], hefty[7]);
		}

		// erste Runde
		blake512_compress<BLOCKSIZE>( h, buf, 0, c_sigma, c_u512 );


		// zweite Runde
#pragma unroll 15
		for (int i=0; i < 15; ++i) buf[i] = c_SecondRound[i];
		buf[15] = SWAP64(8*(BLOCKSIZE+32)); // Blocksize in Bits einsetzen
		blake512_compress<BLOCKSIZE>( h, buf, 1, c_sigma, c_u512 );

		// Hash rauslassen
		uint64_t *outHash = (uint64_t *)outputHash + 8 * hashPosition;
#pragma unroll 8
		for (int i=0; i < 8; ++i) outHash[i] = cuda_swab64( h[i] );
	}
}


// ---------------------------- END CUDA blake512 functions ------------------------------------

// Setup Function
__host__
void blake512_cpu_init(int thr_id, uint32_t threads)
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

	cudaMemcpyToSymbol( c_SecondRound,
						host_SecondRound,
						sizeof(host_SecondRound),
						0, cudaMemcpyHostToDevice);

	// Speicher für alle Ergebnisse belegen
	CUDA_SAFE_CALL(cudaMalloc(&d_hash5output[thr_id], (size_t) 64 * threads));
}

__host__
void blake512_cpu_free(int thr_id)
{
	cudaFree(d_hash5output[thr_id]);
}

static int BLOCKSIZE = 84;

__host__
void blake512_cpu_setBlock(void *pdata, int len)
	// data muss 84-Byte haben!
	// heftyHash hat 32-Byte
{
	unsigned char PaddedMessage[128];
	if (len == 84) {
		// Message mit Padding für erste Runde bereitstellen
		memcpy(PaddedMessage, pdata, 84);
		memset(PaddedMessage+84, 0, 32); // leeres Hefty Hash einfüllen
		memset(PaddedMessage+116, 0, 12);
		PaddedMessage[116] = 0x80;
	} else if (len == 80) {
		memcpy(PaddedMessage, pdata, 80);
		memset(PaddedMessage+80, 0, 32); // leeres Hefty Hash einfüllen
		memset(PaddedMessage+112, 0, 16);
		PaddedMessage[112] = 0x80;
	}
	// die Message (116 Bytes) ohne Padding zur Berechnung auf der GPU
	cudaMemcpyToSymbol( c_PaddedMessage, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	BLOCKSIZE = len;
}

__host__ void blake512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	if (BLOCKSIZE == 80)
		blake512_gpu_hash<80><<<grid, block, shared_size>>>(threads, startNounce, d_hash5output[thr_id], heavy_heftyHashes[thr_id], heavy_nonceVector[thr_id]);
	else if (BLOCKSIZE == 84)
		blake512_gpu_hash<84><<<grid, block, shared_size>>>(threads, startNounce, d_hash5output[thr_id], heavy_heftyHashes[thr_id], heavy_nonceVector[thr_id]);
}
