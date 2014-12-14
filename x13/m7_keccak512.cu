
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdint.h>
#include <memory.h>


extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern int compute_version[8];

#include "cuda_helper.h"
static __constant__ uint64_t stateo[25];
static __constant__ uint64_t RC[24];
static const uint64_t cpu_RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull,
    0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull,
    0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull,
    0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull,
    0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull,
    0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull,
    0x0000000080000001ull, 0x8000000080008008ull
};

static __device__ __forceinline__ void keccak_block(uint64_t *s, const uint64_t *keccak_round_constants) {
    size_t i;
    uint64_t t[5], u[5], v, w;

    /* absorb input */    
    
//#pragma unroll 24
    for (i = 0; i < 24; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		
        t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24]; 
		 
        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		
		uint64_t temp0,temp1,temp2,temp3,temp4;
        temp0 = ROTL64(t[0], 1);
		temp1 = ROTL64(t[1], 1);
		temp2 = ROTL64(t[2], 1);
		temp3 = ROTL64(t[3], 1);
		temp4 = ROTL64(t[4], 1);
		u[0] = xor1(t[4],temp1);
        u[1] = xor1(t[0],temp2);
        u[2] = xor1(t[1],temp3);
        u[3] = xor1(t[2],temp4);
        u[4] = xor1(t[3],temp0);
		
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
        s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
        s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
        s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
        s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
        s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

        /* rho pi: b[..] = rotl(a[..], ..) */
        v = s[ 1];
        s[ 1] = ROTL64(s[ 6], 44);
        s[ 6] = ROTL64(s[ 9], 20);
        s[ 9] = ROTL64(s[22], 61);
        s[22] = ROTL64(s[14], 39);
        s[14] = ROTL64(s[20], 18);
        s[20] = ROTL64(s[ 2], 62);
        s[ 2] = ROTL64(s[12], 43);
        s[12] = ROTL64(s[13], 25);
        s[13] = ROTL64(s[19],  8);
        s[19] = ROTL64(s[23], 56);
        s[23] = ROTL64(s[15], 41);
        s[15] = ROTL64(s[ 4], 27);
        s[ 4] = ROTL64(s[24], 14);
        s[24] = ROTL64(s[21],  2);
        s[21] = ROTL64(s[ 8], 55);
        s[ 8] = ROTL64(s[16], 45);
        s[16] = ROTL64(s[ 5], 36);
        s[ 5] = ROTL64(s[ 3], 28);
        s[ 3] = ROTL64(s[18], 21);
        s[18] = ROTL64(s[17], 15);
        s[17] = ROTL64(s[11], 10);
        s[11] = ROTL64(s[ 7],  6);
        s[ 7] = ROTL64(s[10],  3);
        s[10] = ROTL64(    v,  1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */		

		v = s[ 0]; w = s[ 1]; 
		s[ 0] ^= (~w) & s[ 2]; 
		s[ 1] ^= (~s[ 2]) & s[ 3]; 
		s[ 2] ^= (~s[ 3]) & s[ 4]; 
		s[ 3] ^= (~s[ 4]) & v; 
		s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6];
		s[ 5] ^= (~w) & s[ 7];
		s[ 6] ^= (~s[ 7]) & s[ 8];
		s[ 7] ^= (~s[ 8]) & s[ 9];
		s[ 8] ^= (~s[ 9]) & v;
		s[ 9] ^= (~v) & w;
        v = s[10]; w = s[11];
		s[10] ^= (~w) & s[12];
		s[11] ^= (~s[12]) & s[13];
		s[12] ^= (~s[13]) & s[14];
		s[13] ^= (~s[14]) & v;
		s[14] ^= (~v) & w;
        v = s[15]; w = s[16];
		s[15] ^= (~w) & s[17];
		s[16] ^= (~s[17]) & s[18];
		s[17] ^= (~s[18]) & s[19];
		s[18] ^= (~s[19]) & v;
		s[19] ^= (~v) & w;
        v = s[20]; w = s[21];
		s[20] ^= (~w) & s[22];
		s[21] ^= (~s[22]) & s[23];
		s[22] ^= (~s[23]) & s[24];
		s[23] ^= (~s[24]) & v;
        s[24] ^= (~v) & w;
		
        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }
}

static __device__ __forceinline__ void keccak_blockv35(uint2 *s, const uint64_t *keccak_round_constants) {
	size_t i;
	uint2 t[5], u[5], v, w;


	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(keccak_round_constants[i]);
	}
}


static __forceinline__ void keccak_block_host(uint64_t *s, const uint64_t *keccak_round_constants) {
    size_t i;
    uint64_t t[5], u[5], v, w;

    /* absorb input */    
    
    for (i = 0; i < 24; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        u[0] = t[4] ^ ROTL64(t[1], 1);
        u[1] = t[0] ^ ROTL64(t[2], 1);
        u[2] = t[1] ^ ROTL64(t[3], 1);
        u[3] = t[2] ^ ROTL64(t[4], 1);
        u[4] = t[3] ^ ROTL64(t[0], 1);

        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
        s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
        s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
        s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
        s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
        s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

        /* rho pi: b[..] = rotl(a[..], ..) */
        v = s[ 1];
        s[ 1] = ROTL64(s[ 6], 44);
        s[ 6] = ROTL64(s[ 9], 20);
        s[ 9] = ROTL64(s[22], 61);
        s[22] = ROTL64(s[14], 39);
        s[14] = ROTL64(s[20], 18);
        s[20] = ROTL64(s[ 2], 62);
        s[ 2] = ROTL64(s[12], 43);
        s[12] = ROTL64(s[13], 25);
        s[13] = ROTL64(s[19],  8);
        s[19] = ROTL64(s[23], 56);
        s[23] = ROTL64(s[15], 41);
        s[15] = ROTL64(s[ 4], 27);
        s[ 4] = ROTL64(s[24], 14);
        s[24] = ROTL64(s[21],  2);
        s[21] = ROTL64(s[ 8], 55);
        s[ 8] = ROTL64(s[16], 45);
        s[16] = ROTL64(s[ 5], 36);
        s[ 5] = ROTL64(s[ 3], 28);
        s[ 3] = ROTL64(s[18], 21);
        s[18] = ROTL64(s[17], 15);
        s[17] = ROTL64(s[11], 10);
        s[11] = ROTL64(s[ 7],  6);
        s[ 7] = ROTL64(s[10],  3);
        s[10] = ROTL64(    v,  1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
        v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
        v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
        v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
        v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }
}



 __constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)



__global__ void  m7_keccak512_gpu_hash_120(int threads, uint32_t startNounce, uint64_t *outputHash)
{

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        
		uint32_t nounce = startNounce + thread;

         uint64_t state[25];

        #pragma unroll 16
		 for (int i=9;i<25;i++) {state[i]=stateo[i];}

		state[0] = xor1(stateo[0],c_PaddedMessage80[9]);
		state[1] = xor1(stateo[1],c_PaddedMessage80[10]);
		state[2] = xor1(stateo[2],c_PaddedMessage80[11]);
		state[3] = xor1(stateo[3],c_PaddedMessage80[12]);
		state[4] = xor1(stateo[4],c_PaddedMessage80[13]);
		state[5] = xor1(stateo[5],REPLACE_HIWORD(c_PaddedMessage80[14],nounce));
		state[6] = xor1(stateo[6],c_PaddedMessage80[15]);
		state[7] = stateo[7];
		state[8] = xor1(stateo[8],0x8000000000000000);
		 
		keccak_block(state,RC);

#pragma unroll 8 
for (int i=0;i<8;i++) {outputHash[i*threads+thread]=state[i];}


	} //thread
}

__global__ void  __launch_bounds__(256, 3) m7_keccak512_gpu_hash_120_v35(int threads, uint32_t startNounce, uint64_t *outputHash)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{

		uint32_t nounce = startNounce + thread;

		uint2 state[25];

#pragma unroll 25
		for (int i = 0; i<25; i++) { state[i] = vectorize(stateo[i]); }

		state[0] ^= vectorize(c_PaddedMessage80[9]);
		state[1] ^= vectorize(c_PaddedMessage80[10]);
		state[2] ^= vectorize(c_PaddedMessage80[11]);
		state[3] ^= vectorize(c_PaddedMessage80[12]);
		state[4] ^= vectorize(c_PaddedMessage80[13]);
		state[5] ^= make_uint2(((uint32_t*)c_PaddedMessage80)[28],nounce);
		state[6] ^= vectorize(c_PaddedMessage80[15]);
		
		state[8] ^= make_uint2(0,0x80000000);

		keccak_blockv35(state, RC);

#pragma unroll 8 
		for (int i = 0; i<8; i++) { outputHash[i*threads + thread] = devectorize(state[i]); }


	} //thread
}


void m7_keccak512_cpu_init(int thr_id, int threads)
{
    	
	cudaMemcpyToSymbol( RC,cpu_RC,sizeof(cpu_RC),0,cudaMemcpyHostToDevice);	
} 

__host__ void m7_keccak512_setBlock_120(void *pdata)
{

	unsigned char PaddedMessage[128];
	uint8_t ending =0x01;
	memcpy(PaddedMessage, pdata, 122);
	memset(PaddedMessage+122,ending,1); 
	memset(PaddedMessage+123, 0, 5); 
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	uint64_t* alt_data = (uint64_t*) pdata;
         uint64_t state[25];
		 for(int i=0;i<25;i++) {state[i]=0;}


		for (int i=0;i<9;i++) {state[i]  ^= alt_data[i];}
		keccak_block_host(state,cpu_RC);

		cudaMemcpyToSymbol(stateo, state, 25*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);

}


__host__ void m7_keccak512_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order)
{
    const int threadsperblock = 256;

    dim3 grid(threads/threadsperblock);
    dim3 block(threadsperblock);

    size_t shared_size = 0;
	if (compute_version[thr_id]<35) {
    m7_keccak512_gpu_hash_120<<<grid, block, shared_size>>>(threads, startNounce, d_hash);
	}
	else {
	m7_keccak512_gpu_hash_120_v35 << <grid, block, shared_size >> >(threads, startNounce, d_hash);
	}

    MyStreamSynchronize(NULL, order, thr_id);
}

