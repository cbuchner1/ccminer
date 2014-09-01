#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h> 
#include <stdint.h>
#include <memory.h>

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

#define SPH_C64(x)    ((uint64_t)(x ## ULL))
#define SPH_C32(x)    ((uint32_t)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))
#define ROTR    SPH_ROTR32
#include "cuda_helper.h"
#define host_swab32(x)        ( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

 __constant__ uint32_t c_PaddedMessage80[32]; // padded message (80 bytes + padding)
__constant__ uint64_t pTarget[4];
__constant__ uint32_t pbuf[8];
uint32_t *d_mnounce[8];
uint32_t *d_MNonce[8];


static __constant__ uint32_t H256[8];
static __constant__ uint32_t K[64];
// muss expandiert werden
__constant__ uint32_t sha256_gpu_blockHeader[16]; // 2x512 Bit Message
__constant__ uint32_t sha256_gpu_register[8];


static const uint32_t cpu_H256[8] = {
	SPH_C32(0x6A09E667), SPH_C32(0xBB67AE85), SPH_C32(0x3C6EF372),
	SPH_C32(0xA54FF53A), SPH_C32(0x510E527F), SPH_C32(0x9B05688C),
	SPH_C32(0x1F83D9AB), SPH_C32(0x5BE0CD19)
};
static const uint32_t cpu_K[64] = {
	SPH_C32(0x428A2F98), SPH_C32(0x71374491),
	SPH_C32(0xB5C0FBCF), SPH_C32(0xE9B5DBA5),
	SPH_C32(0x3956C25B), SPH_C32(0x59F111F1),
	SPH_C32(0x923F82A4), SPH_C32(0xAB1C5ED5),
	SPH_C32(0xD807AA98), SPH_C32(0x12835B01),
	SPH_C32(0x243185BE), SPH_C32(0x550C7DC3),
	SPH_C32(0x72BE5D74), SPH_C32(0x80DEB1FE),
	SPH_C32(0x9BDC06A7), SPH_C32(0xC19BF174),
	SPH_C32(0xE49B69C1), SPH_C32(0xEFBE4786),
	SPH_C32(0x0FC19DC6), SPH_C32(0x240CA1CC),
	SPH_C32(0x2DE92C6F), SPH_C32(0x4A7484AA),
	SPH_C32(0x5CB0A9DC), SPH_C32(0x76F988DA),
	SPH_C32(0x983E5152), SPH_C32(0xA831C66D),
	SPH_C32(0xB00327C8), SPH_C32(0xBF597FC7),
	SPH_C32(0xC6E00BF3), SPH_C32(0xD5A79147),
	SPH_C32(0x06CA6351), SPH_C32(0x14292967),
	SPH_C32(0x27B70A85), SPH_C32(0x2E1B2138),
	SPH_C32(0x4D2C6DFC), SPH_C32(0x53380D13),
	SPH_C32(0x650A7354), SPH_C32(0x766A0ABB),
	SPH_C32(0x81C2C92E), SPH_C32(0x92722C85),
	SPH_C32(0xA2BFE8A1), SPH_C32(0xA81A664B),
	SPH_C32(0xC24B8B70), SPH_C32(0xC76C51A3),
	SPH_C32(0xD192E819), SPH_C32(0xD6990624),
	SPH_C32(0xF40E3585), SPH_C32(0x106AA070),
	SPH_C32(0x19A4C116), SPH_C32(0x1E376C08),
	SPH_C32(0x2748774C), SPH_C32(0x34B0BCB5),
	SPH_C32(0x391C0CB3), SPH_C32(0x4ED8AA4A),
	SPH_C32(0x5B9CCA4F), SPH_C32(0x682E6FF3),
	SPH_C32(0x748F82EE), SPH_C32(0x78A5636F),
	SPH_C32(0x84C87814), SPH_C32(0x8CC70208),
	SPH_C32(0x90BEFFFA), SPH_C32(0xA4506CEB),
	SPH_C32(0xBEF9A3F7), SPH_C32(0xC67178F2)
};


static __device__ __forceinline__ uint32_t bsg2_0(uint32_t x)
{
	uint32_t r1 = SPH_ROTR32(x,2);
	uint32_t r2 = SPH_ROTR32(x,13);
	uint32_t r3 = SPH_ROTR32(x,22);
	return xor3b(r1,r2,r3); 
}
static __device__ __forceinline__ uint32_t bsg2_1(uint32_t x)
{
	uint32_t r1 = SPH_ROTR32(x,6);
	uint32_t r2 = SPH_ROTR32(x,11);
	uint32_t r3 = SPH_ROTR32(x,25);
	return xor3b(r1,r2,r3);
}
static __device__ __forceinline__ uint32_t ssg2_0(uint32_t x)
{
	uint64_t r1 = SPH_ROTR32(x,7);
	uint64_t r2 = SPH_ROTR32(x,18);
	uint64_t r3 = shr_t32(x,3);
	return xor3b(r1,r2,r3);
}
static __device__ __forceinline__ uint32_t ssg2_1(uint32_t x)
{
	uint64_t r1 = SPH_ROTR32(x,17);
	uint64_t r2 = SPH_ROTR32(x,19);
	uint64_t r3 = shr_t32(x,10);
	return xor3b(r1,r2,r3);
}

static __device__ __forceinline__ void sha2_step1(uint32_t a,uint32_t b,uint32_t c,uint32_t &d,uint32_t e,uint32_t f,uint32_t g,uint32_t &h,
	                                              uint32_t in,const uint32_t Kshared)
{
uint32_t t1,t2;
uint32_t vxandx = xandx(e, f, g);
uint32_t bsg21 =bsg2_1(e);
uint32_t bsg20 =bsg2_0(a);
uint32_t andorv =andor32(a,b,c);

t1 = h + bsg21 + vxandx + Kshared + in; 
t2 = bsg20 + andorv; 
d = d + t1; 
h = t1 + t2; 
}

static __forceinline__ void sha2_step1_host(uint32_t a,uint32_t b,uint32_t c,uint32_t &d,uint32_t e,uint32_t f,uint32_t g,uint32_t &h,
	                                              uint32_t in,const uint32_t Kshared)
{



uint32_t t1,t2;
uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
uint32_t bsg21 =ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
uint32_t bsg20 =ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
uint32_t andorv =((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

t1 = h + bsg21 + vxandx + Kshared + in; 
t2 = bsg20 + andorv; 
d = d + t1; 
h = t1 + t2; 
}

static __device__ __forceinline__ void sha2_step2(uint32_t a,uint32_t b,uint32_t c,uint32_t &d,uint32_t e,uint32_t f,uint32_t g,uint32_t &h,
	                                              uint32_t* in,uint32_t pc,const uint32_t Kshared)
{
uint32_t t1,t2;

int pcidx1 = (pc-2) & 0xF;
int pcidx2 = (pc-7) & 0xF;
int pcidx3 = (pc-15) & 0xF;
uint32_t inx0 = in[pc];
uint32_t inx1 = in[pcidx1];
uint32_t inx2 = in[pcidx2];
uint32_t inx3 = in[pcidx3];


uint32_t ssg21 = ssg2_1(inx1);
uint32_t ssg20 = ssg2_0(inx3);
uint32_t vxandx = xandx(e, f, g);
uint32_t bsg21 =bsg2_1(e);
uint32_t bsg20 =bsg2_0(a);
uint32_t andorv =andor32(a,b,c);

in[pc] = ssg21+inx2+ssg20+inx0;

t1 = h + bsg21 + vxandx + Kshared + in[pc]; 
t2 = bsg20 + andorv; 
d =  d + t1; 
h = t1 + t2; 

}

static __forceinline__ void sha2_step2_host(uint32_t a,uint32_t b,uint32_t c,uint32_t &d,uint32_t e,uint32_t f,uint32_t g,uint32_t &h,
	                                              uint32_t* in,uint32_t pc,const uint32_t Kshared)
{
uint32_t t1,t2;

int pcidx1 = (pc-2) & 0xF;
int pcidx2 = (pc-7) & 0xF;
int pcidx3 = (pc-15) & 0xF;
uint32_t inx0 = in[pc];
uint32_t inx1 = in[pcidx1];
uint32_t inx2 = in[pcidx2];
uint32_t inx3 = in[pcidx3];


uint32_t ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
uint32_t ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
uint32_t bsg21 =ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
uint32_t bsg20 =ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
uint32_t andorv =((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

in[pc] = ssg21+inx2+ssg20+inx0;

t1 = h + bsg21 + vxandx + Kshared + in[pc]; 
t2 = bsg20 + andorv; 
d =  d + t1; 
h = t1 + t2; 

}


static __device__ __forceinline__ void sha2_round_body(uint32_t* in, uint32_t* r,const uint32_t* Kshared)
{
		
		
		uint32_t a=r[0];
        uint32_t b=r[1];
        uint32_t c=r[2];
        uint32_t d=r[3];
        uint32_t e=r[4];
        uint32_t f=r[5];
        uint32_t g=r[6];
        uint32_t h=r[7];
			
		sha2_step1(a,b,c,d,e,f,g,h,in[0],Kshared[0]);
		sha2_step1(h,a,b,c,d,e,f,g,in[1],Kshared[1]);
		sha2_step1(g,h,a,b,c,d,e,f,in[2],Kshared[2]);
		sha2_step1(f,g,h,a,b,c,d,e,in[3],Kshared[3]);
		sha2_step1(e,f,g,h,a,b,c,d,in[4],Kshared[4]);
		sha2_step1(d,e,f,g,h,a,b,c,in[5],Kshared[5]);
		sha2_step1(c,d,e,f,g,h,a,b,in[6],Kshared[6]);
		sha2_step1(b,c,d,e,f,g,h,a,in[7],Kshared[7]);
		sha2_step1(a,b,c,d,e,f,g,h,in[8],Kshared[8]);
		sha2_step1(h,a,b,c,d,e,f,g,in[9],Kshared[9]);
		sha2_step1(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
		sha2_step1(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
		sha2_step1(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
		sha2_step1(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
		sha2_step1(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
		sha2_step1(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

#pragma unroll 3
		for (int i=0;i<3;i++) {

		sha2_step2(a,b,c,d,e,f,g,h,in,0,Kshared[16+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,1,Kshared[17+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,2,Kshared[18+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,3,Kshared[19+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,4,Kshared[20+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,5,Kshared[21+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,6,Kshared[22+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,7,Kshared[23+16*i]);
		sha2_step2(a,b,c,d,e,f,g,h,in,8,Kshared[24+16*i]);
		sha2_step2(h,a,b,c,d,e,f,g,in,9,Kshared[25+16*i]);
		sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);

		}
		
		

		 r[0] = r[0] + a;
		 r[1] = r[1] + b;
		 r[2] = r[2] + c;
		 r[3] = r[3] + d;
		 r[4] = r[4] + e;
		 r[5] = r[5] + f;
		 r[6] = r[6] + g;
		 r[7] = r[7] + h;
}

static __forceinline__ void sha2_round_body_host(uint32_t* in, uint32_t* r,const uint32_t* Kshared)
{
		
		
		uint32_t a=r[0];
        uint32_t b=r[1];
        uint32_t c=r[2];
        uint32_t d=r[3];
        uint32_t e=r[4];
        uint32_t f=r[5];
        uint32_t g=r[6];
        uint32_t h=r[7];
			
		sha2_step1_host(a,b,c,d,e,f,g,h,in[0],Kshared[0]);
		sha2_step1_host(h,a,b,c,d,e,f,g,in[1],Kshared[1]);
		sha2_step1_host(g,h,a,b,c,d,e,f,in[2],Kshared[2]);
		sha2_step1_host(f,g,h,a,b,c,d,e,in[3],Kshared[3]);
		sha2_step1_host(e,f,g,h,a,b,c,d,in[4],Kshared[4]);
		sha2_step1_host(d,e,f,g,h,a,b,c,in[5],Kshared[5]);
		sha2_step1_host(c,d,e,f,g,h,a,b,in[6],Kshared[6]);
		sha2_step1_host(b,c,d,e,f,g,h,a,in[7],Kshared[7]);
		sha2_step1_host(a,b,c,d,e,f,g,h,in[8],Kshared[8]);
		sha2_step1_host(h,a,b,c,d,e,f,g,in[9],Kshared[9]);
		sha2_step1_host(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
		sha2_step1_host(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
		sha2_step1_host(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
		sha2_step1_host(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
		sha2_step1_host(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
		sha2_step1_host(b,c,d,e,f,g,h,a,in[15],Kshared[15]);


		for (int i=0;i<3;i++) {

		sha2_step2_host(a,b,c,d,e,f,g,h,in,0,Kshared[16+16*i]);
		sha2_step2_host(h,a,b,c,d,e,f,g,in,1,Kshared[17+16*i]);
		sha2_step2_host(g,h,a,b,c,d,e,f,in,2,Kshared[18+16*i]);
		sha2_step2_host(f,g,h,a,b,c,d,e,in,3,Kshared[19+16*i]);
		sha2_step2_host(e,f,g,h,a,b,c,d,in,4,Kshared[20+16*i]);
		sha2_step2_host(d,e,f,g,h,a,b,c,in,5,Kshared[21+16*i]);
		sha2_step2_host(c,d,e,f,g,h,a,b,in,6,Kshared[22+16*i]);
		sha2_step2_host(b,c,d,e,f,g,h,a,in,7,Kshared[23+16*i]);
		sha2_step2_host(a,b,c,d,e,f,g,h,in,8,Kshared[24+16*i]);
		sha2_step2_host(h,a,b,c,d,e,f,g,in,9,Kshared[25+16*i]);
		sha2_step2_host(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha2_step2_host(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha2_step2_host(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha2_step2_host(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha2_step2_host(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha2_step2_host(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);

		}

		 r[0] = r[0] + a;
		 r[1] = r[1] + b;
		 r[2] = r[2] + c;
		 r[3] = r[3] + d;
		 r[4] = r[4] + e;
		 r[5] = r[5] + f;
		 r[6] = r[6] + g;
		 r[7] = r[7] + h;
}


__global__ void __launch_bounds__(512,1) m7_sha256_gpu_hash_120(int threads, uint32_t startNounce, uint64_t *outputHash)
{

   
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {

		uint32_t nounce = startNounce +  thread ; // original implementation

        uint32_t buf[8];
		uint32_t in2[16]={0};
		uint32_t in3[16]={0};

        #pragma unroll 13
		for (int i=0;i<13;i++) {in2[i]= cuda_swab32(c_PaddedMessage80[i+16]);}
		in2[13]=cuda_swab32(nounce);
		in2[14]=cuda_swab32(c_PaddedMessage80[30]);

		                        in3[15]=0x3d0;
          
        #pragma unroll 8
		for (int i=0;i<8;i++) {buf[i]= pbuf[i];}    

                    sha2_round_body(in2,buf,K);
					sha2_round_body(in3,buf,K);

#pragma unroll 4
for (int i=0;i<4;i++) {outputHash[i*threads+thread]=cuda_swab32ll(((uint64_t*)buf)[i]);}


//////////////////////////////////////////////////////////////////////////////////////////////////	  
	} // threads

}


__global__ void  m7_sha256_gpu_hash_300(int threads, uint32_t startNounce, uint64_t *g_hash1, uint64_t *g_nonceVector, uint32_t *resNounce)
{
/*	
	__shared__ uint32_t Kshared[64];
	if (threadIdx.x < 64) {
		Kshared[threadIdx.x]=K[threadIdx.x];
	}
	__syncthreads();
*/
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {

        
     
		
union {
uint8_t h1[304];
uint32_t h4[76];
uint64_t h8[38];
} hash;  


        uint32_t in[16],buf[8];

		
		#pragma unroll 8
		for (int i=0;i<8;i++) {((uint64_t*)in)[i]= cuda_swab32ll(g_hash1[threads*i+thread]);}
        #pragma unroll 8
		for (int i=0;i<8;i++) {buf[i] = H256[i];}    

		sha2_round_body(in,buf,K);

		#pragma unroll 8
		for (int i=0;i<8;i++) {((uint64_t*)in)[i]= cuda_swab32ll(g_hash1[threads*(i+8)+thread]);}
		sha2_round_body(in,buf,K);

		#pragma unroll 8
		for (int i=0;i<8;i++) {((uint64_t*)in)[i]= cuda_swab32ll(g_hash1[threads*(i+16)+thread]);}
		sha2_round_body(in,buf,K);

		#pragma unroll 8
		for (int i=0;i<8;i++) {((uint64_t*)in)[i]= cuda_swab32ll(g_hash1[threads*(i+24)+thread]);}
		sha2_round_body(in,buf,K);

		#pragma unroll 5
		for (int i=0;i<5;i++) {((uint64_t*)in)[i]= cuda_swab32ll(g_hash1[threads*(i+32)+thread]);}
		((uint64_t*)in)[5]= g_hash1[threads*(5+32)+thread];
		in[11]=0;
		in[12]=0;
		in[13]=0;
		in[14]=0;


                   in[15]=0x968;
				   
                   int it=0;				
				   do {
                   in[15]-=8;
				   it++;
				   }  while (((uint8_t*)in)[44-it]==0);
				   ((uint8_t*)in)[44-it+1]=0x80;
		
           ((uint64_t*)in)[5]= cuda_swab32ll(((uint64_t*)in)[5]);

				   sha2_round_body(in,buf,K);

uint32_t nounce = startNounce +thread;
		bool rc = true;


    if (cuda_swab32ll(((uint64_t*)buf)[3]) > pTarget[3]) {rc = false;} 
//// only needed for solo mining, commenting it out will probably increased rejected block (no big deal actually)
	/*
	else if (cuda_swab32ll(((uint64_t*)buf)[3]) == pTarget[3]) {  // in case ptarget=buf=0
		          if (cuda_swab32ll(((uint64_t*)buf)[2]) > pTarget[2]) {rc = false;} 
	         else if (cuda_swab32ll(((uint64_t*)buf)[2]) == pTarget[2]) {
				         if (cuda_swab32ll(((uint64_t*)buf)[1]) > pTarget[1]) {rc = false;} 
	                     else if (cuda_swab32ll(((uint64_t*)buf)[1]) == pTarget[1]) {
				                  if (cuda_swab32ll(((uint64_t*)buf)[0]) > pTarget[0]) {rc = false;} 
								  else if (cuda_swab32ll(((uint64_t*)buf)[0]) == pTarget[0]) {rc = true;}
						 }}}
      */      
	
	

		if(rc == true)
		{
			if(resNounce[0] > nounce)
				resNounce[0] = nounce;

		}


////
	} // threads
}



__host__ void m7_sha256_cpu_init(int thr_id, int threads)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol(	H256,cpu_H256,sizeof(cpu_H256),0, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol(	K,cpu_K,sizeof(cpu_K),0, cudaMemcpyHostToDevice );
	cudaMalloc(&d_MNonce[thr_id], sizeof(uint32_t)); 
	cudaMallocHost(&d_mnounce[thr_id], 1*sizeof(uint32_t));
}


__host__  uint32_t m7_sha256_cpu_hash_300(int thr_id, int threads, uint32_t startNounce, uint64_t *d_nonceVector,uint64_t *d_hash, int order)
{
	
	uint32_t result = 0xffffffff;
	cudaMemset(d_MNonce[thr_id], 0xff, sizeof(uint32_t));
	//const int threadsperblock = 384; // Alignment mit mixtob Grösse. NICHT ÄNDERN
	const int threadsperblock = 512;
	
	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;
	
	
	m7_sha256_gpu_hash_300<<<grid, block, shared_size>>>(threads, startNounce, d_hash, d_nonceVector, d_MNonce[thr_id]);
	cudaMemcpy(d_mnounce[thr_id], d_MNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	MyStreamSynchronize(NULL, order, thr_id);
	result = *d_mnounce[thr_id];
	return result;
}


__host__ void m7_sha256_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{

	const int threadsperblock = 512; // Alignment mit mixtob Grösse. NICHT ÄNDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock); 
//	dim3 grid(1);
//	dim3 block(1);
	size_t shared_size = 0;
	
	m7_sha256_gpu_hash_120<<<grid, block, shared_size>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}

__host__ void m7_sha256_setBlock_120(void *pdata,const void *ptarget)  //not useful
{
	unsigned char PaddedMessage[128];
	uint8_t ending =0x80;
	memcpy(PaddedMessage, pdata, 122);
	memset(PaddedMessage+122,ending,1); 
	memset(PaddedMessage+123, 0, 5); //useless
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( pTarget, ptarget, 4*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	/// do first loop here... ///
    
	uint32_t * alt_data = (uint32_t*) PaddedMessage; 
	uint32_t in[16],buf[8];
	for (int i=0;i<16;i++) {in[i]= host_swab32(alt_data[i]);}
	for (int i=0;i<8;i++) {buf[i]= cpu_H256[i];}     
			                sha2_round_body_host(in,buf,cpu_K);
    cudaMemcpyToSymbol( pbuf, buf, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}
