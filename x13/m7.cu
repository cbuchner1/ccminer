/*
 * m7 algorithm 
 * 
 */

extern "C"
{
#include "sph/sph_sha2.h"
#include "sph/sph_keccak.h"
#include "sph/sph_ripemd.h"
#include "sph/sph_haval.h"
#include "sph/sph_tiger.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_blake.h"
#include "miner.h"
}
//#include "mpir.h"

extern int device_map[8];


static uint64_t *d_hash[8];
static uint64_t *KeccakH[8];
static uint64_t *Sha512H[8];
static uint64_t *d_prod0[8];
static uint64_t *d_prod1[8];

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
/*
static void mpz_set_uint256(mpz_t r, uint8_t *u)
{
    mpz_import(r, 32 / sizeof(unsigned long), -1, sizeof(unsigned long), -1, 0, u);
}

static void mpz_get_uint256(mpz_t r, uint8_t *u)
{
    u=0;
    mpz_export(u, 0, -1, sizeof(unsigned long), -1, 0, r);
}

static void mpz_set_uint512(mpz_t r, uint8_t *u)
{
    mpz_import(r, 64 / sizeof(unsigned long), -1, sizeof(unsigned long), -1, 0, u);
}

static void set_one_if_zero(uint8_t *hash512) {
    for (int i = 0; i < 32; i++) {
        if (hash512[i] != 0) {
            return;
        }
    }
    hash512[0] = 1;
}
*/
//extern uint32_t m7_sha256_cpu_hash_300(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern uint32_t m7_sha256_cpu_hash_300(int thr_id, int threads, uint32_t startNounce, uint64_t *d_nonceVector, uint64_t *d_hash, int order);

extern void m7_sha256_setBlock_120(void *data,const void *ptarget);
extern void m7_sha256_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern void m7_sha256_cpu_init(int thr_id, int threads);


extern void sha512_cpu_init(int thr_id, int threads);
extern void sha512_setBlock_120(void *pdata);
extern void m7_sha512_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void ripemd160_cpu_init(int thr_id, int threads);
extern void ripemd160_setBlock_120(void *pdata);
extern void m7_ripemd160_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void tiger192_cpu_init(int thr_id, int threads);
extern void tiger192_setBlock_120(void *pdata);
extern void m7_tiger192_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);


extern void m7_bigmul_init(int thr_id, int threads);
extern void m7_bigmul_unroll1_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order);
extern void m7_bigmul_unroll2_cpu(int thr_id, int threads,uint64_t* Hash1, uint64_t* Hash2,uint64_t *finalHash,int order);

extern void cpu_mul(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p, int order);
extern void cpu_mulT4(int thr_id, int threads, uint32_t alegs, uint32_t blegs, uint64_t *g_a, uint64_t *g_b, uint64_t *g_p, int order);
extern void mul_init();

	
extern void m7_keccak512_setBlock_120(void *pdata);
extern void m7_keccak512_cpu_hash(int thr_id, int threads, uint32_t startNounce, uint64_t *d_hash, int order);
extern void m7_keccak512_cpu_init(int thr_id, int threads);

extern void whirlpool512_cpu_init(int thr_id, int threads, int flag);
extern void whirlpool512_setBlock_120(void *pdata);
extern void m7_whirlpool512_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);

extern void haval256_cpu_init(int thr_id, int threads);
extern void haval256_setBlock_120(void *data);
extern void m7_haval256_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order);


extern void quark_check_cpu_init(int thr_id, int threads);
extern void quark_check_cpu_setTarget(const void *ptarget);
extern uint32_t quark_check_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint64_t *d_inputHash, int order);



// m7 Hashfunktion
/*
inline void m7_hash(void *state, const void *input,uint32_t TheNonce, int debug)
{
    // sha256(sha256*sha512*keccak512*ripemd160*haval*tiger1*whirlpool) good luck with that...
	
	char data_str[245], hash_str[65], target_str[65];
    uint8_t *bdata = 0;
    mpz_t bns[7];
    mpz_t product;
    int rc = 0;
	
    for(int i=0; i < 7; i++){
        mpz_init(bns[i]);
    }
    mpz_init(product);
	 

	uint32_t data[32] ; 
	uint32_t *data_p64 = data + (116 / sizeof(data[0]));
	uint8_t bhash[7][64];
	uint32_t hash[8];
	memcpy(data,input,122);


	int M7_MIDSTATE_LEN = 116;
	for(int i=0; i < 7; i++){
        mpz_init(bns[i]);
    }

    sph_sha256_context ctx_final_sha256;

    sph_sha256_context ctx_sha256;
    sph_sha512_context ctx_sha512;
    sph_keccak512_context ctx_keccak;
    sph_whirlpool_context ctx_whirlpool;
    sph_haval256_5_context ctx_haval;
    sph_tiger_context ctx_tiger;
    sph_ripemd160_context ctx_ripemd;

    sph_sha256_init(&ctx_sha256);
    sph_sha256 (&ctx_sha256, data, M7_MIDSTATE_LEN);
    
    sph_sha512_init(&ctx_sha512);
    sph_sha512 (&ctx_sha512, data, M7_MIDSTATE_LEN);
    
    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, data, M7_MIDSTATE_LEN);

    sph_whirlpool_init(&ctx_whirlpool);
    sph_whirlpool (&ctx_whirlpool, data, M7_MIDSTATE_LEN);
    
    sph_haval256_5_init(&ctx_haval);
    sph_haval256_5 (&ctx_haval, data, M7_MIDSTATE_LEN);

    sph_tiger_init(&ctx_tiger);
    sph_tiger (&ctx_tiger, data, M7_MIDSTATE_LEN);

    sph_ripemd160_init(&ctx_ripemd);
    sph_ripemd160 (&ctx_ripemd, data, M7_MIDSTATE_LEN);

    sph_sha256_context ctx2_sha256;
    sph_sha512_context ctx2_sha512; 
    sph_keccak512_context ctx2_keccak;
    sph_whirlpool_context ctx2_whirlpool;
    sph_haval256_5_context ctx2_haval;
    sph_tiger_context ctx2_tiger;
    sph_ripemd160_context ctx2_ripemd;

        data[29] = TheNonce;

        memset(bhash, 0, 7 * 64);

        ctx2_sha256 = ctx_sha256;
        sph_sha256 (&ctx2_sha256, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_sha256_close(&ctx2_sha256, (void*)(bhash[0]));

        ctx2_sha512 = ctx_sha512;
        sph_sha512 (&ctx2_sha512, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_sha512_close(&ctx2_sha512, (void*)(bhash[1]));
        
        ctx2_keccak = ctx_keccak;
        sph_keccak512 (&ctx2_keccak, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_keccak512_close(&ctx2_keccak, (void*)(bhash[2]));

        ctx2_whirlpool = ctx_whirlpool;
        sph_whirlpool (&ctx2_whirlpool, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_whirlpool_close(&ctx2_whirlpool, (void*)(bhash[3]));
        
        ctx2_haval = ctx_haval;
        sph_haval256_5 (&ctx2_haval, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_haval256_5_close(&ctx2_haval, (void*)(bhash[4]));

        ctx2_tiger = ctx_tiger;
        sph_tiger (&ctx2_tiger, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_tiger_close(&ctx2_tiger, (void*)(bhash[5]));

        ctx2_ripemd = ctx_ripemd;
        sph_ripemd160 (&ctx2_ripemd, data_p64, 122 - M7_MIDSTATE_LEN);
        sph_ripemd160_close(&ctx2_ripemd, (void*)(bhash[6]));
if (debug == 1) {
		for (int i=0;i<16;i++) {applog(LOG_INFO,"sha256[%d]=%02x %02x %02x %02x sha512[%d]=%02x %02x %02x %02x keccak[%d]=%02x %02x %02x %02x whirlpool[2][%d]=%02x %02x %02x %02x haval[%d]=%02x %02x %02x %02x tiger[%d]=%02x %02x %02x %02x ripemd[%d]=%02x %02x %02x %02x\n",
        i,bhash[0][4*i+3],bhash[0][4*i+2],bhash[0][4*i+1],bhash[0][4*i+0],
        i,bhash[1][4*i+3],bhash[1][4*i+2],bhash[1][4*i+1],bhash[1][4*i+0],
		i,bhash[2][4*i+3],bhash[2][4*i+2],bhash[2][4*i+1],bhash[2][4*i+0],
		i,bhash[3][4*i+3],bhash[3][4*i+2],bhash[3][4*i+1],bhash[3][4*i+0],
		i,bhash[4][4*i+3],bhash[4][4*i+2],bhash[4][4*i+1],bhash[4][4*i+0],
		i,bhash[5][4*i+3],bhash[5][4*i+2],bhash[5][4*i+1],bhash[5][4*i+0],
		i,bhash[6][4*i+3],bhash[6][4*i+2],bhash[6][4*i+1],bhash[6][4*i+0]
	);}
}
        for(int i=0; i < 7; i++){
            set_one_if_zero(bhash[i]);
            mpz_set_uint512(bns[i],bhash[i]);
        }
        
        for(int i=6; i > 0; i--){
            mpz_mul(bns[i-1], bns[i-1], bns[i]);
        }

        int bytes = mpz_sizeinbase(bns[0], 256);
        bdata = (uint8_t *)realloc(bdata, bytes);
        mpz_export((void *)bdata, NULL, -1, 1, 0, 0, bns[0]);
       sph_sha256_init(&ctx_final_sha256);
        sph_sha256 (&ctx_final_sha256, bdata, bytes);
        sph_sha256_close(&ctx_final_sha256, (void*)(hash));

    memcpy(state, hash, 32);
}
*/
extern float tp_coef[8];
extern bool opt_benchmark;


extern "C" int scanhash_m7(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long  *hashes_done)
{

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	
//	const int throughput = 256*256*16;
	const int throughput = 2560*512*1;

	const uint32_t FirstNonce = pdata[29];
 
	static bool init[8] = {0,0,0,0,0,0,0,0};
	if (!init[thr_id])
	{

		cudaSetDevice(device_map[thr_id]);
		cudaMalloc(&d_prod0[thr_id],      35 *sizeof(uint64_t) * throughput*tp_coef[thr_id]);
		cudaMalloc(&d_prod1[thr_id],      38 *sizeof(uint64_t) * throughput*tp_coef[thr_id]);
		cudaMalloc(&KeccakH[thr_id],     8 *sizeof(uint64_t) * throughput*tp_coef[thr_id]);
        cudaMalloc(&Sha512H[thr_id],     8 *sizeof(uint64_t) * throughput*tp_coef[thr_id]);

		   m7_sha256_cpu_init(thr_id, throughput*tp_coef[thr_id]);		
		      sha512_cpu_init(thr_id, throughput*tp_coef[thr_id]);
		m7_keccak512_cpu_init(thr_id, throughput*tp_coef[thr_id]);
		    haval256_cpu_init(thr_id, throughput*tp_coef[thr_id]);
            tiger192_cpu_init(thr_id, throughput*tp_coef[thr_id]);
		whirlpool512_cpu_init(thr_id, throughput*tp_coef[thr_id],0);	
		   ripemd160_cpu_init(thr_id, throughput*tp_coef[thr_id]);
		 quark_check_cpu_init(thr_id, throughput*tp_coef[thr_id]);
		       m7_bigmul_init(thr_id, throughput*tp_coef[thr_id]);
			   mul_init();
		init[thr_id] = true; 
	}
	
	const uint32_t Htarg = ptarget[7];

	whirlpool512_setBlock_120((void*)pdata);
  	   m7_sha256_setBlock_120((void*)pdata,ptarget);
	      sha512_setBlock_120((void*)pdata);
	    haval256_setBlock_120((void*)pdata);
	m7_keccak512_setBlock_120((void*)pdata);
	   ripemd160_setBlock_120((void*)pdata);
	    tiger192_setBlock_120((void*)pdata);
	quark_check_cpu_setTarget(ptarget);
	
	do {

		int order = 0;

          
		  m7_keccak512_cpu_hash(thr_id, throughput*tp_coef[thr_id], pdata[29], KeccakH[thr_id], order++);
         
		   m7_sha512_cpu_hash_120(thr_id, throughput*tp_coef[thr_id], pdata[29], Sha512H[thr_id], order++);

    cpu_mulT4(0, throughput*tp_coef[thr_id], 8, 8, Sha512H[thr_id], KeccakH[thr_id], d_prod0[thr_id],order); //64
	MyStreamSynchronize(0,order++,thr_id);

      m7_whirlpool512_cpu_hash_120(thr_id, throughput*tp_coef[thr_id], pdata[29], KeccakH[thr_id], order++);

	cpu_mulT4(0, throughput*tp_coef[thr_id],8, 16, KeccakH[thr_id], d_prod0[thr_id], d_prod1[thr_id],order); //128
	MyStreamSynchronize(0,order++,thr_id);

m7_sha256_cpu_hash_120(thr_id, throughput*tp_coef[thr_id], pdata[29], KeccakH[thr_id], order++);
cpu_mulT4(0, throughput*tp_coef[thr_id], 4, 24, KeccakH[thr_id], d_prod1[thr_id], d_prod0[thr_id],order); //96
	MyStreamSynchronize(0,order++,thr_id);

		   m7_haval256_cpu_hash_120(thr_id, throughput*tp_coef[thr_id], pdata[29], KeccakH[thr_id], order++);
cpu_mulT4(0, throughput*tp_coef[thr_id], 4, 28, KeccakH[thr_id], d_prod0[thr_id], d_prod1[thr_id],order);  //112
	MyStreamSynchronize(0,order++,thr_id);
		
		m7_tiger192_cpu_hash_120(thr_id, throughput*tp_coef[thr_id], pdata[29], KeccakH[thr_id], order++);
	m7_bigmul_unroll1_cpu(thr_id, throughput*tp_coef[thr_id], KeccakH[thr_id], d_prod1[thr_id], d_prod0[thr_id],order);
	MyStreamSynchronize(0,order++,thr_id);
		
		 m7_ripemd160_cpu_hash_120(thr_id, throughput*tp_coef[thr_id], pdata[29], KeccakH[thr_id], order++);

	m7_bigmul_unroll2_cpu(thr_id, throughput*tp_coef[thr_id], KeccakH[thr_id], d_prod0[thr_id], d_prod1[thr_id],order);
	MyStreamSynchronize(0,order++,thr_id);


uint32_t foundNonce = m7_sha256_cpu_hash_300(thr_id, throughput*tp_coef[thr_id], pdata[29], NULL, d_prod1[thr_id], order);
if  (foundNonce != 0xffffffff) {
			uint32_t vhash64[8];
//			m7_hash(vhash64, pdata,foundNonce,0);
			
//            if( (vhash64[7]<=Htarg )  ) {              
                pdata[29] = foundNonce;
				*hashes_done = foundNonce - FirstNonce + 1;
				return 1;
//			} else {
//				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU! vhash64 %08x and htarg %08x", thr_id, foundNonce,vhash64[7],Htarg);
//			m7_hash(vhash64, pdata,foundNonce,1);
//			} 
        } // foundNonce
		pdata[29] += throughput*tp_coef[thr_id];
*hashes_done +=throughput*tp_coef[thr_id];
	} while (pdata[29] < max_nonce && !work_restart[thr_id].restart);

//*hashes_done = pdata[29] - FirstNonce + 1;
	return 0;
}
