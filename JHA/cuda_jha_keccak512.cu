#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "miner.h"

// ZR5
__constant__ uint32_t d_OriginalData[20];

__constant__ uint32_t c_PaddedMessage[18];
__constant__ uint64_t c_State[25];

#define U32TO64_LE(p) \
	(((uint64_t)(*p)) | (((uint64_t)(*(p + 1))) << 32))

#define U64TO32_LE(p, v) \
	*p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);

static const uint64_t host_keccak_round_constants[24] = {
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

__constant__ uint64_t c_keccak_round_constants[24];

static __device__ __forceinline__ void
keccak_block(uint64_t *s, const uint32_t *in, const uint64_t *keccak_round_constants) {
	size_t i;
	uint64_t t[5], u[5], v, w;

	/* absorb input */
	#pragma unroll 9
	for (i = 0; i < 72 / 8; i++, in += 2)
		s[i] ^= U32TO64_LE(in);

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

// Setup-Funktionen
__host__
void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol( c_keccak_round_constants,
						host_keccak_round_constants,
						sizeof(host_keccak_round_constants),
						0, cudaMemcpyHostToDevice);
}

#define cKeccakB    1600
#define cKeccakR    576

#define cKeccakR_SizeInBytes    (cKeccakR / 8)
#define crypto_hash_BYTES 64

#if (cKeccakB == 1600)
	typedef unsigned long long UINT64;
	typedef UINT64 tKeccakLane;
	#define cKeccakNumberOfRounds 24
#endif

#define cKeccakLaneSizeInBits   (sizeof(tKeccakLane) * 8)

#define ROL(a, offset) ((((tKeccakLane)a) << ((offset) % cKeccakLaneSizeInBits)) ^ (((tKeccakLane)a) >> (cKeccakLaneSizeInBits-((offset) % cKeccakLaneSizeInBits))))
#if ((cKeccakB/25) == 8)
	#define ROL_mult8(a, offset) ((tKeccakLane)a)
#else
	#define ROL_mult8(a, offset) ROL(a, offset)
#endif

const tKeccakLane KeccakF_RoundConstants[cKeccakNumberOfRounds] = {
	(tKeccakLane)0x0000000000000001ULL,
	(tKeccakLane)0x0000000000008082ULL,
	(tKeccakLane)0x800000000000808aULL,
	(tKeccakLane)0x8000000080008000ULL,
	(tKeccakLane)0x000000000000808bULL,
	(tKeccakLane)0x0000000080000001ULL,
	(tKeccakLane)0x8000000080008081ULL,
	(tKeccakLane)0x8000000000008009ULL,
	(tKeccakLane)0x000000000000008aULL,
	(tKeccakLane)0x0000000000000088ULL,
	(tKeccakLane)0x0000000080008009ULL,
	(tKeccakLane)0x000000008000000aULL,
	(tKeccakLane)0x000000008000808bULL,
	(tKeccakLane)0x800000000000008bULL,
	(tKeccakLane)0x8000000000008089ULL,
	(tKeccakLane)0x8000000000008003ULL,
	(tKeccakLane)0x8000000000008002ULL,
	(tKeccakLane)0x8000000000000080ULL
#if (cKeccakB >= 400)
  , (tKeccakLane)0x000000000000800aULL,
	(tKeccakLane)0x800000008000000aULL
#if (cKeccakB >= 800)
  , (tKeccakLane)0x8000000080008081ULL,
	(tKeccakLane)0x8000000000008080ULL
#if (cKeccakB == 1600)
  , (tKeccakLane)0x0000000080000001ULL,
	(tKeccakLane)0x8000000080008008ULL
#endif
#endif
#endif
};

void KeccakF(tKeccakLane * state, const tKeccakLane *in, int laneCount)
{
	while ( --laneCount >= 0 ) {
		state[laneCount] ^= in[laneCount];
	}

	{
		tKeccakLane Aba, Abe, Abi, Abo, Abu;
		tKeccakLane Aga, Age, Agi, Ago, Agu;
		tKeccakLane Aka, Ake, Aki, Ako, Aku;
		tKeccakLane Ama, Ame, Ami, Amo, Amu;
		tKeccakLane Asa, Ase, Asi, Aso, Asu;
		tKeccakLane BCa, BCe, BCi, BCo, BCu;
		tKeccakLane Da, De, Di, Do, Du;
		tKeccakLane Eba, Ebe, Ebi, Ebo, Ebu;
		tKeccakLane Ega, Ege, Egi, Ego, Egu;
		tKeccakLane Eka, Eke, Eki, Eko, Eku;
		tKeccakLane Ema, Eme, Emi, Emo, Emu;
		tKeccakLane Esa, Ese, Esi, Eso, Esu;
		#define    round    laneCount

		//copyFromState(A, state)
		Aba = state[ 0];
		Abe = state[ 1];
		Abi = state[ 2];
		Abo = state[ 3];
		Abu = state[ 4];
		Aga = state[ 5];
		Age = state[ 6];
		Agi = state[ 7];
		Ago = state[ 8];
		Agu = state[ 9];
		Aka = state[10];
		Ake = state[11];
		Aki = state[12];
		Ako = state[13];
		Aku = state[14];
		Ama = state[15];
		Ame = state[16];
		Ami = state[17];
		Amo = state[18];
		Amu = state[19];
		Asa = state[20];
		Ase = state[21];
		Asi = state[22];
		Aso = state[23];
		Asu = state[24];

		for( round = 0; round < cKeccakNumberOfRounds; round += 2 )
		{
			//    prepareTheta
			BCa = Aba^Aga^Aka^Ama^Asa;
			BCe = Abe^Age^Ake^Ame^Ase;
			BCi = Abi^Agi^Aki^Ami^Asi;
			BCo = Abo^Ago^Ako^Amo^Aso;
			BCu = Abu^Agu^Aku^Amu^Asu;

			//thetaRhoPiChiIotaPrepareTheta(round  , A, E)
			Da = BCu^ROL(BCe, 1);
			De = BCa^ROL(BCi, 1);
			Di = BCe^ROL(BCo, 1);
			Do = BCi^ROL(BCu, 1);
			Du = BCo^ROL(BCa, 1);

			Aba ^= Da;
			BCa = Aba;
			Age ^= De;
			BCe = ROL(Age, 44);
			Aki ^= Di;
			BCi = ROL(Aki, 43);
			Amo ^= Do;
			BCo = ROL(Amo, 21);
			Asu ^= Du;
			BCu = ROL(Asu, 14);
			Eba =   BCa ^((~BCe)&  BCi );
			Eba ^= (tKeccakLane)KeccakF_RoundConstants[round];
			Ebe =   BCe ^((~BCi)&  BCo );
			Ebi =   BCi ^((~BCo)&  BCu );
			Ebo =   BCo ^((~BCu)&  BCa );
			Ebu =   BCu ^((~BCa)&  BCe );

			Abo ^= Do;
			BCa = ROL(Abo, 28);
			Agu ^= Du;
			BCe = ROL(Agu, 20);
			Aka ^= Da;
			BCi = ROL(Aka,  3);
			Ame ^= De;
			BCo = ROL(Ame, 45);
			Asi ^= Di;
			BCu = ROL(Asi, 61);
			Ega =   BCa ^((~BCe)&  BCi );
			Ege =   BCe ^((~BCi)&  BCo );
			Egi =   BCi ^((~BCo)&  BCu );
			Ego =   BCo ^((~BCu)&  BCa );
			Egu =   BCu ^((~BCa)&  BCe );

			Abe ^= De;
			BCa = ROL(Abe,  1);
			Agi ^= Di;
			BCe = ROL(Agi,  6);
			Ako ^= Do;
			BCi = ROL(Ako, 25);
			Amu ^= Du;
			BCo = ROL_mult8(Amu,  8);
			Asa ^= Da;
			BCu = ROL(Asa, 18);
			Eka =   BCa ^((~BCe)&  BCi );
			Eke =   BCe ^((~BCi)&  BCo );
			Eki =   BCi ^((~BCo)&  BCu );
			Eko =   BCo ^((~BCu)&  BCa );
			Eku =   BCu ^((~BCa)&  BCe );

			Abu ^= Du;
			BCa = ROL(Abu, 27);
			Aga ^= Da;
			BCe = ROL(Aga, 36);
			Ake ^= De;
			BCi = ROL(Ake, 10);
			Ami ^= Di;
			BCo = ROL(Ami, 15);
			Aso ^= Do;
			BCu = ROL_mult8(Aso, 56);
			Ema =   BCa ^((~BCe)&  BCi );
			Eme =   BCe ^((~BCi)&  BCo );
			Emi =   BCi ^((~BCo)&  BCu );
			Emo =   BCo ^((~BCu)&  BCa );
			Emu =   BCu ^((~BCa)&  BCe );

			Abi ^= Di;
			BCa = ROL(Abi, 62);
			Ago ^= Do;
			BCe = ROL(Ago, 55);
			Aku ^= Du;
			BCi = ROL(Aku, 39);
			Ama ^= Da;
			BCo = ROL(Ama, 41);
			Ase ^= De;
			BCu = ROL(Ase,  2);
			Esa =   BCa ^((~BCe)&  BCi );
			Ese =   BCe ^((~BCi)&  BCo );
			Esi =   BCi ^((~BCo)&  BCu );
			Eso =   BCo ^((~BCu)&  BCa );
			Esu =   BCu ^((~BCa)&  BCe );

			//    prepareTheta
			BCa = Eba^Ega^Eka^Ema^Esa;
			BCe = Ebe^Ege^Eke^Eme^Ese;
			BCi = Ebi^Egi^Eki^Emi^Esi;
			BCo = Ebo^Ego^Eko^Emo^Eso;
			BCu = Ebu^Egu^Eku^Emu^Esu;

			//thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
			Da = BCu^ROL(BCe, 1);
			De = BCa^ROL(BCi, 1);
			Di = BCe^ROL(BCo, 1);
			Do = BCi^ROL(BCu, 1);
			Du = BCo^ROL(BCa, 1);

			Eba ^= Da;
			BCa = Eba;
			Ege ^= De;
			BCe = ROL(Ege, 44);
			Eki ^= Di;
			BCi = ROL(Eki, 43);
			Emo ^= Do;
			BCo = ROL(Emo, 21);
			Esu ^= Du;
			BCu = ROL(Esu, 14);
			Aba =   BCa ^((~BCe)&  BCi );
			Aba ^= (tKeccakLane)KeccakF_RoundConstants[round+1];
			Abe =   BCe ^((~BCi)&  BCo );
			Abi =   BCi ^((~BCo)&  BCu );
			Abo =   BCo ^((~BCu)&  BCa );
			Abu =   BCu ^((~BCa)&  BCe );

			Ebo ^= Do;
			BCa = ROL(Ebo, 28);
			Egu ^= Du;
			BCe = ROL(Egu, 20);
			Eka ^= Da;
			BCi = ROL(Eka, 3);
			Eme ^= De;
			BCo = ROL(Eme, 45);
			Esi ^= Di;
			BCu = ROL(Esi, 61);
			Aga =   BCa ^((~BCe)&  BCi );
			Age =   BCe ^((~BCi)&  BCo );
			Agi =   BCi ^((~BCo)&  BCu );
			Ago =   BCo ^((~BCu)&  BCa );
			Agu =   BCu ^((~BCa)&  BCe );

			Ebe ^= De;
			BCa = ROL(Ebe, 1);
			Egi ^= Di;
			BCe = ROL(Egi, 6);
			Eko ^= Do;
			BCi = ROL(Eko, 25);
			Emu ^= Du;
			BCo = ROL_mult8(Emu, 8);
			Esa ^= Da;
			BCu = ROL(Esa, 18);
			Aka =   BCa ^((~BCe)&  BCi );
			Ake =   BCe ^((~BCi)&  BCo );
			Aki =   BCi ^((~BCo)&  BCu );
			Ako =   BCo ^((~BCu)&  BCa );
			Aku =   BCu ^((~BCa)&  BCe );

			Ebu ^= Du;
			BCa = ROL(Ebu, 27);
			Ega ^= Da;
			BCe = ROL(Ega, 36);
			Eke ^= De;
			BCi = ROL(Eke, 10);
			Emi ^= Di;
			BCo = ROL(Emi, 15);
			Eso ^= Do;
			BCu = ROL_mult8(Eso, 56);
			Ama =   BCa ^((~BCe)&  BCi );
			Ame =   BCe ^((~BCi)&  BCo );
			Ami =   BCi ^((~BCo)&  BCu );
			Amo =   BCo ^((~BCu)&  BCa );
			Amu =   BCu ^((~BCa)&  BCe );

			Ebi ^= Di;
			BCa = ROL(Ebi, 62);
			Ego ^= Do;
			BCe = ROL(Ego, 55);
			Eku ^= Du;
			BCi = ROL(Eku, 39);
			Ema ^= Da;
			BCo = ROL(Ema, 41);
			Ese ^= De;
			BCu = ROL(Ese, 2);
			Asa =   BCa ^((~BCe)&  BCi );
			Ase =   BCe ^((~BCi)&  BCo );
			Asi =   BCi ^((~BCo)&  BCu );
			Aso =   BCo ^((~BCu)&  BCa );
			Asu =   BCu ^((~BCa)&  BCe );
		}

		//copyToState(state, A)
		state[ 0] = Aba;
		state[ 1] = Abe;
		state[ 2] = Abi;
		state[ 3] = Abo;
		state[ 4] = Abu;
		state[ 5] = Aga;
		state[ 6] = Age;
		state[ 7] = Agi;
		state[ 8] = Ago;
		state[ 9] = Agu;
		state[10] = Aka;
		state[11] = Ake;
		state[12] = Aki;
		state[13] = Ako;
		state[14] = Aku;
		state[15] = Ama;
		state[16] = Ame;
		state[17] = Ami;
		state[18] = Amo;
		state[19] = Amu;
		state[20] = Asa;
		state[21] = Ase;
		state[22] = Asi;
		state[23] = Aso;
		state[24] = Asu;

		#undef    round
	}
}

// inlen kann 72...143 betragen
__host__
void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen)
{
	const unsigned char *in = (const unsigned char*)pdata;

	tKeccakLane state[5 * 5];
	unsigned char temp[cKeccakR_SizeInBytes];

	memset( state, 0, sizeof(state) );

	for ( /* empty */; inlen >= cKeccakR_SizeInBytes; inlen -= cKeccakR_SizeInBytes, in += cKeccakR_SizeInBytes)
	{
		KeccakF( state, (const tKeccakLane*)in, cKeccakR_SizeInBytes / sizeof(tKeccakLane) );
	}

	// Copy state of the first round (72 Bytes)
	// in Constant Memory
	cudaMemcpyToSymbol( c_State,
						state,
						sizeof(state),
						0, cudaMemcpyHostToDevice);

	// second part
	memcpy(temp, in, inlen);
	temp[inlen++] = 1;
	memset(temp + inlen, 0, cKeccakR_SizeInBytes - inlen);
	temp[cKeccakR_SizeInBytes-1] |= 0x80;

	// Copy rest of the message in constant memory
	cudaMemcpyToSymbol( c_PaddedMessage,
						temp,
						cKeccakR_SizeInBytes,
						0, cudaMemcpyHostToDevice);
}

__global__
void jackpot_keccak512_gpu_hash(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		int hashPosition = nounce - startNounce;

		uint32_t message[18];
		#pragma unroll 18
		for(int i=0;i<18;i++)
			message[i] = c_PaddedMessage[i];

		message[1] = cuda_swab32(nounce);

		// State init
		uint64_t keccak_gpu_state[25];
		#pragma unroll 25
		for (int i=0; i<25; i++)
			keccak_gpu_state[i] = c_State[i];

		// den Block einmal gut durchschütteln
		keccak_block(keccak_gpu_state, message, c_keccak_round_constants);

		uint32_t hash[16];

		#pragma unroll 8
		for (size_t i = 0; i < 64; i += 8) {
			U64TO32_LE((&hash[i/4]), keccak_gpu_state[i / 8]);
		}

		// copy hash
		uint32_t *outpHash = (uint32_t*)&g_hash[8 * hashPosition];

		#pragma unroll 16
		for(int i=0;i<16;i++)
			outpHash[i] = hash[i];
	}
}

__host__
void jackpot_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	jackpot_keccak512_gpu_hash<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash);
	MyStreamSynchronize(NULL, order, thr_id);
}


/* zr5 keccak, no nonce swab32 */

__global__
void zr5_keccak512_gpu_hash(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;
		uint32_t message[18];

		#pragma unroll 18
		for(int i=0; i<18; i++)
			message[i] = c_PaddedMessage[i];

		message[1] = nounce;

		// Get mid-state
		uint64_t keccak_gpu_state[25];
		#pragma unroll 25
		for (int i=0; i<25; i++)
			keccak_gpu_state[i] = c_State[i];

		keccak_block(keccak_gpu_state, message, c_keccak_round_constants);

		uint32_t hash[16];

		#pragma unroll 8
		for (int i = 0; i < 8; i++) {
			U64TO32_LE((&hash[i*2]), keccak_gpu_state[i]);
		}

		// Output (64 bytes hash required)
		uint32_t hashPosition = nounce - startNounce;
		//uint32_t *outpHash = (uint32_t*) (&g_hash[hashPosition*8]);
		//#pragma unroll 16
		//for(int i=0; i<16; i++)
		//	outpHash[i] = hash[i];

		uint4 *outpHash = (uint4*) (&g_hash[hashPosition*8]);
		uint4 *psrc = (uint4*) hash;
		outpHash[0] = psrc[0];
		outpHash[1] = psrc[1];
		outpHash[2] = psrc[2];
		outpHash[3] = psrc[3];
	}
}

__host__
void zr5_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	zr5_keccak512_gpu_hash<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash);
	MyStreamSynchronize(NULL, 0, thr_id);
}

/* required for the second hash part of zr5 */

__global__
void zr5_keccak512_gpu_hash_pok(uint32_t threads, uint32_t startNounce, uint32_t *g_hash, uint16_t *d_poks, uint32_t version)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

		uint32_t message[18]; /* 72 bytes */

		// pok - hash[0] from prev hash
		message[0] = version | (0x10000UL * d_poks[thread]);
		#pragma unroll
		for (int i=1; i<18; i++) {
			message[i]=d_OriginalData[i];
		}

		// first bloc
		uint64_t keccak_gpu_state[25] = { 0 };
		keccak_block(keccak_gpu_state, message, c_keccak_round_constants);

		// second bloc
		message[0] = d_OriginalData[18];
		message[1] = nounce; //cuda_swab32(nounce);
		message[2] = 1;

		#pragma unroll
		for(int i=3; i<17; i++)
			message[i] = 0;

		message[17] = 0x80000000UL;

		keccak_block(keccak_gpu_state, message, c_keccak_round_constants);

		uint32_t hash[16];

		#pragma unroll 8
		for (size_t i = 0; i < 8; i++) {
			U64TO32_LE((&hash[i*2]), keccak_gpu_state[i]);
		}

		//uint32_t *outpHash = &g_hash[thread * 16];
		//#pragma unroll 16
		//for(int i=0; i<16; i++)
		//	outpHash[i] = hash[i];

		uint4 *outpHash = (uint4*) (&g_hash[thread * 16]);
		uint4 *psrc = (uint4*) hash;
		outpHash[0] = psrc[0];
		outpHash[1] = psrc[1];
		outpHash[2] = psrc[2];
		outpHash[3] = psrc[3];
	}
}

__host__
void zr5_keccak512_cpu_hash_pok(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* pdata, uint32_t *d_hash, uint16_t *d_poks)
{
	const uint32_t threadsperblock = 256;
	const uint32_t version = (pdata[0] & (~POK_DATA_MASK)) | (use_pok ? POK_BOOL_MASK : 0);

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cudaMemcpyToSymbol(d_OriginalData, pdata, sizeof(d_OriginalData), 0, cudaMemcpyHostToDevice);
	zr5_keccak512_gpu_hash_pok<<<grid, block>>>(threads, startNounce, d_hash, d_poks, version);
	MyStreamSynchronize(NULL, 10, thr_id);
}
