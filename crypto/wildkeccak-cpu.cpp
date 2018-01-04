// Memory-hard extension of keccak for PoW
// Copyright (c) 2012-2013 The Cryptonote developers
// Copyright (c) 2014 The Boolberry developers

// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// Modified for CPUminer by Lucas Jones
// Adapted for ccminer by Tanguy Pruvot - 2016

#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#include <emmintrin.h>
#include <openssl/opensslv.h>
#if OPENSSL_VERSION_NUMBER < 0x10100000L
#include "compat/bignum_ssl10.hpp"
#else
#include "bignum.hpp"
#endif
#include "int128_c.h"
#else
#include <x86intrin.h>
#endif

#include <miner.h>

#include "xmr-rpc.h"

extern uint64_t* pscratchpad_buff;

struct reciprocal_value64 {
	uint64_t m;
	uint8_t sh1, sh2;
};

static inline int fls64(uint64_t x)
{
#if defined(_WIN64)
	unsigned long bitpos = 0;
	_BitScanReverse64(&bitpos, x);
	return (int) (bitpos + 1);
#elif defined(WIN32)
	unsigned long hipos = 0, bitpos = 0;
	uint32_t hi = x >> 32;
	_BitScanReverse(&hipos, hi);
	if (!hipos) {
		_BitScanReverse(&bitpos, (uint32_t) x);
	}
	return (int) hipos ? hipos + 33 : bitpos + 1;
#else
	/*
	* AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
	* dest reg is undefined if x==0, but their CPU architect says its
	* value is written to set it to the same as before.
	*/
	register long bitpos = -1;
	asm("bsrq %1,%0" : "+r" (bitpos) : "rm" (x));
	return bitpos + 1;
#endif
}

static inline struct reciprocal_value64 reciprocal_val64(uint64_t d)
{
	struct reciprocal_value64 R;
	int l;

	l = fls64(d - 1);

#ifdef _MSC_VER
	uint128 v1;
	v1.Lo = (1ULL << l) - d;v1.Hi=0;
	uint128 v2;
	v2.Hi = 1; v2.Lo = 0;

	uint128 v;
	mult128(v1,v2,&v);
	divmod128by64(v.Hi,v.Lo,d,&v.Hi,&v.Lo);
	Increment(&v);
	R.m = (uint64_t)v.Hi;
#else
    __uint128_t m;
    m = (((__uint128_t)1 << 64) * ((1ULL << l) - d));
    m /= d;
	++m;
	R.m = (uint64_t)m;
#endif

	R.sh1 = min(l, 1);
	R.sh2 = max(l - 1, 0);

	return R;
}

static inline uint64_t reciprocal_divide64(uint64_t a, struct reciprocal_value64 R)
{
#ifdef _MSC_VER
    uint128 v;
	mult64to128(a,R.m,&v.Hi,&v.Lo);
	uint64_t t = v.Hi;
#else
    uint64_t t = (uint64_t)(((__uint128_t)a * R.m) >> 64);
#endif
	return (t + ((a - t) >> R.sh1)) >> R.sh2;
}

static inline uint64_t reciprocal_remainder64(uint64_t A, uint64_t B, struct reciprocal_value64 R)
{
	uint64_t div, mod;

	div = reciprocal_divide64(A, R);
	mod = A - (uint64_t) (div * B);
	if (mod >= B) mod -= B;
	return mod;
}

//#define UNROLL_SCR_MIX

static inline uint64_t rotl641(uint64_t x) { return((x << 1) | (x >> 63)); }
static inline uint64_t rotl64_1(uint64_t x, uint64_t y) { return((x << y) | (x >> (64 - y))); }
static inline uint64_t rotl64_2(uint64_t x, uint64_t y) { return(rotl64_1((x >> 32) | (x << 32), y)); }
static inline uint64_t bitselect(uint64_t a, uint64_t b, uint64_t c) { return(a ^ (c & (b ^ a))); }

static inline void keccakf_mul(uint64_t *s)
{
	uint64_t bc[5], t[5];
	uint64_t tmp1, tmp2;
	int i;

	for(i = 0; i < 5; i++)
		t[i] = s[i + 0] ^ s[i + 5] ^ s[i + 10] * s[i + 15] * s[i + 20];

	bc[0] = t[0] ^ rotl641(t[2]);
	bc[1] = t[1] ^ rotl641(t[3]);
	bc[2] = t[2] ^ rotl641(t[4]);
	bc[3] = t[3] ^ rotl641(t[0]);
	bc[4] = t[4] ^ rotl641(t[1]);

	tmp1 = s[1] ^ bc[0];

	s[ 0] ^= bc[4];
	s[ 1] = rotl64_1(s[ 6] ^ bc[0], 44);
	s[ 6] = rotl64_1(s[ 9] ^ bc[3], 20);
	s[ 9] = rotl64_1(s[22] ^ bc[1], 61);
	s[22] = rotl64_1(s[14] ^ bc[3], 39);
	s[14] = rotl64_1(s[20] ^ bc[4], 18);
	s[20] = rotl64_1(s[ 2] ^ bc[1], 62);
	s[ 2] = rotl64_1(s[12] ^ bc[1], 43);
	s[12] = rotl64_1(s[13] ^ bc[2], 25);
	s[13] = rotl64_1(s[19] ^ bc[3], 8);
	s[19] = rotl64_1(s[23] ^ bc[2], 56);
	s[23] = rotl64_1(s[15] ^ bc[4], 41);
	s[15] = rotl64_1(s[ 4] ^ bc[3], 27);
	s[ 4] = rotl64_1(s[24] ^ bc[3], 14);
	s[24] = rotl64_1(s[21] ^ bc[0], 2);
	s[21] = rotl64_1(s[ 8] ^ bc[2], 55);
	s[ 8] = rotl64_1(s[16] ^ bc[0], 45);
	s[16] = rotl64_1(s[ 5] ^ bc[4], 36);
	s[ 5] = rotl64_1(s[ 3] ^ bc[2], 28);
	s[ 3] = rotl64_1(s[18] ^ bc[2], 21);
	s[18] = rotl64_1(s[17] ^ bc[1], 15);
	s[17] = rotl64_1(s[11] ^ bc[0], 10);
	s[11] = rotl64_1(s[ 7] ^ bc[1], 6);
	s[ 7] = rotl64_1(s[10] ^ bc[4], 3);
	s[10] = rotl64_1(tmp1, 1);

	tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
	tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
	tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
	tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
	tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);

	s[0] ^= 0x0000000000000001ULL;
}

static inline void keccakf_mul_last(uint64_t *s)
{
	uint64_t bc[5], xormul[5];
	uint64_t tmp1, tmp2;
	int i;

	for(i = 0; i < 5; i++)
		xormul[i] = s[i + 0] ^ s[i + 5] ^ s[i + 10] * s[i + 15] * s[i + 20];

	bc[0] = xormul[0] ^ rotl641(xormul[2]);
	bc[1] = xormul[1] ^ rotl641(xormul[3]);
	bc[2] = xormul[2] ^ rotl641(xormul[4]);
	bc[3] = xormul[3] ^ rotl641(xormul[0]);
	bc[4] = xormul[4] ^ rotl641(xormul[1]);

	s[0] ^= bc[4];
	s[1] = rotl64_2(s[6] ^ bc[0], 12);
	s[2] = rotl64_2(s[12] ^ bc[1], 11);
	s[4] = rotl64_1(s[24] ^ bc[3], 14);
	s[3] = rotl64_1(s[18] ^ bc[2], 21);

	tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]);
	s[0] ^= 0x0000000000000001ULL;
}

struct reciprocal_value64 cached_recip;
static uint64_t cached_scr_size = 0;

static inline void scr_mix(uint64_t *st, uint64_t scr_size, struct reciprocal_value64 recip)
{
#define KK_MIXIN_SIZE 24
	uint64_t _ALIGN(128) idx[KK_MIXIN_SIZE];

#ifdef _MSC_VER
	#define pscr pscratchpad_buff
	int x;

	// non-optimized 64bit operations
	for (x = 0; x < KK_MIXIN_SIZE; x++) {
		idx[x] = reciprocal_remainder64(st[x], scr_size, recip) << 2;
	}
	if (idx[7] > scr_size*4) {
		applog(LOG_WARNING, "Wrong remainder64 returned by the cpu hash %016llx > %016llx",
			(unsigned long long) idx[7], (unsigned long long) scr_size*4);
		return;
	}
	for(x = 0; x < KK_MIXIN_SIZE; x += 4) {
		st[x + 0] ^= pscr[idx[x] + 0] ^ pscr[idx[x + 1] + 0] ^ pscr[idx[x + 2] + 0] ^ pscr[idx[x + 3] + 0];
		st[x + 1] ^= pscr[idx[x] + 1] ^ pscr[idx[x + 1] + 1] ^ pscr[idx[x + 2] + 1] ^ pscr[idx[x + 3] + 1];
		st[x + 2] ^= pscr[idx[x] + 2] ^ pscr[idx[x + 1] + 2] ^ pscr[idx[x + 2] + 2] ^ pscr[idx[x + 3] + 2];
		st[x + 3] ^= pscr[idx[x] + 3] ^ pscr[idx[x + 1] + 3] ^ pscr[idx[x + 2] + 3] ^ pscr[idx[x + 3] + 3];
	}
	return;

#elif !defined(UNROLL_SCR_MIX)

	#pragma GCC ivdep
	for(int x = 0; x < 3; ++x)
	{
		__m128i *st0, *st1, *st2, *st3;

		idx[0] = reciprocal_remainder64(st[(x << 3) + 0], scr_size, recip) << 2;
		idx[1] = reciprocal_remainder64(st[(x << 3) + 1], scr_size, recip) << 2;
		idx[2] = reciprocal_remainder64(st[(x << 3) + 2], scr_size, recip) << 2;
		idx[3] = reciprocal_remainder64(st[(x << 3) + 3], scr_size, recip) << 2;
		idx[4] = reciprocal_remainder64(st[(x << 3) + 4], scr_size, recip) << 2;
		idx[5] = reciprocal_remainder64(st[(x << 3) + 5], scr_size, recip) << 2;
		idx[6] = reciprocal_remainder64(st[(x << 3) + 6], scr_size, recip) << 2;
		idx[7] = reciprocal_remainder64(st[(x << 3) + 7], scr_size, recip) << 2;

		for(int y = 0; y < 8; y++) _mm_prefetch((const char*) (&pscratchpad_buff[idx[y]]), _MM_HINT_T1);

		st0 = (__m128i *)&st[(x << 3) + 0];
		st1 = (__m128i *)&st[(x << 3) + 2];
		st2 = (__m128i *)&st[(x << 3) + 4];
		st3 = (__m128i *)&st[(x << 3) + 6];

		*st0 = _mm_xor_si128(*st0, *((__m128i *)&pscratchpad_buff[idx[0]]));
		*st0 = _mm_xor_si128(*st0, *((__m128i *)&pscratchpad_buff[idx[1]]));
		*st0 = _mm_xor_si128(*st0, *((__m128i *)&pscratchpad_buff[idx[2]]));
		*st0 = _mm_xor_si128(*st0, *((__m128i *)&pscratchpad_buff[idx[3]]));

		*st1 = _mm_xor_si128(*st1, *((__m128i *)&pscratchpad_buff[idx[0] + 2]));
		*st1 = _mm_xor_si128(*st1, *((__m128i *)&pscratchpad_buff[idx[1] + 2]));
		*st1 = _mm_xor_si128(*st1, *((__m128i *)&pscratchpad_buff[idx[2] + 2]));
		*st1 = _mm_xor_si128(*st1, *((__m128i *)&pscratchpad_buff[idx[3] + 2]));

		*st2 = _mm_xor_si128(*st2, *((__m128i *)&pscratchpad_buff[idx[4]]));
		*st2 = _mm_xor_si128(*st2, *((__m128i *)&pscratchpad_buff[idx[5]]));
		*st2 = _mm_xor_si128(*st2, *((__m128i *)&pscratchpad_buff[idx[6]]));
		*st2 = _mm_xor_si128(*st2, *((__m128i *)&pscratchpad_buff[idx[7]]));

		*st3 = _mm_xor_si128(*st3, *((__m128i *)&pscratchpad_buff[idx[4] + 2]));
		*st3 = _mm_xor_si128(*st3, *((__m128i *)&pscratchpad_buff[idx[5] + 2]));
		*st3 = _mm_xor_si128(*st3, *((__m128i *)&pscratchpad_buff[idx[6] + 2]));
		*st3 = _mm_xor_si128(*st3, *((__m128i *)&pscratchpad_buff[idx[7] + 2]));
	}

#else
	#warning using AVX2 optimizations

	idx[ 0] = reciprocal_remainder64(st[0], scr_size, recip) << 2;
	idx[ 1] = reciprocal_remainder64(st[1], scr_size, recip) << 2;
	idx[ 2] = reciprocal_remainder64(st[2], scr_size, recip) << 2;
	idx[ 3] = reciprocal_remainder64(st[3], scr_size, recip) << 2;
	idx[ 4] = reciprocal_remainder64(st[4], scr_size, recip) << 2;
	idx[ 5] = reciprocal_remainder64(st[5], scr_size, recip) << 2;
	idx[ 6] = reciprocal_remainder64(st[6], scr_size, recip) << 2;
	idx[ 7] = reciprocal_remainder64(st[7], scr_size, recip) << 2;

	for(int y = 0; y < 8; y++) _mm_prefetch(&pscratchpad_buff[idx[y]], _MM_HINT_T1);

	idx[ 8] = reciprocal_remainder64(st[8], scr_size, recip) << 2;
	idx[ 9] = reciprocal_remainder64(st[9], scr_size, recip) << 2;
	idx[10] = reciprocal_remainder64(st[10], scr_size, recip) << 2;
	idx[11] = reciprocal_remainder64(st[11], scr_size, recip) << 2;
	idx[12] = reciprocal_remainder64(st[12], scr_size, recip) << 2;
	idx[13] = reciprocal_remainder64(st[13], scr_size, recip) << 2;
	idx[14] = reciprocal_remainder64(st[14], scr_size, recip) << 2;
	idx[15] = reciprocal_remainder64(st[15], scr_size, recip) << 2;

	for(int y = 8; y < 16; ++y) _mm_prefetch(&pscratchpad_buff[idx[y]], _MM_HINT_T1);

	idx[16] = reciprocal_remainder64(st[16], scr_size, recip) << 2;
	idx[17] = reciprocal_remainder64(st[17], scr_size, recip) << 2;
	idx[18] = reciprocal_remainder64(st[18], scr_size, recip) << 2;
	idx[19] = reciprocal_remainder64(st[19], scr_size, recip) << 2;
	idx[20] = reciprocal_remainder64(st[20], scr_size, recip) << 2;
	idx[21] = reciprocal_remainder64(st[21], scr_size, recip) << 2;
	idx[22] = reciprocal_remainder64(st[22], scr_size, recip) << 2;
	idx[23] = reciprocal_remainder64(st[23], scr_size, recip) << 2;

	for(int y = 16; y < 24; ++y) _mm_prefetch(&pscratchpad_buff[idx[y]], _MM_HINT_T1);

	__m256i *st0 = (__m256i *)&st[0];

	for(int x = 0; x < 6; ++x)
	{
		*st0 = _mm256_xor_si256(*st0, *((__m256i *)&pscratchpad_buff[idx[(x << 2) + 0]]));
		*st0 = _mm256_xor_si256(*st0, *((__m256i *)&pscratchpad_buff[idx[(x << 2) + 1]]));
		*st0 = _mm256_xor_si256(*st0, *((__m256i *)&pscratchpad_buff[idx[(x << 2) + 2]]));
		*st0 = _mm256_xor_si256(*st0, *((__m256i *)&pscratchpad_buff[idx[(x << 2) + 3]]));
		++st0;
	}

#endif
	return;
}

static void wild_keccak_hash_dbl(uint8_t * __restrict md, const uint8_t * __restrict in)
{
	uint64_t _ALIGN(32) st[25];
	uint64_t scr_size, i;
	struct reciprocal_value64 recip;

	scr_size = scratchpad_size >> 2;
	if (scr_size == cached_scr_size)
		recip = cached_recip;
	else {
		cached_recip = recip = reciprocal_val64(scr_size);
		cached_scr_size = scr_size;
	}

	// Wild Keccak #1
	memcpy(st, in, 88);
	st[10] = (st[10] & 0x00000000000000FFULL) | 0x0000000000000100ULL;
	memset(&st[11], 0, 112);
	st[16] |= 0x8000000000000000ULL;

	for(i = 0; i < 23; i++) {
		keccakf_mul(st);
		scr_mix(st, scr_size, recip);
	}

	keccakf_mul_last(st);

	// Wild Keccak #2
	memset(&st[4], 0x00, 168);
	st[ 4] = 0x0000000000000001ULL;
	st[16] = 0x8000000000000000ULL;

	for(i = 0; i < 23; i++) {
		keccakf_mul(st);
		scr_mix(st, scr_size, recip);
	}

	keccakf_mul_last(st);

	memcpy(md, st, 32);
	return;
}

void wildkeccak_hash(void* output, const void* input, uint64_t* scratchpad, uint64_t ssize)
{
	if (scratchpad) pscratchpad_buff = scratchpad;
	if (!scratchpad_size) scratchpad_size = ssize;
	wild_keccak_hash_dbl((uint8_t*)output, (uint8_t*)input);
}
