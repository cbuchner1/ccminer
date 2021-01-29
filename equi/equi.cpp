/*
 * Port to Generic C of C++ implementation of the Equihash Proof-of-Work
 * algorithm from zcashd.
 *
 * Copyright (c) 2016 abc at openwall dot com
 * Copyright (c) 2016 Jack Grigg
 * Copyright (c) 2016 The Zcash developers
 * Copyright (c) 2017 tpruvot
 *
 * Distributed under the MIT software license, see the accompanying
 * file COPYING or http://www.opensource.org/licenses/mit-license.php.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include "equihash.h"

//#define USE_LIBSODIUM

#ifdef USE_LIBSODIUM
#include "sodium.h"
#define blake2b_state crypto_generichash_blake2b_state
#else
#include "blake2/blake2.h"
#define be32toh(x) swab32(x)
#define htole32(x) (x)
#define HASHOUT 50
#endif

#include <miner.h>

static void digestInit(blake2b_state *S, const uint32_t n, const uint32_t k)
{
	uint32_t le_N = htole32(n);
	uint32_t le_K = htole32(k);
#ifdef USE_LIBSODIUM
	uint8_t personalization[crypto_generichash_blake2b_PERSONALBYTES] = { 0 };

	memcpy(personalization, "ZcashPoW", 8);
	memcpy(personalization + 8,  &le_N, 4);
	memcpy(personalization + 12, &le_K, 4);

	crypto_generichash_blake2b_init_salt_personal(S,
		NULL, 0, (512 / n) * n / 8, NULL, personalization);
#else
	unsigned char personal[] = "ZcashPoW01230123";
	memcpy(personal + 8, &le_N, 4);
	memcpy(personal + 12, &le_K, 4);
	blake2b_param P[1];
	P->digest_length = HASHOUT;
	P->key_length = 0;
	P->fanout = 1;
	P->depth = 1;
	P->leaf_length = 0;
	P->node_offset = 0;
	P->node_depth = 0;
	P->inner_length = 0;
	memset(P->reserved, 0, sizeof(P->reserved));
	memset(P->salt, 0, sizeof(P->salt));
	memcpy(P->personal, (const uint8_t *)personal, 16);
	eq_blake2b_init_param(S, P);
#endif
}

static void expandArray(const unsigned char *in, const uint32_t in_len,
	unsigned char *out, const uint32_t out_len,
	const uint32_t bit_len, const uint32_t byte_pad)
{
	assert(bit_len >= 8);
	assert(8 * sizeof(uint32_t) >= 7 + bit_len);

	const uint32_t out_width = (bit_len + 7) / 8 + byte_pad;
	assert(out_len == 8 * out_width * in_len / bit_len);

	const uint32_t bit_len_mask = ((uint32_t)1 << bit_len) - 1;

	// The acc_bits least-significant bits of acc_value represent a bit sequence
	// in big-endian order.
	uint32_t acc_bits = 0;
	uint32_t acc_value = 0;
	uint32_t j = 0;

	for (uint32_t i = 0; i < in_len; i++)
	{
		acc_value = (acc_value << 8) | in[i];
		acc_bits += 8;

		// When we have bit_len or more bits in the accumulator, write the next
		// output element.
		if (acc_bits >= bit_len) {
			acc_bits -= bit_len;
			for (uint32_t x = 0; x < byte_pad; x++) {
				out[j + x] = 0;
			}
			for (uint32_t x = byte_pad; x < out_width; x++) {
				out[j + x] = (
					// Big-endian
					acc_value >> (acc_bits + (8 * (out_width - x - 1)))
				) & (
					// Apply bit_len_mask across byte boundaries
					(bit_len_mask >> (8 * (out_width - x - 1))) & 0xFF
				);
			}
			j += out_width;
		}
	}
}

static void generateHash(blake2b_state *S, const uint32_t g, uint8_t *hash, const size_t hashLen)
{
	const uint32_t le_g = htole32(g);
	blake2b_state digest = *S; /* copy */
#ifdef USE_LIBSODIUM
	crypto_generichash_blake2b_update(&digest, (uint8_t *)&le_g, sizeof(le_g));
	crypto_generichash_blake2b_final(&digest, hash, hashLen);
#else
	eq_blake2b_update(&digest, (const uint8_t*) &le_g, sizeof(le_g));
	eq_blake2b_final(&digest, hash, (uint8_t) (hashLen & 0xFF));
#endif
}

static int isZero(const uint8_t *hash, size_t len)
{
	// This doesn't need to be constant time.
	for (size_t i = 0; i < len; i++) {
		if (hash[i] != 0) return 0;
	}
	return 1;
}

// hdr -> header including nonce (140 bytes)
// soln -> equihash solution (excluding 3 bytes with size, so 1344 bytes length)
bool equi_verify(uint8_t* const hdr, uint8_t* const soln)
{
	const uint32_t n = WN; // 200
	const uint32_t k = WK; // 9
	const uint32_t collisionBitLength = n / (k + 1);
	const uint32_t collisionByteLength = (collisionBitLength + 7) / 8;
	const uint32_t hashLength = (k + 1) * collisionByteLength;
	const uint32_t indicesPerHashOutput = 512 / n;
	const uint32_t hashOutput = indicesPerHashOutput * n / 8;
	const uint32_t equihashSolutionSize = (1 << k) * (n / (k + 1) + 1) / 8;
	const uint32_t solnr = 1 << k;

	uint32_t indices[512] = { 0 };
	uint8_t vHash[hashLength] = { 0 };

	blake2b_state state;
	digestInit(&state, n, k);
#ifdef USE_LIBSODIUM
	crypto_generichash_blake2b_update(&state, hdr, 140);
#else
	eq_blake2b_update(&state, hdr, 140);
#endif

	expandArray(soln, equihashSolutionSize, (uint8_t*) &indices, sizeof(indices), collisionBitLength + 1, 1);

	for (uint32_t j = 0; j < solnr; j++) {
		uint8_t tmpHash[hashOutput];
		uint8_t hash[hashLength];
		uint32_t i = be32toh(indices[j]);
		generateHash(&state, i / indicesPerHashOutput, tmpHash, hashOutput);
		expandArray(tmpHash + (i % indicesPerHashOutput * n / 8), n / 8, hash, hashLength, collisionBitLength, 0);
		for (uint32_t k = 0; k < hashLength; k++)
			vHash[k] ^= hash[k];
	}
	return isZero(vHash, sizeof(vHash));
}
