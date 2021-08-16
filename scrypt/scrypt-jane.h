#ifndef SCRYPT_JANE_H
#define SCRYPT_JANE_H

/*
	Nfactor: Increases CPU & Memory Hardness
	N = (1 << (Nfactor + 1)): How many times to mix a chunk and how many temporary chunks are used

	rfactor: Increases Memory Hardness
	r = (1 << rfactor): How large a chunk is

	pfactor: Increases CPU Hardness
	p = (1 << pfactor): Number of times to mix the main chunk

	A block is the basic mixing unit (salsa/chacha block = 64 bytes)
	A chunk is (2 * r) blocks

	~Memory used = (N + 2) * ((2 * r) * block size)
*/

#include <stdlib.h>
#include <stdint.h>
#include <memory.h>

typedef void (*scrypt_fatal_errorfn)(const char *msg);
void scrypt_set_fatal_error(scrypt_fatal_errorfn fn);

void scrypt_N_1_1(const unsigned char *password, size_t password_len, const unsigned char *salt, size_t salt_len, uint32_t N, unsigned char *out, size_t bytes, uint8_t *X, uint8_t *Y, uint8_t *V);

#endif /* SCRYPT_JANE_H */
