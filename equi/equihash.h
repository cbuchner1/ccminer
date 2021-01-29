#ifndef EQUIHASH_H
#define EQUIHASH_H

#include <stdint.h>

// miner nonce "cursor" unique for each thread
#define EQNONCE_OFFSET 30 /* 27:34 */

#define WK 9
#define WN 200
//#define CONFIG_MODE_1 9, 1248, 12, 640, packer_cantor /* eqcuda.hpp */

extern "C" {
	void equi_hash(const void* input, void* output, int len);
	int  equi_verify_sol(void* const hdr, void* const soln);
	bool equi_verify(uint8_t* const hdr, uint8_t* const soln);
}

#endif
