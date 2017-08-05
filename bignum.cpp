/**
 * Wrapper to OpenSSL BIGNUM used by net diff (nBits)
 */

#include <stdio.h>

#include "uint256.h"
#include "bignum.hpp"

#include "miner.h" // hex2bin

extern "C" double bn_convert_nbits(const uint32_t nBits)
{
	uint256 bn = CBigNum().SetCompact(nBits).getuint256();
	return bn.getdouble();
}

// copy the big number to 32-bytes uchar
extern "C" void bn_nbits_to_uchar(const uint32_t nBits, unsigned char *target)
{
	char buff[65];
	uint256 bn = CBigNum().SetCompact(nBits).getuint256();

	snprintf(buff, 65, "%s\n", bn.ToString().c_str()); buff[64] = '\0';
	hex2bin(target, buff, 32);
}
