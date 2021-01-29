/**
 * Wrapper to OpenSSL BIGNUM used by net diff (nBits)
 */

#include <stdio.h>

#include "uint256.h"

#include <openssl/opensslv.h>
#if OPENSSL_VERSION_NUMBER < 0x10100000L
#include "compat/bignum_ssl10.hpp"
#else
#include "bignum.hpp"
#endif

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

// unused, but should allow more than 256bits targets
#if 0
extern "C" double bn_hash_target_ratio(uint32_t* hash, uint32_t* target)
{
	double dhash;

	if (!opt_showdiff)
		return 0.0;

	CBigNum h(0), t(0);
	std::vector<unsigned char> vch(32);

	memcpy(&vch[0], (void*) target, 32);
	t.setvch(vch);
	memcpy(&vch[0], (void*) hash, 32);
	h.setvch(vch);

	dhash = h.getuint256().getdouble();
	if (dhash > 0.)
		return t.getuint256().getdouble() / dhash;
	else
		return dhash;
}
#endif

// compute the diff ratio between a found hash and the target
extern "C" double bn_hash_target_ratio(uint32_t* hash, uint32_t* target)
{
	uint256 h, t;
	double dhash;

	if (!opt_showdiff)
		return 0.0;

	memcpy(&t, (void*) target, 32);
	memcpy(&h, (void*) hash, 32);

	dhash = h.getdouble();
	if (dhash > 0.)
		return t.getdouble() / dhash;
	else
		return dhash;
}

// store ratio in work struct
extern "C" void bn_store_hash_target_ratio(uint32_t* hash, uint32_t* target, struct work* work, int nonce)
{
	// only if the option is enabled (to reduce cpu usage)
	if (!opt_showdiff) return;
	if (nonce < 0 || nonce >= MAX_NONCES) return;

	work->shareratio[nonce] = bn_hash_target_ratio(hash, target);
	work->sharediff[nonce] = work->targetdiff * work->shareratio[nonce];
}

// new method to save all nonce(s) share diff/ration
extern "C" void bn_set_target_ratio(struct work* work, uint32_t* hash, int nonce)
{
	bn_store_hash_target_ratio(hash, work->target, work, nonce);
}

// compat (only store single nonce share diff per work)
extern "C" void work_set_target_ratio(struct work* work, uint32_t* hash)
{
	bn_store_hash_target_ratio(hash, work->target, work, work->submit_nonce_id);
}

