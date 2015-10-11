#ifndef ALGOS_H
#define ALGOS_H

enum sha_algos {
	ALGO_BLAKECOIN = 0,
	ALGO_BLAKE,
	ALGO_BMW,
	ALGO_C11,
	ALGO_DEEP,
	ALGO_DMD_GR,
	ALGO_FRESH,
	ALGO_FUGUE256,		/* Fugue256 */
	ALGO_GROESTL,
	ALGO_HEAVY,		/* Heavycoin hash */
	ALGO_KECCAK,
	ALGO_JACKPOT,
	ALGO_LUFFA,
	ALGO_LYRA2,
	ALGO_LYRA2v2,
	ALGO_MJOLLNIR,		/* Hefty hash */
	ALGO_MYR_GR,
	ALGO_NEOSCRYPT,
	ALGO_NIST5,
	ALGO_PENTABLAKE,
	ALGO_QUARK,
	ALGO_QUBIT,
	ALGO_SCRYPT,
	ALGO_SCRYPT_JANE,
	ALGO_SKEIN,
	ALGO_SKEIN2,
	ALGO_S3,
	ALGO_X11,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X17,
	ALGO_WHIRLCOIN,
	ALGO_WHIRLPOOL,
	ALGO_WHIRLPOOLX,
	ALGO_ZR5,
	ALGO_AUTO,
	ALGO_COUNT
};

static const char *algo_names[] = {
	"blakecoin",
	"blake",
	"bmw",
	"c11",
	"deep",
	"dmd-gr",
	"fresh",
	"fugue256",
	"groestl",
	"heavy",
	"keccak",
	"jackpot",
	"luffa",
	"lyra2",
	"lyra2v2",
	"mjollnir",
	"myr-gr",
	"neoscrypt",
	"nist5",
	"penta",
	"quark",
	"qubit",
	"scrypt",
	"scrypt-jane",
	"skein",
	"skein2",
	"s3",
	"x11",
	"x13",
	"x14",
	"x15",
	"x17",
	"whirlcoin",
	"whirlpool",
	"whirlpoolx",
	"zr5",
	"auto", /* reserved for multi algo */
	""
};


#endif
