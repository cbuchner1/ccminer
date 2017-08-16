/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2014 pooler
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */
 
#include "cpuminer-config.h"
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h> 
#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#ifdef WIN32

#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/resource.h>
#if HAVE_SYS_SYSCTL_H
#include <sys/types.h>
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <sys/sysctl.h>
#endif
#endif
#include <jansson.h>
#include <curl/curl.h>
#include <openssl/sha.h>
#include "compat.h"
#include "miner.h"

#ifdef WIN32
#include <Mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

#define PROGRAM_NAME		"ccminer djm edition"
#define LP_SCANTIME		60
#define HEAVYCOIN_BLKHDR_SZ		84
#define MNR_BLKHDR_SZ 80

// from heavy.cu
#ifdef __cplusplus
extern "C"
{
#endif
int cuda_num_devices();
void cuda_devicenames();
int cuda_finddevice(char *name);
#ifdef __cplusplus
}
#endif


#ifdef __linux /* Linux specific policy and affinity management */
#include <sched.h>
static inline void drop_policy(void)
{
	struct sched_param param;
	param.sched_priority = 0;

#ifdef SCHED_IDLE
	if (unlikely(sched_setscheduler(0, SCHED_IDLE, &param) == -1))
#endif
#ifdef SCHED_BATCH
		sched_setscheduler(0, SCHED_BATCH, &param);
#endif
}

static inline void affine_to_cpu(int id, int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(&set), &set);
}
#elif defined(__FreeBSD__) /* FreeBSD specific policy and affinity management */
#include <sys/cpuset.h>
static inline void drop_policy(void)
{
}

static inline void affine_to_cpu(int id, int cpu)
{
	cpuset_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(cpuset_t), &set);
}
#else
static inline void drop_policy(void)
{
}

static inline void affine_to_cpu(int id, int cpu)
{
}
#endif
		
enum workio_commands {
	WC_GET_WORK,
	WC_SUBMIT_WORK,
};

struct workio_cmd {
	enum workio_commands	cmd;
	struct thr_info		*thr;
	union {
		struct work	*work;
	} u;
};

typedef enum {
	ALGO_HEAVY,		/* Heavycoin hash */
	ALGO_MJOLLNIR,		/* Mjollnir hash */
	ALGO_FUGUE256,		/* Fugue256 */
	ALGO_GROESTL,
	ALGO_MYR_GR,
	ALGO_JACKPOT,
	ALGO_QUARK,
	ALGO_ANIME,
	ALGO_QUBIT,
	ALGO_FRESH,
	ALGO_NIST5,
	ALGO_X11,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X17,
	ALGO_WH,
	ALGO_KECCAK,
	ALGO_M7,
	ALGO_LYRA,
    ALGO_NEOSCRYPT,
	ALGO_PLUCK,
	ALGO_DEEP,
	ALGO_DOOM,
	ALGO_DMD_GR,
	ALGO_GOAL,
} sha256_algos;

static const char *algo_names[] = {
	"heavy",
	"mjollnir",
	"fugue256",
	"groestl",
	"myr-gr",
	"jackpot",
	"quark",
	"anime",
	"qubit",
	"fresh",
	"nist5",
	"x11",
	"x13",
	"x14",
	"x15",
	"x17",
	"whirlcoin",
	"keccak",
	"m7",
	"lyra2",
	"neoscrypt",
	"pluck",
	"deep",
	"doom",
	"dmd-gr",
	"goalcoin",
};

bool opt_debug = false;
bool opt_protocol = false;
bool opt_benchmark = false;
bool want_longpoll = true;
bool have_longpoll = false;
bool want_stratum = true;
bool have_gbt = true;
bool have_stratum = false;
bool allow_getwork = true;
bool opt_redirect = true;
static bool submit_old = false;
static char* lp_id;
bool use_syslog = false;
static bool opt_background = false;
static bool opt_quiet = false;
static int opt_retries = -1;
static int opt_fail_pause = 30;
int opt_timeout = 270;
static int opt_scantime = 5;
static json_t *opt_config;
static const bool opt_time = true;
static sha256_algos opt_algo = ALGO_HEAVY;
static int opt_n_threads = 0;
static double opt_difficulty = 1.; // CH
bool opt_trust_pool = false;
uint16_t opt_vote = 9999;
static int num_processors;
int device_map[8] = {0,1,2,3,4,5,6,7}; // CB
char *device_name[8]; // CB
float tp_coef[8] = { -1.0};
static char *rpc_url;
static char *rpc_userpass;
static char *rpc_user, *rpc_pass;
static int pk_script_size;
static unsigned char pk_script[25];
static char coinbase_sig[101] = "";
char *opt_cert;
char *opt_proxy;
long opt_proxy_type;
struct thr_info *thr_info;
static int work_thr_id;
int longpoll_thr_id = -1;
int stratum_thr_id = -1;
struct work_restart *work_restart = NULL;
static struct stratum_ctx stratum;
//// m7 stuff
static unsigned char pblank[1];
const void* ptr; 
    size_t sz; 
uint32_t *m7buf;
////////////////


pthread_mutex_t applog_lock;
static pthread_mutex_t stats_lock;

static unsigned long accepted_count = 0L;
static unsigned long rejected_count = 0L;
static double *thr_hashrates;

#ifdef HAVE_GETOPT_LONG
#include <getopt.h>
#else
struct option {
	const char *name;
	int has_arg;
	int *flag;
	int val;
};
#endif 

static char const usage[] = "\
Usage: " PROGRAM_NAME " [OPTIONS]\n\
Options:\n\
  -a, --algo=ALGO       specify the algorithm to use\n\
                        fugue256  Fuguecoin hash\n\
                        heavy     Heavycoin hash\n\
                        mjollnir  Mjollnircoin hash\n\
                        groestl   Groestlcoin hash\n\
                        myr-gr    Myriad-Groestl hash\n\
                        jackpot   Jackpot hash\n\
                        quark     Quark hash\n\
                        anime     Animecoin hash\n\
		        qubit     qubitcoin hash\n\
		        fresh     freshcoin hash\n\
                        nist5     NIST5 (TalkCoin) hash\n\
                        x11       X11 (DarkCoin) hash\n\
                        x13       X13 (MaruCoin) hash\n\
                        x14       X14 (MoronCoin) hash\n\
			x15       X15 (BitBlock) hash\n\
			x17       X17 (people currency coin) hash\n\
			whirlcoin  whirlcoin (whirlcoin) hash\n\
			keccak     keccak256 (maxcoin) hash\n\
			m7         m7  (crytonite) hash\n\
			lyra2      lyra2RE  (VertCoin) hash\n\
            neoscrypt  neoscrypt (FeatherCoin) hash\n\
            pluck      pluck (SupCoin) hash\n\
			deep       deep  (deepcoin) hash\n\
			doom       doomcoin  hash\n\
                        dmd-gr    Diamond-Groestl hash\n\
			goalcoin   goalcoin hash\n\
  -d, --devices         takes a comma separated list of CUDA devices to use.\n\
                        Device IDs start counting from 0! Alternatively takes\n\
                        string names of your cards like gtx780ti or gt640#2\n\
                        (matching 2nd gt640 in the PC)\n\
  -F, --throughput     coefficient to apply to the number of threads\n\
  -f, --diff            Divide difficulty by this factor (std is 1) \n\
  -v, --vote=VOTE       block reward vote (for HeavyCoin)\n\
  -m, --trust-pool      trust the max block reward vote (maxvote) sent by the pool\n\
  -o, --url=URL         URL of mining server\n\
  -O, --userpass=U:P    username:password pair for mining server\n\
  -u, --user=USERNAME   username for mining server\n\
  -p, --pass=PASSWORD   password for mining server\n\
      --cert=FILE       certificate for mining server using SSL\n\
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy\n\
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs)\n\
  -r, --retries=N       number of times to retry if a network call fails\n\
                          (default: retry indefinitely)\n\
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 30)\n\
  -T, --timeout=N       network timeout, in seconds (default: 270)\n\
  -s, --scantime=N      upper bound on time spent scanning current work when\n\
                          long polling is unavailable, in seconds (default: 5)\n\
      --no-longpoll     disable X-Long-Polling support\n\
      --no-stratum      disable X-Stratum support\n\
  -q, --quiet           disable per-thread hashmeter output\n\
  -D, --debug           enable debug output\n\
  -P, --protocol-dump   verbose dump of protocol-level activities\n"
#ifdef HAVE_SYSLOG_H
"\
  -S, --syslog          use system log for output messages\n"
#endif
#ifndef WIN32
"\
  -B, --background      run the miner in the background\n"
#endif
"\
      --benchmark       run in offline benchmark mode\n\
  -c, --config=FILE     load a JSON-format configuration file\n\
  -V, --version         display version information and exit\n\
  -h, --help            display this help text and exit\n\
";

static char const short_options[] =
#ifndef WIN32
	"B"
#endif
#ifdef HAVE_SYSLOG_H
	"S"
#endif
	"a:c:Dhp:Px:qr:R:s:t:T:o:u:O:Vd:F:f:mv:";
 
static struct option const options[] = {
	{ "algo", 1, NULL, 'a' },
#ifndef WIN32
	{ "background", 0, NULL, 'B' },
#endif
	{ "benchmark", 0, NULL, 1005 },
	{ "cert", 1, NULL, 1001 },
	{ "coinbase-addr", 1, NULL, 1013 },
	{ "coinbase-sig", 1, NULL, 1015 },
	{ "config", 1, NULL, 'c' },
	{ "debug", 0, NULL, 'D' },
	{ "help", 0, NULL, 'h' },
	{ "no-gbt", 0, NULL, 1011 },
	{ "no-getwork", 0, NULL, 1010 },
	{ "no-longpoll", 0, NULL, 1003 },
	{ "no-stratum", 0, NULL, 1007 },
	{ "pass", 1, NULL, 'p' },
	{ "protocol-dump", 0, NULL, 'P' },
	{ "proxy", 1, NULL, 'x' },
	{ "quiet", 0, NULL, 'q' },
	{ "retries", 1, NULL, 'r' },
	{ "retry-pause", 1, NULL, 'R' },
	{ "scantime", 1, NULL, 's' },
#ifdef HAVE_SYSLOG_H
	{ "syslog", 0, NULL, 'S' },
#endif
	{ "threads", 1, NULL, 't' },
	{ "vote", 1, NULL, 'v' },
	{ "trust-pool", 0, NULL, 'm' },
	{ "timeout", 1, NULL, 'T' },
	{ "url", 1, NULL, 'o' },
	{ "user", 1, NULL, 'u' },
	{ "userpass", 1, NULL, 'O' },
	{ "version", 0, NULL, 'V' },
	{ "devices", 1, NULL, 'd' },
	{ "throughput", 1, NULL, 'F'},
	{ "diff", 1, NULL, 'f' },
	{ 0, 0, 0, 0 }
};

struct work {

	union {
		uint16_t data16[64];
		uint32_t data[32];
		uint64_t data64[16];
	};
	uint32_t target[8];
	uint32_t maxvote;
	uint32_t hash[8];
int height;
char *txs;
char *workid;
	char job_id[128];
	size_t xnonce2_len;
	unsigned char xnonce2[32];
};
/*
struct work7 {
	CBlockHeader 	data;
	uint32_t	target[8],hash[8];
};
*/
static struct work g_work;
static time_t g_work_time;
static pthread_mutex_t g_work_lock;

static inline void work_free(struct work *w)
{
	free(w->txs);
	free(w->workid);
	free(w->job_id);
	free(w->xnonce2);
}

static inline void work_copy(struct work *dest, const struct work *src)
{
	memcpy(dest, src, sizeof(struct work));
	if (src->txs)
		dest->txs = strdup(src->txs);
	if (src->workid)
		dest->workid = strdup(src->workid);
//	if (src->job_id)
//		dest->job_id = strdup(src->job_id);
//	if (src->xnonce2) {
//		dest->xnonce2 = (unsigned char*) malloc(src->xnonce2_len);
//		memcpy(dest->xnonce2, src->xnonce2, src->xnonce2_len);
//	}
}


static bool jobj_binary(const json_t *obj, const char *key,
			void *buf, size_t buflen)
{
	const char *hexstr;
	json_t *tmp;

	tmp = json_object_get(obj, key);
	if (unlikely(!tmp)) {
		applog(LOG_ERR, "JSON key '%s' not found", key);
		return false;
	}
	hexstr = json_string_value(tmp);
	if (unlikely(!hexstr)) {
		applog(LOG_ERR, "JSON key '%s' is not a string", key);
		return false;
	}
	if (!hex2bin((unsigned char*)buf, hexstr, buflen))
		return false;

	return true;
}

static bool work_decode(const json_t *val, struct work *work)
{
	int i;
	if (opt_algo == ALGO_M7) {
		// printf("\n work decode \n");

	if (unlikely(!jobj_binary(val, "data", work->data, 122))) {
		applog(LOG_ERR, "JSON invalid data");
		goto err_out;
	}
	if (unlikely(!jobj_binary(val, "target", work->target, sizeof(work->target)))) {
		applog(LOG_ERR, "JSON invalid target");
		goto err_out;
	}
	
 
	} else {
	if (unlikely(!jobj_binary(val, "data", work->data, (opt_algo==ALGO_NEOSCRYPT)?84:sizeof(work->data)))) {
		applog(LOG_ERR, "JSON inval data fucked up");
		goto err_out;
	}
	if (unlikely(!jobj_binary(val, "target", work->target, sizeof(work->target)))) {
		applog(LOG_ERR, "JSON inval target");
		goto err_out;
	}
	if (opt_algo == ALGO_HEAVY) {
		if (unlikely(!jobj_binary(val, "maxvote", &work->maxvote, sizeof(work->maxvote)))) {
			work->maxvote = 1024;
		}
	} else work->maxvote = 0;
int data_size = (opt_algo == ALGO_NEOSCRYPT) ? 21 : ARRAY_SIZE(work->data);
	for (i = 0; i < data_size; i++)
		work->data[i] = le32dec(work->data + i);
	for (i = 0; i < ARRAY_SIZE(work->target); i++)
		work->target[i] = le32dec(work->target + i);
	}
	return true;

err_out:
	return false;
}

#define BLOCK_VERSION_MASK 0x000000ff
#define BLOCK_VERSION_CURRENT 3

static bool gbt_work_decode(const json_t *val, struct work *work)
{
	int i, n;
	uint32_t version, curtime, bits;
	uint32_t prevhash[8];
	uint32_t target[8];
	int cbtx_size;
	unsigned char *cbtx = NULL;
	int tx_count, tx_size;
	unsigned char txc_vi[9];
	unsigned char(*merkle_tree)[32] = NULL;
	bool coinbase_append = false;
	bool submit_coinbase = false;
	bool version_force = false;
	bool version_reduce = false;
	json_t *tmp, *txa;
	bool rc = false;

	tmp = json_object_get(val, "mutable");
	if (tmp && json_is_array(tmp)) {
		n = json_array_size(tmp);
		for (i = 0; i < n; i++) {
			const char *s = json_string_value(json_array_get(tmp, i));
			if (!s)
				continue;
			if (!strcmp(s, "coinbase/append"))
				coinbase_append = true;
			else if (!strcmp(s, "submit/coinbase"))
				submit_coinbase = true;
			else if (!strcmp(s, "version/force"))
				version_force = true;
			else if (!strcmp(s, "version/reduce"))
				version_reduce = true;
		}
	}

	tmp = json_object_get(val, "height");
	if (!tmp || !json_is_integer(tmp)) {
		applog(LOG_ERR, "JSON invalid height");
		goto out;
	}
	work->height = json_integer_value(tmp);

	tmp = json_object_get(val, "version");
	if (!tmp || !json_is_integer(tmp)) {
		applog(LOG_ERR, "JSON invalid version");
		goto out;
	}
	version = json_integer_value(tmp);
	if ((version & BLOCK_VERSION_MASK) > BLOCK_VERSION_CURRENT) {
		if (version_reduce) {
			version = (version & ~BLOCK_VERSION_MASK) | BLOCK_VERSION_CURRENT;
		}
		else if (!version_force) {
			applog(LOG_ERR, "Unrecognized block version: %u", version);
			goto out;
		}
	}

	if (unlikely(!jobj_binary(val, "previousblockhash", prevhash, sizeof(prevhash)))) {
		applog(LOG_ERR, "JSON invalid previousblockhash");
		goto out;
	}

	tmp = json_object_get(val, "curtime");
	if (!tmp || !json_is_integer(tmp)) {
		applog(LOG_ERR, "JSON invalid curtime");
		goto out;
	}
	curtime = json_integer_value(tmp);

	if (unlikely(!jobj_binary(val, "bits", &bits, sizeof(bits)))) {
		applog(LOG_ERR, "JSON invalid bits");
		goto out;
	}

	/* find count and size of transactions */
	txa = json_object_get(val, "transactions");
	if (!txa || !json_is_array(txa)) {
		applog(LOG_ERR, "JSON invalid transactions");
		goto out;
	}
	tx_count = json_array_size(txa);
	tx_size = 0;
	for (i = 0; i < tx_count; i++) {
		const json_t *tx = json_array_get(txa, i);
		const char *tx_hex = json_string_value(json_object_get(tx, "data"));
		if (!tx_hex) {
			applog(LOG_ERR, "JSON invalid transactions");
			goto out;
		}
		tx_size += strlen(tx_hex) / 2;
	}

	/* build coinbase transaction */
	tmp = json_object_get(val, "coinbasetxn");
	if (tmp) {
		const char *cbtx_hex = json_string_value(json_object_get(tmp, "data"));
		cbtx_size = cbtx_hex ? strlen(cbtx_hex) / 2 : 0;
		cbtx = (unsigned char*)malloc(cbtx_size + 100);
		if (cbtx_size < 60 || !hex2bin(cbtx, cbtx_hex, cbtx_size)) {
			applog(LOG_ERR, "JSON invalid coinbasetxn");
			goto out;
		}
	}
	else {
		int64_t cbvalue;
		if (!pk_script_size) {
			if (allow_getwork) {
				applog(LOG_INFO, "No payout address provided, switching to getwork");
				have_gbt = false;
			}
			else
				applog(LOG_ERR, "No payout address provided");
			goto out;
		}
		tmp = json_object_get(val, "coinbasevalue");
		if (!tmp || !json_is_number(tmp)) {
			applog(LOG_ERR, "JSON invalid coinbasevalue");
			goto out;
		}
		cbvalue = json_is_integer(tmp) ? json_integer_value(tmp) : json_number_value(tmp);
		cbtx = (unsigned char*) malloc(256);
		le32enc((uint32_t *)cbtx, 1); /* version */
		cbtx[4] = 1; /* in-counter */
		memset(cbtx + 5, 0x00, 32); /* prev txout hash */
		le32enc((uint32_t *)(cbtx + 37), 0xffffffff); /* prev txout index */
		cbtx_size = 43;
		/* BIP 34: height in coinbase */
		for (n = work->height; n; n >>= 8)
			cbtx[cbtx_size++] = n & 0xff;
		cbtx[42] = cbtx_size - 43;
		cbtx[41] = cbtx_size - 42; /* scriptsig length */
		le32enc((uint32_t *)(cbtx + cbtx_size), 0xffffffff); /* sequence */
		cbtx_size += 4;
		cbtx[cbtx_size++] = 1; /* out-counter */
		le32enc((uint32_t *)(cbtx + cbtx_size), (uint32_t)cbvalue); /* value */
		le32enc((uint32_t *)(cbtx + cbtx_size + 4), cbvalue >> 32);
		cbtx_size += 8;
		cbtx[cbtx_size++] = pk_script_size; /* txout-script length */
		memcpy(cbtx + cbtx_size, pk_script, pk_script_size);
		cbtx_size += pk_script_size;
		le32enc((uint32_t *)(cbtx + cbtx_size), 0); /* lock time */
		cbtx_size += 4;
		coinbase_append = true;
	}
	if (coinbase_append) {
		unsigned char xsig[100];
		int xsig_len = 0;
		if (*coinbase_sig) {
			n = strlen(coinbase_sig);
			if (cbtx[41] + xsig_len + n <= 100) {
				memcpy(xsig + xsig_len, coinbase_sig, n);
				xsig_len += n;
			}
			else {
				applog(LOG_WARNING, "Signature does not fit in coinbase, skipping");
			}
		}
		tmp = json_object_get(val, "coinbaseaux");
		if (tmp && json_is_object(tmp)) {
			void *iter = json_object_iter(tmp);
			while (iter) {
				unsigned char buf[100];
				const char *s = json_string_value(json_object_iter_value(iter));
				n = s ? strlen(s) / 2 : 0;
				if (!s || n > 100 || !hex2bin(buf, s, n)) {
					applog(LOG_ERR, "JSON invalid coinbaseaux");
					break;
				}
				if (cbtx[41] + xsig_len + n <= 100) {
					memcpy(xsig + xsig_len, buf, n);
					xsig_len += n;
				}
				iter = json_object_iter_next(tmp, iter);
			}
		}
		if (xsig_len) {
			unsigned char *ssig_end = cbtx + 42 + cbtx[41];
			int push_len = cbtx[41] + xsig_len < 76 ? 1 :
				cbtx[41] + 2 + xsig_len > 100 ? 0 : 2;
			n = xsig_len + push_len;
			memmove(ssig_end + n, ssig_end, cbtx_size - 42 - cbtx[41]);
			cbtx[41] += n;
			if (push_len == 2)
				*(ssig_end++) = 0x4c; /* OP_PUSHDATA1 */
			if (push_len)
				*(ssig_end++) = xsig_len;
			memcpy(ssig_end, xsig, xsig_len);
			cbtx_size += n;
		}
	}

	n = varint_encode(txc_vi, 1 + tx_count);
	work->txs = (char*)malloc(2 * (n + cbtx_size + tx_size) + 1);
	abin2hex(work->txs, txc_vi, n);
	abin2hex(work->txs + 2 * n, cbtx, cbtx_size);

	/* generate merkle root */
	merkle_tree = (unsigned char(*)[32]) malloc(32 * ((1 + tx_count + 1) & ~1));
	sha256d(merkle_tree[0], cbtx, cbtx_size);
	for (i = 0; i < tx_count; i++) {
		tmp = json_array_get(txa, i);
		const char *tx_hex = json_string_value(json_object_get(tmp, "data"));
		const int tx_size = tx_hex ? strlen(tx_hex) / 2 : 0;
		unsigned char *tx = (unsigned char*)malloc(tx_size);
		if (!tx_hex || !hex2bin(tx, tx_hex, tx_size)) {
			applog(LOG_ERR, "JSON invalid transactions");
			free(tx);
			goto out;
		}
		sha256d(merkle_tree[1 + i], tx, tx_size);
		if (!submit_coinbase)
			strcat(work->txs, tx_hex);
	}
	n = 1 + tx_count;
	while (n > 1) {
		if (n % 2) {
			memcpy(merkle_tree[n], merkle_tree[n - 1], 32);
			++n;
		}
		n /= 2;
		for (i = 0; i < n; i++)
			sha256d(merkle_tree[i], merkle_tree[2 * i], 64);
	}

	/* assemble block header */
	work->data[0] = swab32(version);
	for (i = 0; i < 8; i++)
		work->data[8 - i] = le32dec(prevhash + i);
	for (i = 0; i < 8; i++)
		work->data[9 + i] = be32dec((uint32_t *)merkle_tree[0] + i);
	work->data[17] = swab32(curtime);
	work->data[18] = le32dec(&bits);
	memset(work->data + 19, 0x00, 52);
	work->data[20] = 0x80000000;
	work->data[31] = 0x00000280;

	if (unlikely(!jobj_binary(val, "target", target, sizeof(target)))) {
		applog(LOG_ERR, "JSON invalid target");
		goto out;
	}
	for (i = 0; i < ARRAY_SIZE(work->target); i++)
		work->target[7 - i] = be32dec(target + i);

	tmp = json_object_get(val, "workid");
	if (tmp) {
		if (!json_is_string(tmp)) {
			applog(LOG_ERR, "JSON invalid workid");
			goto out;
		}
		work->workid = strdup(json_string_value(tmp));
	}

	/* Long polling */
	tmp = json_object_get(val, "longpollid");
	if (want_longpoll && json_is_string(tmp)) {
		free(lp_id);
		lp_id = strdup(json_string_value(tmp));
		if (!have_longpoll) {
			char *lp_uri;
			tmp = json_object_get(val, "longpolluri");
			lp_uri = json_is_string(tmp) ? strdup(json_string_value(tmp)) : rpc_url;
			have_longpoll = true;
			tq_push(thr_info[longpoll_thr_id].q, lp_uri);
		}
	}

	rc = true;

out:
	free(cbtx);
	free(merkle_tree);
	return rc;
}


/*
static void share_result(int result, const char *reason)
{
	char s[345];
	double hashrate;
	int i;

	hashrate = 0.;
	pthread_mutex_lock(&stats_lock);
	for (i = 0; i < opt_n_threads; i++)
		hashrate += thr_hashrates[i];
	result ? accepted_count++ : rejected_count++;
	pthread_mutex_unlock(&stats_lock);
	
	sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.2f", 1e-3 * hashrate);
	applog(LOG_INFO, "accepted: %lu/%lu (%.2f%%), %s khash/s %s",
		   accepted_count,
		   accepted_count + rejected_count,
		   100. * accepted_count / (accepted_count + rejected_count),
		   s,
		   result ? "(yay!!!)" : "(booooo)");

	if (opt_debug && reason)
		applog(LOG_DEBUG, "DEBUG: reject reason: %s", reason);
}
*/
int hashratessize=250;
double hashrates [250]= { }; 
double totalhashrate = 0.;
double totalhashsquare =0.;
int hashcomplete=0;
int hashrow=0;
static void share_result(int result, const char *reason)
{
	char s[345];
	char s1[345];
	char s2[345];
	double hashrate;
	int i;
	double averagehashrate=0.;
	double avsquare=0.;
	double stddev=0.;
	hashrate = 0.;
	pthread_mutex_lock(&stats_lock);
	for (i = 0; i < opt_n_threads; i++)
		hashrate += thr_hashrates[i];
	result ? accepted_count++ : rejected_count++;
	pthread_mutex_unlock(&stats_lock);
	
	sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.2f", 1e-3 * hashrate);	
	totalhashrate+=(double) hashrate;
	totalhashsquare+=pow((double)hashrate,2);
	hashrow++;
	averagehashrate=totalhashrate/(double)hashrow;
	avsquare=totalhashsquare/(double)hashrow;
	stddev = sqrt(avsquare-pow(averagehashrate,2));
	sprintf(s1, hashrate >= 1e6 ? "%.0f" : "%.2f", 1e-3 * averagehashrate);
	sprintf(s2, hashrate >= 1e6 ? "%.0f" : "%.2f", 1e-3 * stddev);
	
		applog(LOG_INFO, "accepted: %lu/%lu (%.2f%%), %s kh/s (%s +/- %s) %s",
				accepted_count,
				accepted_count + rejected_count,
				100. * accepted_count / (accepted_count + rejected_count),
				s,s1,s2, result ? "(yay!!!)" : "(booooo)");

	if (opt_debug && reason)
		applog(LOG_DEBUG, "DEBUG: reject reason: %s", reason);
	
}

static bool submit_upstream_work(CURL *curl, struct work *work)
{
	char *str = NULL;
	json_t *val, *res, *reason;
	char data_str[2 * sizeof(work->data) + 1];
	char s[345];
	int i;
	bool rc = false;

	/* pass if the previous hash is not the current previous hash */
	if (opt_algo == ALGO_M7) {
		if (memcmp(work->data , g_work.data , 96)) {
			if (opt_debug)
				applog(LOG_DEBUG, "DEBUG: stale work detected, discarding");
			return true;
		} 
	} else {
	if (memcmp(work->data + 1, g_work.data + 1, 32)) {
		if (opt_debug)
			applog(LOG_DEBUG, "DEBUG: stale work detected, discarding");
		return true;
	}
	}
	if (have_stratum) {
		if (opt_algo == ALGO_M7) {
			
			uint64_t ntime, nonce;
			char *ntimestr, *noncestr, *xnonce2str;

			be64enc(&ntime, work->data64[12]);
			be32enc(&nonce, work->data[29]);
			ntimestr=bin2hex((const unsigned char *)(&ntime), 8);
			noncestr=bin2hex((const unsigned char *)(&nonce), 4);
			xnonce2str = bin2hex(work->xnonce2, work->xnonce2_len);
			sprintf(s,
				"{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}",
				rpc_user, work->job_id, xnonce2str, ntimestr, noncestr);
			free(xnonce2str);
		} else {
		uint32_t ntime, nonce;
		uint16_t nvote;
		char *ntimestr, *noncestr, *xnonce2str, *nvotestr;

		le32enc(&ntime, work->data[17]);
		le32enc(&nonce, work->data[19]);
		be16enc(&nvote, *((uint16_t*)&work->data[20]));

		ntimestr = bin2hex((const unsigned char *)(&ntime), 4);
		noncestr = bin2hex((const unsigned char *)(&nonce), 4);
		xnonce2str = bin2hex(work->xnonce2, work->xnonce2_len);
		nvotestr = bin2hex((const unsigned char *)(&nvote), 2);
		if (opt_algo == ALGO_HEAVY) {
			sprintf(s,
				"{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}",
				rpc_user, work->job_id, xnonce2str, ntimestr, noncestr, nvotestr);
		} else {
			sprintf(s,
				"{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}",
				rpc_user, work->job_id, xnonce2str, ntimestr, noncestr);
		}
		free(ntimestr);
		free(noncestr);
		free(xnonce2str);
		free(nvotestr);
		}
		if (unlikely(!stratum_send_line(&stratum, s))) {
			applog(LOG_ERR, "submit_upstream_work stratum_send_line failed");
			goto out;
		}
// gbt
	}
	else if (work->txs) {
		char *req;

		for (i = 0; i < ARRAY_SIZE(work->data); i++)
			be32enc(work->data + i, work->data[i]);
		abin2hex(data_str, (unsigned char *)work->data, 80);
		if (work->workid) {
			char *params;
			val = json_object();
			json_object_set_new(val, "workid", json_string(work->workid));
			params = json_dumps(val, 0);
			json_decref(val);
			req = (char*)malloc(128 + 2 * 80 + strlen(work->txs) + strlen(params));
			sprintf(req,
				"{\"method\": \"submitblock\", \"params\": [\"%s%s\", %s], \"id\":1}\r\n",
				data_str, work->txs, params);
			free(params);
		}
		else {
			req = (char*)malloc(128 + 2 * 80 + strlen(work->txs));
			sprintf(req,
				"{\"method\": \"submitblock\", \"params\": [\"%s%s\"], \"id\":1}\r\n",
				data_str, work->txs);
		}
		val = json_rpc_call2(curl, rpc_url, rpc_userpass, req, NULL, 0);
		free(req);
		if (unlikely(!val)) {
			applog(LOG_ERR, "submit_upstream_work json_rpc_call failed");
			goto out;
		}

		res = json_object_get(val, "result");
		if (json_is_object(res)) {
			char *res_str;
			bool sumres = false;
			void *iter = json_object_iter(res);
			while (iter) {
				if (json_is_null(json_object_iter_value(iter))) {
					sumres = true;
					break;
				}
				iter = json_object_iter_next(res, iter);
			}
			res_str = json_dumps(res, 0);
			share_result(sumres, res_str);
			free(res_str);
		}
		else
			share_result(json_is_null(res), json_string_value(res));

		json_decref(val);
///
	} else {

		/* build hex string */
		if (opt_algo != ALGO_M7) {
		if (opt_algo != ALGO_HEAVY && opt_algo != ALGO_MJOLLNIR && opt_algo) {
			int data_size = (opt_algo == ALGO_NEOSCRYPT) ? 80 : sizeof(work->data);			
			for (i = 0; i < (data_size >>2); i++)
				le32enc(work->data + i, work->data[i]);
			}
		int data_size = (opt_algo == ALGO_NEOSCRYPT) ? 80 : sizeof(work->data);
		str = bin2hex((unsigned char *)work->data,data_size);
			if (unlikely(!str)) {
				applog(LOG_ERR, "submit_upstream_work OOM");
				goto out;
			}
		} else {
			
			
			abin2hex(data_str,(unsigned char *)work->data, 122);
			if (unlikely(!data_str)) {
				applog(LOG_ERR, "submit_upstream_work OOM");
				goto out;
			}
		} // M7

			if (opt_algo == ALGO_M7) {
        sprintf(s,
			"{\"method\": \"getwork\", \"params\": [ \"%s\" ], \"id\":1}\r\n",
			data_str);

			} else {
		/* build JSON-RPC request */
		sprintf(s,
			"{\"method\": \"getwork\", \"params\": [ \"%s\" ], \"id\":1}\r\n",
			str);
			}
		/* issue JSON-RPC request */
		val = json_rpc_call(curl, rpc_url, rpc_userpass, s, false, false, NULL);
		if (unlikely(!val)) {
			applog(LOG_ERR, "submit_upstream_work json_rpc_call failed");
			goto out;
		}

		res = json_object_get(val, "result");
		reason = json_object_get(val, "reject-reason");
		share_result(json_is_true(res), reason ? json_string_value(reason) : NULL);

		json_decref(val);
	}

	rc = true;

out:
	free(str);
	return rc;
}

static const char *rpc_req =
	"{\"method\": \"getwork\", \"params\": [], \"id\":0}\r\n";

#define GBT_CAPABILITIES "[\"coinbasetxn\", \"coinbasevalue\", \"longpoll\", \"workid\"]"
static const char *gbt_req =
"{\"method\": \"getblocktemplate\", \"params\": [{\"capabilities\": "
GBT_CAPABILITIES "}], \"id\":0}\r\n";
static const char *gbt_lp_req =
"{\"method\": \"getblocktemplate\", \"params\": [{\"capabilities\": "
GBT_CAPABILITIES ", \"longpollid\": \"%s\"}], \"id\":0}\r\n";


static bool get_upstream_work(CURL *curl, struct work *work)
{
	json_t *val;
	bool rc;
	int err;
	struct timeval tv_start, tv_end, diff;
start:
	gettimeofday(&tv_start, NULL);
	val = json_rpc_call2(curl, rpc_url, rpc_userpass,
		have_gbt ? gbt_req : rpc_req,
		&err, have_gbt ? JSON_RPC_QUIET_404 : 0);
//	val = json_rpc_call(curl, rpc_url, rpc_userpass, rpc_req,
//			    want_longpoll, false, NULL);
	gettimeofday(&tv_end, NULL);
		
	if (have_stratum) {
		if (val)
			json_decref(val);
		return true;
	}

	if (!have_gbt && !allow_getwork) {
		applog(LOG_ERR, "No usable protocol");
		if (val)
			json_decref(val);
		return false;
	}

	if (have_gbt && allow_getwork && !val && err == CURLE_OK) {
		applog(LOG_INFO, "getblocktemplate failed, falling back to getwork");
		have_gbt = false;
		goto start;
	}

	if (!val)
		return false;

	if (have_gbt) {
		rc = gbt_work_decode(json_object_get(val, "result"), work);
		if (!have_gbt) {
			json_decref(val);
			goto start;
		}
	}	else {
	rc = work_decode(json_object_get(val, "result"), work);
    }
	if (opt_debug && rc) {
		timeval_subtract(&diff, &tv_end, &tv_start);
		applog(LOG_DEBUG, "DEBUG: got new work in %d ms",
		       diff.tv_sec * 1000 + diff.tv_usec / 1000);
	}

	json_decref(val);

	return rc;
}

static void workio_cmd_free(struct workio_cmd *wc)
{
	if (!wc)
		return;

	switch (wc->cmd) {
	case WC_SUBMIT_WORK:
//		work_free(wc->u.work);
		free(wc->u.work);
		break;
	default: /* do nothing */
		break;
	}

	memset(wc, 0, sizeof(*wc));	/* poison */
	free(wc);
}

static bool workio_get_work(struct workio_cmd *wc, CURL *curl)
{
	struct work *ret_work;
	int failures = 0;


	ret_work = (struct work*)calloc(1, sizeof(*ret_work));
	if (!ret_work)
		return false;

	/* obtain new work from bitcoin via JSON-RPC */
	while (!get_upstream_work(curl, ret_work)) {
		if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
			applog(LOG_ERR, "json_rpc_call failed, terminating workio thread");
			free(ret_work);
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "json_rpc_call failed, retry after %d seconds",
			opt_fail_pause);
		sleep(opt_fail_pause);
	}

	/* send work to requesting thread */
	if (!tq_push(wc->thr->q, ret_work))
		free(ret_work);

	return true;
}

static bool workio_submit_work(struct workio_cmd *wc, CURL *curl)
{
	int failures = 0;

	/* submit solution to bitcoin via JSON-RPC */
	while (!submit_upstream_work(curl, wc->u.work)) {
		if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
			applog(LOG_ERR, "...terminating workio thread");
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "...retry after %d seconds",
			opt_fail_pause);
		sleep(opt_fail_pause);
	}

	return true;
}

static void *workio_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info*)userdata;
	CURL *curl;
	bool ok = true;

	curl = curl_easy_init();
	if (unlikely(!curl)) {
		applog(LOG_ERR, "CURL initialization failed");
		return NULL;
	}
	// printf("workio thread\n");
	while (ok) {
		struct workio_cmd *wc;

		/* wait for workio_cmd sent to us, on our queue */
		wc = (struct workio_cmd *)tq_pop(mythr->q, NULL);
		if (!wc) {
			ok = false;
			break;
		}

		/* process workio_cmd */
		switch (wc->cmd) {
		case WC_GET_WORK:
			ok = workio_get_work(wc, curl);
			break;
		case WC_SUBMIT_WORK:
			ok = workio_submit_work(wc, curl);
			break;

		default:		/* should never happen */
			ok = false;
			break;
		}

		workio_cmd_free(wc);
	}

	tq_freeze(mythr->q);
	curl_easy_cleanup(curl);

	return NULL;
}

static bool get_work(struct thr_info *thr, struct work *work)
{
	struct workio_cmd *wc;
	struct work *work_heap;

	if (opt_benchmark) {
		memset(work->data, 0x55, 76);
		work->data[17] = swab32((uint32_t)time(NULL));
		memset(work->data + 19, 0x00, 52);
		work->data[20] = 0x80000000;
		work->data[31] = 0x00000280;
		memset(work->target, 0x00, sizeof(work->target));
		return true;
	}
	
	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if (!wc)
		return false;

	wc->cmd = WC_GET_WORK;
	wc->thr = thr;
	
	/* send work request to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc)) {
		workio_cmd_free(wc);
		return false;
	}
	
	/* wait for response, a unit of work */
	work_heap = (struct work *)tq_pop(thr->q, NULL);
	if (!work_heap)
		return false;

	/* copy returned work into storage provided by caller */
	memcpy(work, work_heap, sizeof(*work));
	free(work_heap);
	// printf("getwork 4\n");
	return true;
}

static bool submit_work(struct thr_info *thr, const struct work *work_in)
{
	struct workio_cmd *wc;
	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if (!wc)
		return false;

	wc->u.work = (struct work *)malloc(sizeof(*work_in));
	if (!wc->u.work)
		goto err_out;

	wc->cmd = WC_SUBMIT_WORK;
	wc->thr = thr;
//	memcpy(wc->u.work, work_in, sizeof(*work_in));
	work_copy(wc->u.work, work_in);

	/* send solution to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc))
		goto err_out;

	return true;

err_out:
	workio_cmd_free(wc);
	return false;
}

static void stratum_gen_work(struct stratum_ctx *sctx, struct work *work)
{
	unsigned char merkle_root[64];
	int i;
	// printf("\n stratum_gen_work\n ");
	pthread_mutex_lock(&sctx->work_lock);

	strcpy(work->job_id, sctx->job.job_id);
	work->xnonce2_len = sctx->xnonce2_size;
	memcpy(work->xnonce2, sctx->job.xnonce2, sctx->xnonce2_size);

	/* Generate merkle root */
	if (opt_algo == ALGO_HEAVY || opt_algo == ALGO_MJOLLNIR)
		heavycoin_hash(merkle_root, sctx->job.coinbase, (int)sctx->job.coinbase_size);
	else
	if (opt_algo == ALGO_FUGUE256 || opt_algo == ALGO_GROESTL || opt_algo == ALGO_WH || opt_algo == ALGO_KECCAK )
		SHA256((unsigned char*)sctx->job.coinbase, sctx->job.coinbase_size, (unsigned char*)merkle_root);
	else
		sha256d(merkle_root, sctx->job.coinbase, (int)sctx->job.coinbase_size);

	for (i = 0; i < sctx->job.merkle_count; i++) {
		memcpy(merkle_root + 32, sctx->job.merkle[i], 32);
		if (opt_algo == ALGO_HEAVY || opt_algo == ALGO_MJOLLNIR)
			heavycoin_hash(merkle_root, merkle_root, 64);
		else
			sha256d(merkle_root, merkle_root, 64);
	}
	
	/* Increment extranonce2 */
	for (i = 0; i < (int)sctx->xnonce2_size && !++sctx->job.xnonce2[i]; i++);

	/* Assemble block header */
	memset(work->data, 0, 128);
	work->data[0] = le32dec(sctx->job.version);
	for (i = 0; i < 8; i++)
		work->data[1 + i] = le32dec((uint32_t *)sctx->job.prevhash + i);
	for (i = 0; i < 8; i++)
		work->data[9 + i] = be32dec((uint32_t *)merkle_root + i);
	work->data[17] = le32dec(sctx->job.ntime);
	work->data[18] = le32dec(sctx->job.nbits);
	if (opt_algo == ALGO_MJOLLNIR)
	{
		for (i = 0; i < 20; i++)
			work->data[i] = be32dec((uint32_t *)&work->data[i]);
	}

	work->data[20] = 0x80000000;
	work->data[31] = (opt_algo == ALGO_MJOLLNIR) ? 0x000002A0 : 0x00000280;

	// HeavyCoin
	if (opt_algo == ALGO_HEAVY) {
		uint16_t *ext;
		work->maxvote = 1024;
		ext = (uint16_t*)(&work->data[20]);
		ext[0] = opt_vote;
		ext[1] = be16dec(sctx->job.nreward);

		for (i = 0; i < 20; i++)
			work->data[i] = be32dec((uint32_t *)&work->data[i]);
	}
	//

	pthread_mutex_unlock(&sctx->work_lock);

	if (opt_debug) {
		char *xnonce2str = bin2hex(work->xnonce2, sctx->xnonce2_size);
		applog(LOG_DEBUG, "DEBUG: job_id='%s' extranonce2=%s ntime=%08x",
		       work->job_id, xnonce2str, swab32(work->data[17]));
		free(xnonce2str);
	} 
	
	if (opt_algo == ALGO_JACKPOT || opt_algo == ALGO_NEOSCRYPT || opt_algo == ALGO_PLUCK)
		diff_to_target(work->target, sctx->job.diff / (65536.0 * opt_difficulty));
	else if (opt_algo == ALGO_FUGUE256 || opt_algo == ALGO_GROESTL || opt_algo == ALGO_DMD_GR || opt_algo == ALGO_FRESH)
		diff_to_target(work->target, sctx->job.diff / (256.0 * opt_difficulty));
    else if (opt_algo == ALGO_KECCAK ) // || opt_algo == ALGO_LYRA)
		diff_to_target(work->target, sctx->job.diff / (128.0 * opt_difficulty));  // seems to work best, minimize rejected share
	else
		diff_to_target(work->target, sctx->job.diff / opt_difficulty);
}

static void stratum_gen_work_m7(struct stratum_ctx *sctx, struct work *work)
{

	pthread_mutex_lock(&sctx->work_lock);
	strcpy(work->job_id, sctx->job.job_id);
	work->xnonce2_len = sctx->xnonce2_size;
	memcpy(work->xnonce2, sctx->job.xnonce2, sctx->xnonce2_size);

	/* Increment extranonce2 */
	for (int i = 0; i < (int) sctx->xnonce2_size && !++sctx->job.xnonce2[i]; i++);

	/* Assemble block header */
	memset(work->data, 0, 122);
	memcpy(work->data, sctx->job.m7prevblock, 32);
	memcpy(work->data + 8, sctx->job.m7accroot, 32);
	memcpy(work->data + 16, sctx->job.m7merkleroot, 32);
	work->data64[12] = be64dec(sctx->job.m7ntime);
	work->data64[13] = be64dec(sctx->job.m7height);
	unsigned char *xnonce_ptr = (unsigned char *)(work->data + 28);
	for (int i = 0; i < (int) sctx->xnonce1_size; i++) {
		*(xnonce_ptr + i) = sctx->xnonce1[i];
	}
	for (int i = 0; i < (int) work->xnonce2_len; i++) { 
		*(xnonce_ptr + sctx->xnonce1_size + i) = work->xnonce2[i];
	}
	work->data16[60] = be16dec(sctx->job.m7version);

	pthread_mutex_unlock(&sctx->work_lock);

	diff_to_target(work->target, sctx->job.diff / (65536.0* opt_difficulty));

	if (opt_debug) {
		char data_str[245], target_str[65];
		abin2hex(data_str, (unsigned char *)work->data, 122);
		applog(LOG_DEBUG, "DEBUG: stratum_gen_work data %s", data_str);
		abin2hex(target_str, (unsigned char *)work->target, 32);
		applog(LOG_DEBUG, "DEBUG: stratum_gen_work target %s", target_str);
	}
}

static void *miner_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	int thr_id = mythr->id;
	struct work work;
	uint32_t max_nonce;
	uint32_t end_nonce = (0xffffffffU) / opt_n_threads * (thr_id + 1) - 0x20;
	unsigned char *scratchbuf = NULL;
	char s[16];

    static int rounds = 0;
	
	memset(&work, 0, sizeof(work)); // prevent work from being used uninitialized

	/* Set worker threads to nice 19 and then preferentially to SCHED_IDLE
	 * and if that fails, then SCHED_BATCH. No need for this to be an
	 * error if it fails */
	if (!opt_benchmark) {
		setpriority(PRIO_PROCESS, 0, 19);
		drop_policy();
	}

	/* Cpu affinity only makes sense if the number of threads is a multiple
	 * of the number of CPUs */
	if (num_processors > 1 && opt_n_threads % num_processors == 0) {
		if (!opt_quiet)
			applog(LOG_INFO, "Binding thread %d to cpu %d",
			       thr_id, thr_id % num_processors);
		affine_to_cpu(thr_id, thr_id % num_processors);
	}
	// printf("\n miner threads 2\n");
	while (1) {
		unsigned long hashes_done;
		
		struct timeval tv_start, tv_end, diff;
		int64_t max64;
		int rc;

		if (have_stratum) {
			
			while (time(NULL) >= g_work_time + 120)
				sleep(1);
			pthread_mutex_lock(&g_work_lock);
       bool nonce_over;
			if (opt_algo == ALGO_M7) {
				nonce_over = work.data[29] >= end_nonce;
			} else {
				nonce_over = work.data[19] >= end_nonce;
			}
		//	printf("nonce over %d\n",nonce_over);
       if (opt_algo == ALGO_M7) {		       
			if (work.data[29] >= end_nonce && !memcmp(work.data, g_work.data, 116))
					stratum_gen_work_m7(&stratum, &g_work);
				
			} else {
				
				if (work.data[19] >= end_nonce && !memcmp(work.data, g_work.data, 76))		
					stratum_gen_work(&stratum, &g_work);
			}
		} else {
			int min_scantime = have_longpoll ? LP_SCANTIME : opt_scantime;
			/* obtain new work from internal workio thread */
			pthread_mutex_lock(&g_work_lock);
			bool nonce_over;
			if (opt_algo == ALGO_M7) {
				nonce_over = work.data[29] >= end_nonce;
			} else {
				nonce_over = work.data[19] >= end_nonce;
			}
			
			if (!have_stratum && (time(NULL) - g_work_time >= min_scantime || nonce_over)) {
				if (unlikely(!get_work(mythr, &g_work))) {
					applog(LOG_ERR, "work retrieval failed, exiting "
						"mining thread %d", mythr->id);
					pthread_mutex_unlock(&g_work_lock);
					goto out;
				}
				g_work_time = have_stratum ? 0 : time(NULL);
			}
		}
///weird stuff
/*
			if (have_stratum) {
				pthread_mutex_unlock(&g_work_lock);
				continue;
			}
*/		
		if (opt_algo == ALGO_M7) {


			if (memcmp(work.data, g_work.data, 116)) {
				memcpy(&work, &g_work, sizeof(struct work));
//				work_free(&work);
//				work_copy(&work, &g_work);
				work.data[29] = (0xffffffffU) / opt_n_threads * thr_id;				
			} else
				work.data[29]++; // todo
		} else {
		if (memcmp(work.data, g_work.data, 76)) {
			memcpy(&work, &g_work, sizeof(struct work));
//			work_free(&work);
//			work_copy(&work, &g_work);
			work.data[19] = 0xffffffffU / opt_n_threads * thr_id;
		} else
			work.data[19]++;
		}
		pthread_mutex_unlock(&g_work_lock);
		work_restart[thr_id].restart = 0;

		/* adjust max_nonce to meet target scan time */
		if (have_stratum)
			max64 = LP_SCANTIME;
		else
			max64 = g_work_time + (have_longpoll ? LP_SCANTIME : opt_scantime)
			      - time(NULL);
		max64 *= (int64_t)thr_hashrates[thr_id];
		
        if (max64 <= 0) {
			switch (opt_algo) {
			case ALGO_JACKPOT:
				max64 = 0x1fffLL;
				break;
            case ALGO_NEOSCRYPT:
            case ALGO_PLUCK:
				max64 = 0xfffLL;
				break;
			case ALGO_M7:
				max64 = 0x3ffffLL;
				break;
			default: 
				max64 = 0xfffffLL;
				break;
			}
		}
		if (opt_algo == ALGO_M7) {
			if ((int64_t) work.data[29] + max64 > (int64_t) end_nonce)
				max_nonce = end_nonce;
			else
				max_nonce = (uint32_t)(work.data[29] + max64);
		} else {
			if ((int64_t) work.data[19] + max64 > (int64_t) end_nonce) {
				max_nonce = end_nonce;}
			else {
				max_nonce = (uint32_t) (work.data[19] + max64);}
		}


		hashes_done = 0;
		gettimeofday(&tv_start, NULL);

		/* scan nonces for a proof-of-work hash */
		switch (opt_algo) {

		case ALGO_HEAVY:
			rc = scanhash_heavy(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done, work.maxvote, HEAVYCOIN_BLKHDR_SZ);
			break;

		case ALGO_MJOLLNIR:
			rc = scanhash_heavy(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done, 0, MNR_BLKHDR_SZ);
			break;

		case ALGO_FUGUE256:
			rc = scanhash_fugue256(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_GROESTL:
		case ALGO_DMD_GR:
			rc = scanhash_groestlcoin(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_MYR_GR:
			rc = scanhash_myriad(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_JACKPOT:
			rc = scanhash_jackpot(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_QUARK:
			rc = scanhash_quark(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_ANIME:
			rc = scanhash_anime(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
		case ALGO_QUBIT:
			rc = scanhash_qubit(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
        case ALGO_DOOM:
			rc = scanhash_doom(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
        case ALGO_FRESH:
			rc = scanhash_fresh(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
		case ALGO_NIST5:
			rc = scanhash_nist5(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_X11:
			rc = scanhash_x11(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_X13:
			rc = scanhash_x13(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
        case ALGO_X14:
			rc = scanhash_x14(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

        case ALGO_X15:
			rc = scanhash_x15(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

        case ALGO_X17:
			rc = scanhash_x17(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
        case ALGO_M7:

			rc = scanhash_m7(thr_id,work.data, work.target,max_nonce, &hashes_done);
			
			break;
        case ALGO_LYRA:
			rc = scanhash_lyra(thr_id,work.data, work.target,max_nonce, &hashes_done);			
			break;

		case ALGO_PLUCK:
			rc = scanhash_pluck(thr_id, work.data, work.target, max_nonce, &hashes_done);
			break;


        case ALGO_WH:
			rc = scanhash_wh(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
        case ALGO_DEEP:
			rc = scanhash_deep(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
		case ALGO_KECCAK:
			rc = scanhash_keccak256(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		default:
			/* should never happen */
			goto out;
		}


		/* record scanhash elapsed time */
		gettimeofday(&tv_end, NULL);
		

		timeval_subtract(&diff, &tv_end, &tv_start);
		if (diff.tv_usec || diff.tv_sec) {
			pthread_mutex_lock(&stats_lock);
			thr_hashrates[thr_id] =	hashes_done / (diff.tv_sec + 1e-6 * diff.tv_usec);
			pthread_mutex_unlock(&stats_lock);
		}

		if (!opt_quiet) {
			sprintf(s, thr_hashrates[thr_id] >= 1e6 ? "%.0f" : "%.2f",
				1e-3 * thr_hashrates[thr_id]);
			applog(LOG_INFO, "GPU #%d: %s, %s khash/s",
				device_map[thr_id], device_name[thr_id], s);
		}

		if (opt_benchmark && thr_id == opt_n_threads - 1) {
			double hashrate = 0.;
			int i;
			for (i = 0; i < opt_n_threads && thr_hashrates[i]; i++) 
				hashrate += thr_hashrates[i];
			if (i == opt_n_threads) {
				sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.2f", 1e-3 * hashrate);
				applog(LOG_INFO, "Total: %s khash/s", s);
			}
		}

		/* if nonce found, submit work */
		if (rc && !opt_benchmark && !submit_work(mythr, &work))
			break;
	}

out:
	tq_freeze(mythr->q);

	return NULL;
}

static void restart_threads(void)
{
	int i;

	for (i = 0; i < opt_n_threads; i++)
		work_restart[i].restart = 1;
}

static void *longpoll_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	CURL *curl = NULL;
	char *copy_start, *hdr_path = NULL, *lp_url = NULL;
	bool need_slash = false;

	curl = curl_easy_init();
	if (unlikely(!curl)) {
		applog(LOG_ERR, "CURL initialization failed");
		goto out;
	}

start:
	hdr_path = (char*)tq_pop(mythr->q, NULL);
	if (!hdr_path)
		goto out;

	/* full URL */
	if (strstr(hdr_path, "://")) {
		lp_url = hdr_path;
		hdr_path = NULL;
	}
	
	/* absolute path, on current server */
	else {
		copy_start = (*hdr_path == '/') ? (hdr_path + 1) : hdr_path;
		if (rpc_url[strlen(rpc_url) - 1] != '/')
			need_slash = true;

		lp_url = (char*)malloc(strlen(rpc_url) + strlen(copy_start) + 2);
		if (!lp_url)
			goto out;

		sprintf(lp_url, "%s%s%s", rpc_url, need_slash ? "/" : "", copy_start);
	}

	applog(LOG_INFO, "Long-polling activated for %s", lp_url);

	while (1) {
		char *req = NULL;
		json_t *val, *soval, *res;
		int err;
		if (have_gbt) {
			req = (char*)malloc(strlen(gbt_lp_req) + strlen(lp_id) + 1);
			sprintf(req, gbt_lp_req, lp_id);
		}
//		val = json_rpc_call(curl, lp_url, rpc_userpass, rpc_req,
//				    false, true, &err);

		val = json_rpc_call2(curl, lp_url, rpc_userpass,
			req ? req : rpc_req, &err,
			JSON_RPC_LONGPOLL);
		free(req);


		if (have_stratum) {
			if (val)
				json_decref(val);
			goto out;
		}
		if (likely(val)) {
			if (!opt_quiet) applog(LOG_INFO, "LONGPOLL detected new block");
			res = json_object_get(val, "result");
			soval = json_object_get(res, "submitold");
			submit_old = soval ? json_is_true(soval) : false;
			pthread_mutex_lock(&g_work_lock);
            bool rc;
			if (have_gbt)
				rc = gbt_work_decode(res, &g_work);
			else
				rc = work_decode(res, &g_work);
			if (rc) {
				time(&g_work_time);
				restart_threads();
			}
/*
			if (work_decode(json_object_get(val, "result"), &g_work)) {
				if (opt_debug)
					applog(LOG_DEBUG, "DEBUG: got new work");
				time(&g_work_time);
				restart_threads();
			}
*/
			pthread_mutex_unlock(&g_work_lock);
			json_decref(val);
		} else {
			pthread_mutex_lock(&g_work_lock);
			g_work_time -= LP_SCANTIME;
			pthread_mutex_unlock(&g_work_lock);
			if (err == CURLE_OPERATION_TIMEDOUT) {
				restart_threads();
			} else {
				have_longpoll = false;
				restart_threads();
				free(hdr_path);
				free(lp_url);
				lp_url = NULL;
				sleep(opt_fail_pause);
				goto start;
			}
		}
	}

out:
	free(hdr_path);
	free(lp_url);
	tq_freeze(mythr->q);
	if (curl)
		curl_easy_cleanup(curl);

	return NULL;
}

static bool stratum_handle_response(char *buf)
{
	json_t *val, *err_val, *res_val, *id_val;
	json_error_t err;
	bool ret = false;

	val = JSON_LOADS(buf, &err);
	if (!val) {
		applog(LOG_INFO, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");
	id_val = json_object_get(val, "id");

	if (!id_val || json_is_null(id_val) || !res_val)
		goto out;

	share_result(json_is_true(res_val),
		err_val ? json_string_value(json_array_get(err_val, 1)) : NULL);

	ret = true;
out:
	if (val)
		json_decref(val);

	return ret;
}

static void *stratum_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	char *s;
	// printf("coming here stratum thread");
	stratum.url = (char*)tq_pop(mythr->q, NULL);
	if (!stratum.url)
		goto out;
	applog(LOG_INFO, "Starting Stratum on %s", stratum.url);

	while (1) {
		int failures = 0;

		while (!stratum.curl) {
			pthread_mutex_lock(&g_work_lock);
			g_work_time = 0;
			pthread_mutex_unlock(&g_work_lock);
			restart_threads();

			if (!stratum_connect(&stratum, stratum.url) ||
			    !stratum_subscribe(&stratum) ||
			    !stratum_authorize(&stratum, rpc_user, rpc_pass)) {
				stratum_disconnect(&stratum);
				if (opt_retries >= 0 && ++failures > opt_retries) {
					applog(LOG_ERR, "...terminating workio thread");
					tq_push(thr_info[work_thr_id].q, NULL);
					goto out;
				}
				applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
				sleep(opt_fail_pause);
			}
		}

		if (stratum.job.job_id &&
		    (strcmp(stratum.job.job_id, g_work.job_id) || !g_work_time)) {
			pthread_mutex_lock(&g_work_lock);
			if (opt_algo == ALGO_M7) {
				stratum_gen_work_m7(&stratum, &g_work);
			} else {
				stratum_gen_work(&stratum, &g_work);
			}			
			time(&g_work_time);
			pthread_mutex_unlock(&g_work_lock);
			if (stratum.job.clean) {
				if (!opt_quiet) applog(LOG_INFO, "Stratum detected new block");
				restart_threads();
			}
		}
		
		if (!stratum_socket_full(&stratum, 120)) {
			applog(LOG_ERR, "Stratum connection timed out");
			s = NULL;
		} else
			s = stratum_recv_line(&stratum);
		if (!s) {
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "Stratum connection interrupted");
			continue;
		}
		if (opt_algo == ALGO_M7) {
		if (!stratum_handle_method_m7(&stratum, s))
			stratum_handle_response(s);
		} else {
		if (!stratum_handle_method(&stratum, s))
			stratum_handle_response(s);
		}
		free(s);
	}

out:
	return NULL;
}

static void show_version_and_exit(void)
{
	 printf("%s\n%s\n", PACKAGE_STRING, curl_version());
	exit(0);
}

static void show_usage_and_exit(int status)
{
	if (status)
		fprintf(stderr, "Try `" PROGRAM_NAME " --help' for more information.\n");
	else
		printf(usage);
	exit(status);
}

static void parse_arg (int key, char *arg)
{
	char *p;
	int v, i;
	double d;

	switch(key) {
	case 'a':
		for (i = 0; i < ARRAY_SIZE(algo_names); i++) {
			if (algo_names[i] &&
			    !strcmp(arg, algo_names[i])) {
				opt_algo = (sha256_algos)i;
				break;
			}
		}
		if (i == ARRAY_SIZE(algo_names))
			show_usage_and_exit(1);
		break;
	case 'B':
		opt_background = true;
		break;
	case 'c': {
		json_error_t err;
		if (opt_config)
			json_decref(opt_config);
#if JANSSON_VERSION_HEX >= 0x020000
		opt_config = json_load_file(arg, 0, &err);
#else
		opt_config = json_load_file(arg, &err);
#endif
		if (!json_is_object(opt_config)) {
			applog(LOG_ERR, "JSON decode of %s failed", arg);
			exit(1);
		}
		break;
	}
	case 'q':
		opt_quiet = true;
		break;
	case 'D':
		opt_debug = true;
		break;
	case 'p':
		free(rpc_pass);
		rpc_pass = strdup(arg);
		break;
	case 'P':
		opt_protocol = true;
		break;
	case 'r':
		v = atoi(arg);
		if (v < -1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_retries = v;
		break;
	case 'R':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_fail_pause = v;
		break;
	case 's':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_scantime = v;
		break;
	case 'T':
		v = atoi(arg);
		if (v < 1 || v > 99999)	/* sanity check */
			show_usage_and_exit(1);
		opt_timeout = v;
		break;
	case 't':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_n_threads = v;
		break;
	case 'v':
		v = atoi(arg);
		if (v < 0 || v > 1024)	/* sanity check */
			show_usage_and_exit(1);
		opt_vote = (uint16_t)v;
		break;
	case 'm':
		opt_trust_pool = true;
		break;
	case 'u':
		free(rpc_user);
		rpc_user = strdup(arg);
		break;
	case 'o':			/* --url */
		p = strstr(arg, "://");
		if (p) {
			if (strncasecmp(arg, "http://", 7) && strncasecmp(arg, "https://", 8) &&
					strncasecmp(arg, "stratum+tcp://", 14))
				show_usage_and_exit(1);
			free(rpc_url);
			rpc_url = strdup(arg);
		} else {
			if (!strlen(arg) || *arg == '/')
				show_usage_and_exit(1);
			free(rpc_url);
			rpc_url = (char*)malloc(strlen(arg) + 8);
			sprintf(rpc_url, "http://%s", arg);
		}
		p = strrchr(rpc_url, '@');
		if (p) {
			char *sp, *ap;
			*p = '\0';
			ap = strstr(rpc_url, "://") + 3;
			sp = strchr(ap, ':');
			if (sp) {
				free(rpc_userpass);
				rpc_userpass = strdup(ap);
				free(rpc_user);
				rpc_user = (char*)calloc(sp - ap + 1, 1);
				strncpy(rpc_user, ap, sp - ap);
				free(rpc_pass);
				rpc_pass = strdup(sp + 1);
			} else {
				free(rpc_user);
				rpc_user = strdup(ap);
			}
			memmove(ap, p + 1, strlen(p + 1) + 1);
		}
		have_stratum = !opt_benchmark && !strncasecmp(rpc_url, "stratum", 7);
		break;
	case 'O':			/* --userpass */
		p = strchr(arg, ':');
		if (!p)
			show_usage_and_exit(1);
		free(rpc_userpass);
		rpc_userpass = strdup(arg);
		free(rpc_user);
		rpc_user = (char*)calloc(p - arg + 1, 1);
		strncpy(rpc_user, arg, p - arg);
		free(rpc_pass);
		rpc_pass = strdup(p + 1);
		break;
	case 'x':			/* --proxy */
		if (!strncasecmp(arg, "socks4://", 9))
			opt_proxy_type = CURLPROXY_SOCKS4;
		else if (!strncasecmp(arg, "socks5://", 9))
			opt_proxy_type = CURLPROXY_SOCKS5;
#if LIBCURL_VERSION_NUM >= 0x071200
		else if (!strncasecmp(arg, "socks4a://", 10))
			opt_proxy_type = CURLPROXY_SOCKS4A;
		else if (!strncasecmp(arg, "socks5h://", 10))
			opt_proxy_type = CURLPROXY_SOCKS5_HOSTNAME;
#endif
		else
			opt_proxy_type = CURLPROXY_HTTP;
		free(opt_proxy);
		opt_proxy = strdup(arg);
		break;
	case 1001:
		free(opt_cert);
		opt_cert = strdup(arg);
		break;
	case 1005:
		opt_benchmark = true;
		want_longpoll = false;
		want_stratum = false;
		have_stratum = false;
		break;
	case 1003:
		want_longpoll = false;
		break;
	case 1007:
		want_stratum = false;
		break;
	case 1010:
		allow_getwork = false;
		break;
	case 1011:
		have_gbt = false;
		break;
	case 1013:			/* --coinbase-addr */
		pk_script_size = address_to_script(pk_script, sizeof(pk_script), arg);
		if (!pk_script_size) {
/*
			fprintf(stderr, "%s: invalid address -- '%s'\n",
				pname, arg);
*/
			show_usage_and_exit(1);
		}
		break;
	case 1015:			/* --coinbase-sig */
		if (strlen(arg) + 1 > sizeof(coinbase_sig)) {
//			fprintf(stderr, "%s: coinbase signature too long\n", pname);
			show_usage_and_exit(1);
		}
		strcpy(coinbase_sig, arg);
		break;
	case 'S':
		use_syslog = true;
		break;
	case 'd': // CB
		{
			char * pch = strtok (arg,",");
			opt_n_threads = 0;
			while (pch != NULL) {
				if (pch[0] >= '0' && pch[0] <= '9' && pch[1] == '\0')
				{
					if (atoi(pch) < num_processors)
						device_map[opt_n_threads++] = atoi(pch);
					else {
						applog(LOG_ERR, "Non-existant CUDA device #%d specified in -d option", atoi(pch));
						exit(1);
					}
				} else {
					int device = cuda_finddevice(pch);
					if (device >= 0 && device < num_processors)
						device_map[opt_n_threads++] = device;
					else {
						applog(LOG_ERR, "Non-existant CUDA device '%s' specified in -d option", pch);
						exit(1);
					}
				}
				pch = strtok (NULL, ",");
			}
		} 
		break;

    case 'F': 
		{
			char * pch = strtok (arg,",");
			int tmp_n_threads = 0;
            float last = 0;
			while (pch != NULL) {
				tp_coef[tmp_n_threads++] = last = atof(pch);
				pch = strtok (NULL, ",");
			}
			while (tmp_n_threads < 8) tp_coef[tmp_n_threads++] = last;
		}
       break;
	case 'f': // CH - Divisor for Difficulty
		d = atof(arg);
		if (d == 0)	/* sanity check */
			show_usage_and_exit(1);
		opt_difficulty = d;
		break;
	case 'V':
		show_version_and_exit();
	case 'h':
		show_usage_and_exit(0);
	default:
		show_usage_and_exit(1);
	}
}

static void parse_config(void)
{
	int i;
	json_t *val;

	if (!json_is_object(opt_config))
		return;

	for (i = 0; i < ARRAY_SIZE(options); i++) {
		if (!options[i].name)
			break;
		if (!strcmp(options[i].name, "config"))
			continue;

		val = json_object_get(opt_config, options[i].name);
		if (!val)
			continue;

		if (options[i].has_arg && json_is_string(val)) {
			char *s = strdup(json_string_value(val));
			if (!s)
				break;
			parse_arg(options[i].val, s);
			free(s);
		} else if (!options[i].has_arg && json_is_true(val))
			parse_arg(options[i].val, "");
		else
			applog(LOG_ERR, "JSON option %s invalid",
				options[i].name);
	}

	if (opt_algo == ALGO_HEAVY && opt_vote == 9999) {
		fprintf(stderr, "Heavycoin hash requires block reward vote parameter (see --vote)\n");
		show_usage_and_exit(1);
	}
}

static void parse_cmdline(int argc, char *argv[])
{
	int key;

	while (1) {
#if HAVE_GETOPT_LONG
		key = getopt_long(argc, argv, short_options, options, NULL);
#else
		key = getopt(argc, argv, short_options);
#endif
		if (key < 0)
			break;

		parse_arg(key, optarg);
	}
	if (optind < argc) {
		fprintf(stderr, "%s: unsupported non-option argument '%s'\n",
			argv[0], argv[optind]);
		show_usage_and_exit(1);
	}

	//if (opt_algo == ALGO_HEAVY && opt_vote == 9999) {
	//	fprintf(stderr, "%s: Heavycoin hash requires block reward vote parameter (see --vote)\n",
	//		argv[0]);
	//	show_usage_and_exit(1);
	//}

	parse_config();
}

#ifndef WIN32
static void signal_handler(int sig)
{
	switch (sig) {
	case SIGHUP:
		applog(LOG_INFO, "SIGHUP received");
		break;
	case SIGINT:
		applog(LOG_INFO, "SIGINT received, exiting");
		exit(0);
		break;
	case SIGTERM:
		applog(LOG_INFO, "SIGTERM received, exiting");
		exit(0);
		break;
	}
}
#endif

#define PROGRAM_VERSION "djm34 pluck0.1"
int main(int argc, char *argv[])
{
	struct thr_info *thr;
	long flags;
	int i;

#ifdef WIN32
	SYSTEM_INFO sysinfo;
#endif

	 printf("        ***** ccMiner for nVidia GPUs by djm34  *****\n");
	 printf("\t             This is version "PROGRAM_VERSION" \n");
	 printf("	based on original ccMiner by Christian Buchner and Christian H. 2014 ***\n");	 
	 printf("\t  based on pooler-cpuminer 2.3.2 (c) 2010 Jeff Garzik, 2012 pooler\n");
	 printf("\t  based on pooler-cpuminer extension for HVC from\n\t       https://github.com/heavycoin/cpuminer-heavycoin\n");
	 printf("\t\t\tand\n\t       http://hvc.1gh.com/\n");
	 printf("\tCuda additions Copyright 2014 Christian Buchner, Christian H.\n");
	 printf("\tCuda additions Copyright 2014 DJM34\n");
	 printf("\t  FTC donation address: 6esbN82brbg3eai8fqzNGm5tmbpiYu3czM\n");
	 printf("\t  BTC donation address: 1NENYmxwZGHsKFmyjTc5WferTn5VTFb7Ze\n");
	 printf("\t  VTC donation address: VrLUQmH6Jk5gFii7fASc8vJ7eEgKJqhX11\n");
    
	 for (int i = 0; i<8; i++) {tp_coef[i]=-1;}
    opt_difficulty = 1. ;
	rpc_user = strdup("");
	rpc_pass = strdup("");

	pthread_mutex_init(&applog_lock, NULL);
	num_processors = cuda_num_devices();

	/* parse command line */
	parse_cmdline(argc, argv);
	
	cuda_devicenames();

	if (!opt_benchmark && !rpc_url) {
		fprintf(stderr, "%s: no URL supplied\n", argv[0]);
		show_usage_and_exit(1);
	}

	if (!rpc_userpass) {
		rpc_userpass = (char*)malloc(strlen(rpc_user) + strlen(rpc_pass) + 2);
		if (!rpc_userpass)
			return 1;
		sprintf(rpc_userpass, "%s:%s", rpc_user, rpc_pass);
	}

	pthread_mutex_init(&stats_lock, NULL);
	pthread_mutex_init(&g_work_lock, NULL);
	pthread_mutex_init(&stratum.sock_lock, NULL);
	pthread_mutex_init(&stratum.work_lock, NULL);

	flags = !opt_benchmark && strncmp(rpc_url, "https:", 6)
	      ? (CURL_GLOBAL_ALL & ~CURL_GLOBAL_SSL)
	      : CURL_GLOBAL_ALL;
	if (curl_global_init(flags)) {
		applog(LOG_ERR, "CURL initialization failed");
		return 1;
	}

#ifndef WIN32
	if (opt_background) {
		i = fork();
		if (i < 0) exit(1);
		if (i > 0) exit(0);
		i = setsid();
		if (i < 0)
			applog(LOG_ERR, "setsid() failed (errno = %d)", errno);
		i = chdir("/");
		if (i < 0)
			applog(LOG_ERR, "chdir() failed (errno = %d)", errno);
		signal(SIGHUP, signal_handler);
		signal(SIGINT, signal_handler);
		signal(SIGTERM, signal_handler);
	}
#endif

	if (num_processors == 0)
	{
		applog(LOG_ERR, "No CUDA devices found! terminating.");
		exit(1);
	}
	if (!opt_n_threads)
		opt_n_threads = num_processors;

#ifdef HAVE_SYSLOG_H
	if (use_syslog)
		openlog("cpuminer", LOG_PID, LOG_USER);
#endif

	work_restart = (struct work_restart *)calloc(opt_n_threads, sizeof(*work_restart));
	if (!work_restart)
		return 1;

	thr_info = (struct thr_info *)calloc(opt_n_threads + 3, sizeof(*thr));
	if (!thr_info)
		return 1;
	
	thr_hashrates = (double *) calloc(opt_n_threads, sizeof(double));
	if (!thr_hashrates)
		return 1;
	
	/* init workio thread info */
	work_thr_id = opt_n_threads;
	thr = &thr_info[work_thr_id];
	thr->id = work_thr_id;
	thr->q = tq_new();
	if (!thr->q)
		return 1;

	/* start work I/O thread */
	if (pthread_create(&thr->pth, NULL, workio_thread, thr)) {
		applog(LOG_ERR, "workio thread create failed");
		return 1;
	}

	if (want_longpoll && !have_stratum) {
		/* init longpoll thread info */
		longpoll_thr_id = opt_n_threads + 1;
		thr = &thr_info[longpoll_thr_id];
		thr->id = longpoll_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start longpoll thread */
		if (unlikely(pthread_create(&thr->pth, NULL, longpoll_thread, thr))) {
			applog(LOG_ERR, "longpoll thread create failed");
			return 1;
		}
	}
	if (want_stratum) {
		/* init stratum thread info */
		stratum_thr_id = opt_n_threads + 2;
		thr = &thr_info[stratum_thr_id];
		thr->id = stratum_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start stratum thread */
		if (unlikely(pthread_create(&thr->pth, NULL, stratum_thread, thr))) {
			applog(LOG_ERR, "stratum thread create failed");
			return 1;
		}

		if (have_stratum)
			tq_push(thr_info[stratum_thr_id].q, strdup(rpc_url));
	}

	/* start mining threads */
	for (i = 0; i < opt_n_threads; i++) {
		thr = &thr_info[i];

		thr->id = i;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		if (unlikely(pthread_create(&thr->pth, NULL, miner_thread, thr))) {
			applog(LOG_ERR, "thread %d create failed", i);
			return 1;
		}
	}

	applog(LOG_INFO, "%d miner threads started, "
		"using '%s' algorithm.",
		opt_n_threads,
		algo_names[opt_algo]);

#ifdef WIN32
	timeBeginPeriod(1); // enable high timer precision (similar to Google Chrome Trick)
#endif

	/* main loop - simply wait for workio thread to exit */
	pthread_join(thr_info[work_thr_id].pth, NULL);

#ifdef WIN32
	timeEndPeriod(1); // be nice and forego high timer precision
#endif

	applog(LOG_INFO, "workio thread dead, exiting.");

	return 0;
}
