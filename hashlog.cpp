//#include <inttypes.h>
#include <stdlib.h>
#include <map>

#include "miner.h"

#define HI_DWORD(u64) ((uint32_t) (u64 >> 32))
#define LO_DWORD(u64) ((uint32_t) u64)

struct hashlog_data {
	uint32_t ntime;
	uint32_t scanned_from;
	uint32_t scanned_to;
};

static std::map<uint64_t, hashlog_data> tlastshares;

#define LOG_PURGE_TIMEOUT 15*60

/**
 * str hex to uint32
 */
static uint64_t hextouint(char* jobid)
{
	char *ptr;
	return strtoull(jobid, &ptr, 16);
}

/**
 * Store submitted nonces of a job
 */
extern "C" void hashlog_remember_submit(char* jobid, uint32_t nonce, uint64_t range)
{
	uint64_t njobid = hextouint(jobid);
	uint64_t key = (njobid << 32) + nonce;
	struct hashlog_data data;
	data.ntime = (uint32_t) time(NULL);
	data.scanned_from = LO_DWORD(range);
	data.scanned_to   = HI_DWORD(range);
	tlastshares[key] = data;
}

/**
 * Search last submitted nonce for a job
 * @return max nonce
 */
extern "C" uint32_t hashlog_get_last_sent(char* jobid)
{
	uint32_t ret = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) == keypfx && LO_DWORD(i->first) > ret) {
			ret = LO_DWORD(i->first);
		}
		i++;
	}
	return ret;
}

/**
 * @return time of a job/nonce submission (or last nonce if nonce is 0)
 */
extern "C" uint32_t hashlog_already_submittted(char* jobid, uint32_t nonce)
{
	uint32_t ret = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t key = (njobid << 32) + nonce;
	if (nonce == 0) {
		// search last submitted nonce for job
		ret = hashlog_get_last_sent(jobid);
	} else if (tlastshares.find(key) != tlastshares.end()) {
		hashlog_data data = tlastshares[key];
		ret = data.ntime;
	}
	return ret;
}

/**
 * Remove entries of a job... not used yet
 */
extern "C" void hashlog_purge_job(char* jobid)
{
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) == keypfx)
			tlastshares.erase(i);
		i++;
	}
}

/**
 * Remove old entries to reduce memory usage
 */
extern "C" void hashlog_purge_old(void)
{
	int deleted = 0;
	uint32_t now = (uint32_t) time(NULL);
	uint32_t sz = tlastshares.size();
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((now - i->second.ntime) > LOG_PURGE_TIMEOUT) {
			deleted++;
			tlastshares.erase(i);
		}
		i++;
	}
	if (opt_debug && deleted) {
		applog(LOG_DEBUG, "hashlog: %d/%d purged", deleted, sz);
	}
}

/**
 * Reset the submitted nonces cache
 */
extern "C" void hashlog_purge_all(void)
{
	tlastshares.clear();
}
