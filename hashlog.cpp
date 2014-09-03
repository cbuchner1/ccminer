#include <stdlib.h>
#include <memory.h>
#include <map>

#include "miner.h"

#define HI_DWORD(u64) ((uint32_t) (u64 >> 32))
#define LO_DWORD(u64) ((uint32_t) u64)
#define MK_HI64(u32) (0x100000000ULL * u32)

struct hashlog_data {
	uint32_t ntime;
	uint32_t scanned_from;
	uint32_t scanned_to;
	uint32_t last_from;
};

static std::map<uint64_t, hashlog_data> tlastshares;

#define LOG_PURGE_TIMEOUT 5*60

/**
 * str hex to uint32
 */
static uint64_t hextouint(char* jobid)
{
	char *ptr;
	return strtoull(jobid, &ptr, 16);
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
 * Store submitted nonces of a job
 */
extern "C" void hashlog_remember_submit(char* jobid, uint32_t nonce)
{
	uint64_t njobid = hextouint(jobid);
	uint64_t keyall = (njobid << 32);
	uint64_t key = keyall + nonce;
	struct hashlog_data data;

	data = tlastshares[keyall];
	data.ntime = (uint32_t) time(NULL);
	tlastshares[key] = data;
}

/**
 * Update job scanned range
 */
extern "C" void hashlog_remember_scan_range(char* jobid, uint32_t scanned_from, uint32_t scanned_to)
{
	uint64_t njobid = hextouint(jobid);
	uint64_t keyall = (njobid << 32);
	struct hashlog_data data;

	// global scan range of a job
	data = tlastshares[keyall];
	if (hashlog_get_scan_range(jobid) == 0) {
		memset(&data, 0, sizeof(data));
	}

	if (data.scanned_from == 0 || scanned_to == (data.scanned_from - 1))
		data.scanned_from = scanned_from ? scanned_from : 1; // min 1
	if (data.scanned_to == 0 || scanned_from == data.scanned_to + 1)
		data.scanned_to = scanned_to;

	data.last_from = scanned_from;

	tlastshares[keyall] = data;
	applog(LOG_BLUE, "job %s range : %x %x -> %x %x (%x)", jobid,
		scanned_from, scanned_to, data.scanned_from, data.scanned_to, data.ntime);/* */
}

/**
 * Returns the range of a job
 * @return uint64_t to|from
 */
extern "C" uint64_t hashlog_get_scan_range(char* jobid)
{
	uint64_t ret = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) == keypfx) {
			hashlog_data data = i->second;
			ret = data.scanned_from;
			ret += MK_HI64(data.scanned_to);
		}
		i++;
	}
	return ret;
}

/**
 * Search last submitted nonce for a job
 * @return max nonce
 */
extern "C" uint32_t hashlog_get_last_sent(char* jobid)
{
	uint32_t nonce = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) == keypfx && i->second.ntime > 0) {
			nonce = LO_DWORD(i->first);
		}
		i++;
	}
	return nonce;
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
