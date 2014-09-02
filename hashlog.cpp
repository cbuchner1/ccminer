#include <inttypes.h>
#include <stdlib.h>
#include <map>

#include "miner.h"

static std::map<uint64_t, uint32_t> tlastshares;

/**
 * Purge entries after 15 minutes
 */
#define LOG_PURGE_TIMEOUT 15*60

/**
 * Store submitted nounces of a job
 */
extern "C" void hashlog_remember_submit(char* jobid, uint32_t nounce)
{
	char *ptr;
	uint64_t njobid = (uint64_t) strtoul(jobid, &ptr, 16);
	uint64_t key = (njobid << 32) + nounce;
	tlastshares[key] = (uint32_t) time(NULL);
}

/**
 * @return time of submission
 */
extern "C" uint32_t hashlog_already_submittted(char* jobid, uint32_t nounce)
{
	char *ptr;
	uint32_t ret = 0;
	uint64_t njobid = (uint64_t) strtoul(jobid, &ptr, 16);
	uint64_t key = (njobid << 32) + nounce;
	std::map<uint64_t, uint32_t>::iterator i = tlastshares.find(key);
	if (i != tlastshares.end())
		ret = (uint32_t) tlastshares[key];
	return ret;
}

/**
 * Remove entries of a job... not used yet
 */
extern "C" void hashlog_purge_job(char* jobid)
{
	char *ptr;
	uint64_t njobid = strtoul(jobid, &ptr, 16);
	uint64_t keypfx = (njobid << 32);
	std::map<uint64_t, uint32_t>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) != 0)
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
	std::map<uint64_t, uint32_t>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((now - i->second) > LOG_PURGE_TIMEOUT) {
			deleted++;
			tlastshares.erase(i);
		}
		i++;
	}
	if (opt_debug && deleted) {
		applog(LOG_DEBUG, "hashlog: %d/%d purged",
			deleted, tlastshares.size());
	}
}

/**
 * Reset the submitted nounce cache
 */
extern "C" void hashlog_purge_all(void)
{
	tlastshares.clear();
}

