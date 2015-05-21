/**
 * Hash log of submitted job nonces
 * Prevent duplicate shares
 *
 * (to be merged later with stats)
 *
 * tpruvot@github 2014
 */
#include <stdlib.h>
#include <memory.h>
#include <map>

#include "miner.h"

#define HI_DWORD(u64) ((uint32_t) (u64 >> 32))
#define LO_DWORD(u64) ((uint32_t) u64)
#define MK_HI64(u32) (0x100000000ULL * u32)

/* from miner.h
struct hashlog_data {
	uint8_t npool;
	uint8_t pool_type;
	uint32_t height;
	uint32_t njobid;
	uint32_t nonce;
	uint32_t scanned_from;
	uint32_t scanned_to;
	uint32_t last_from;
	uint32_t tm_add;
	uint32_t tm_upd;
	uint32_t tm_sent;
};
*/

static std::map<uint64_t, hashlog_data> tlastshares;

#define LOG_PURGE_TIMEOUT 5*60

/**
 * str hex to uint32
 */
static uint64_t hextouint(char* jobid)
{
	char *ptr;
	/* dont use strtoull(), only since VS2013 */
	return (uint64_t) strtoul(jobid, &ptr, 16);
}

/**
 * @return time of a job/nonce submission (or last nonce if nonce is 0)
 */
uint32_t hashlog_already_submittted(char* jobid, uint32_t nonce)
{
	uint32_t ret = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t key = (njobid << 32) + nonce;

	if (nonce == 0) {
		// search last submitted nonce for job
		ret = hashlog_get_last_sent(jobid);
	} else if (tlastshares.find(key) != tlastshares.end()) {
		hashlog_data data = tlastshares[key];
		ret = data.tm_sent;
	}
	return ret;
}
/**
 * Store submitted nonces of a job
 */
void hashlog_remember_submit(struct work* work, uint32_t nonce)
{
	uint64_t njobid = hextouint(work->job_id);
	uint64_t key = (njobid << 32) + nonce;
	hashlog_data data;

	memset(&data, 0, sizeof(data));
	data.scanned_from = work->scanned_from;
	data.scanned_to = nonce;
	data.height = work->height;
	data.njobid = (uint32_t) njobid;
	data.tm_add = data.tm_upd = data.tm_sent = (uint32_t) time(NULL);
	data.npool = (uint8_t) cur_pooln;
	data.pool_type = pools[cur_pooln].type;
	tlastshares[key] = data;
}

/**
 * Update job scanned range
 */
void hashlog_remember_scan_range(struct work* work)
{
	uint64_t njobid = hextouint(work->job_id);
	uint64_t key = (njobid << 32);
	uint64_t range = hashlog_get_scan_range(work->job_id);
	hashlog_data data;

	// global scan range of a job
	data = tlastshares[key];
	if (range == 0) {
		memset(&data, 0, sizeof(data));
		data.njobid = (uint32_t) njobid;
	} else {
		// get min and max from all sent records
		data.scanned_from = LO_DWORD(range);
		data.scanned_to   = HI_DWORD(range);
	}

	if (data.tm_add == 0)
		data.tm_add = (uint32_t) time(NULL);

	data.last_from = work->scanned_from;

	if (work->scanned_from < work->scanned_to) {
		if (data.scanned_to == 0 || work->scanned_from == data.scanned_to + 1)
			data.scanned_to = work->scanned_to;
		if (data.scanned_from == 0)
			data.scanned_from = work->scanned_from ? work->scanned_from : 1; // min 1
		else if (work->scanned_from < data.scanned_from || work->scanned_to == (data.scanned_from - 1))
			data.scanned_from = work->scanned_from;
	}

	data.tm_upd = (uint32_t) time(NULL);

	tlastshares[key] = data;
/* 	applog(LOG_BLUE, "job %s range : %x %x -> %x %x", jobid,
		scanned_from, scanned_to, data.scanned_from, data.scanned_to); */
}

/**
 * Returns the range of a job
 * @return uint64_t to|from
 */
uint64_t hashlog_get_scan_range(char* jobid)
{
	uint64_t ret = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	uint64_t keymsk = (0xffffffffULL << 32);
	hashlog_data data;

	data.scanned_from = 0;
	data.scanned_to = 0;
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keymsk & i->first) == keypfx && i->second.scanned_to > ret) {
			if (i->second.scanned_to > data.scanned_to)
				data.scanned_to = i->second.scanned_to;
			if (i->second.scanned_from < data.scanned_from || data.scanned_from == 0)
				data.scanned_from = i->second.scanned_from;
		}
		i++;
	}
	ret = data.scanned_from;
	ret += MK_HI64(data.scanned_to);
	return ret;
}

/**
 * Search last submitted nonce for a job
 * @return max nonce
 */
uint32_t hashlog_get_last_sent(char* jobid)
{
	uint32_t nonce = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) == keypfx && i->second.tm_sent > 0) {
			nonce = LO_DWORD(i->first);
		}
		i++;
	}
	return nonce;
}

/**
 * Export data for api calls
 */
int hashlog_get_history(struct hashlog_data *data, int max_records)
{
	int records = 0;

	std::map<uint64_t, hashlog_data>::reverse_iterator it = tlastshares.rbegin();
	while (it != tlastshares.rend() && records < max_records) {
		memcpy(&data[records], &(it->second), sizeof(struct hashlog_data));
		data[records].nonce = LO_DWORD(it->first);
		data[records].njobid = (uint32_t) HI_DWORD(it->first);
		records++;
		++it;
	}
	return records;
}

/**
 * Remove entries of a job...
 */
void hashlog_purge_job(char* jobid)
{
	int deleted = 0;
	uint64_t njobid = hextouint(jobid);
	uint64_t keypfx = (njobid << 32);
	uint32_t sz = (uint32_t) tlastshares.size();
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((keypfx & i->first) == keypfx) {
			deleted++;
			tlastshares.erase(i++);
		}
		else ++i;
	}
	if (opt_debug && deleted) {
		applog(LOG_DEBUG, "hashlog: purge job %s, del %d/%d", jobid, deleted, sz);
	}
}

/**
 * Remove old entries to reduce memory usage
 */
void hashlog_purge_old(void)
{
	int deleted = 0;
	uint32_t now = (uint32_t) time(NULL);
	uint32_t sz = (uint32_t) tlastshares.size();
	std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
	while (i != tlastshares.end()) {
		if ((now - i->second.tm_sent) > LOG_PURGE_TIMEOUT) {
			deleted++;
			tlastshares.erase(i++);
		}
		else ++i;
	}
	if (opt_debug && deleted) {
		applog(LOG_DEBUG, "hashlog: %d/%d purged", deleted, sz);
	}
}

/**
 * Reset the submitted nonces cache
 */
void hashlog_purge_all(void)
{
	tlastshares.clear();
}

/**
 * API meminfo
 */
void hashlog_getmeminfo(uint64_t *mem, uint32_t *records)
{
	(*records) = (uint32_t) tlastshares.size();
	(*mem) = (*records) * sizeof(hashlog_data);
}

/**
 * Used to debug ranges...
 */
void hashlog_dump_job(char* jobid)
{
	if (opt_debug) {
		uint64_t njobid = hextouint(jobid);
		uint64_t keypfx = (njobid << 32);
		// uint32_t sz = tlastshares.size();
		std::map<uint64_t, hashlog_data>::iterator i = tlastshares.begin();
		while (i != tlastshares.end()) {
			if ((keypfx & i->first) == keypfx) {
				if (i->first != keypfx)
					applog(LOG_DEBUG, CL_YLW "job %s, found %08x ", jobid, LO_DWORD(i->first));
				else
					applog(LOG_DEBUG, CL_YLW "job %s(%u) range done: %08x-%08x", jobid,
						i->second.height, i->second.scanned_from, i->second.scanned_to);
			}
			i++;
		}
	}
}
