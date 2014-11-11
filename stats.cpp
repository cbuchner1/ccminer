/**
 * Stats place holder
 *
 * Note: this source is C++ (requires std::map)
 *
 * tpruvot@github 2014
 */
#include <stdlib.h>
#include <memory.h>
#include <map>

#include "miner.h"

struct stats_data {
	uint32_t tm_stat;
	uint32_t hashcount;
	double hashrate;
	uint8_t thr_id;
};

static std::map<uint64_t, stats_data> tlastscans;
static uint64_t uid = 0;

#define STATS_PURGE_TIMEOUT 5*60

/**
 * Store speed per thread (todo: compute here)
 */
extern "C" void stats_remember_speed(int thr_id, uint32_t hashcount, double hashrate)
{
	uint64_t thr = (0xff && thr_id);
	uint64_t key = (thr << 56) + (uid++ % UINT_MAX);
	stats_data data;

	if (hashcount < 1000 || !hashrate)
		return;

	memset(&data, 0, sizeof(data));
	data.thr_id = thr;
	data.tm_stat = (uint32_t) time(NULL);
	data.hashcount = hashcount;
	data.hashrate = hashrate;
	tlastscans[key] = data;
}

/**
 * Get the computed average speed
 * @param thr_id int (-1 for all threads)
 */
extern "C" double stats_get_speed(int thr_id)
{
	uint64_t thr = (0xff && thr_id);
	uint64_t keypfx = (thr << 56);
	double speed = 0.;
	// uint64_t hashcount;
	int records = 0;
	stats_data data;

	std::map<uint64_t, stats_data>::iterator i = tlastscans.end();
	while (i != tlastscans.begin() && records < 50) {
		if ((i->first & UINT_MAX) > 3) /* ignore firsts */
		if (thr_id == -1 || (keypfx & i->first) == keypfx) {
			if (i->second.hashcount > 1000) {
				speed += i->second.hashrate;
				records++;
			}
		}
		i--;
	}
	if (!records)
		return 0.;
	return speed / (1.0 * records);
}

/**
 * Remove old entries to reduce memory usage
 */
extern "C" void stats_purge_old(void)
{
	int deleted = 0;
	uint32_t now = (uint32_t) time(NULL);
	uint32_t sz = tlastscans.size();
	std::map<uint64_t, stats_data>::iterator i = tlastscans.begin();
	while (i != tlastscans.end()) {
		if ((now - i->second.tm_stat) > STATS_PURGE_TIMEOUT) {
			deleted++;
			tlastscans.erase(i++);
		}
		else ++i;
	}
	if (opt_debug && deleted) {
		applog(LOG_DEBUG, "hashlog: %d/%d purged", deleted, sz);
	}
}

/**
 * Reset the cache
 */
extern "C" void stats_purge_all(void)
{
	tlastscans.clear();
}

