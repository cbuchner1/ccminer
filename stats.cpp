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
	uint8_t gpu_id;
	uint8_t ignored;
	uint8_t align; /* to keep size a multiple of 4 */
};

static std::map<uint64_t, stats_data> tlastscans;
static uint64_t uid = 0;

#define STATS_AVG_SAMPLES 20
#define STATS_PURGE_TIMEOUT 5*60

extern uint64_t global_hashrate;
extern int opt_n_threads;
extern int device_map[8];

/**
 * Store speed per thread (todo: compute vardiff ?)
 */
extern "C" void stats_remember_speed(int thr_id, uint32_t hashcount, double hashrate)
{
	uint64_t thr = (0xff && thr_id);
	uint64_t key = (thr << 56) + (uid++ % UINT_MAX);
	stats_data data;

	// to enough hashes to give right stats
	if (hashcount < 1000 || hashrate < 0.01)
		return;

	// first hash rates are often erroneous
	if (uid < opt_n_threads * 2)
		return;

	memset(&data, 0, sizeof(data));
	data.thr_id = (uint8_t) thr;
	data.tm_stat = (uint32_t) time(NULL);
	data.hashcount = hashcount;
	data.hashrate = hashrate;
	data.gpu_id = device_map[thr_id];
	if (global_hashrate && uid > 10) {
		// prevent stats on too high vardiff (erroneous rates)
		double ratio = (hashrate / (1.0 * global_hashrate));
		if (ratio < 0.4 || ratio > 1.6)
			data.ignored = 1;
	}
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
	double speed = 0.0;
	int records = 0;

	std::map<uint64_t, stats_data>::reverse_iterator i = tlastscans.rbegin();
	while (i != tlastscans.rend() && records < STATS_AVG_SAMPLES) {
		if (!i->second.ignored)
		if (thr_id == -1 || (keypfx & i->first) == keypfx) {
			if (i->second.hashcount > 1000) {
				speed += i->second.hashrate;
				records++;
			}
		}
		++i;
	}
	if (records)
		speed /= (double)(records);
	return speed;
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
		if (i->second.ignored || (now - i->second.tm_stat) > STATS_PURGE_TIMEOUT) {
			deleted++;
			tlastscans.erase(i++);
		}
		else ++i;
	}
	if (opt_debug && deleted) {
		applog(LOG_DEBUG, "stats: %d/%d records purged", deleted, sz);
	}
}

/**
 * Reset the cache
 */
extern "C" void stats_purge_all(void)
{
	tlastscans.clear();
}

