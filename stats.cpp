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

static std::map<uint64_t, stats_data> tlastscans;
static uint64_t uid = 0;

#define STATS_AVG_SAMPLES 30
#define STATS_PURGE_TIMEOUT 120*60 /* 120 mn */

extern uint64_t global_hashrate;
extern int opt_statsavg;

/**
 * Store speed per thread
 */
void stats_remember_speed(int thr_id, uint32_t hashcount, double hashrate, uint8_t found, uint32_t height)
{
	const uint64_t key = uid++;
	stats_data data;
	// to enough hashes to give right stats
	if (hashcount < 1000 || hashrate < 0.01)
		return;

	// first hash rates are often erroneous
	//if (uid < opt_n_threads * 2)
	//	return;

	memset(&data, 0, sizeof(data));
	data.uid = (uint32_t) uid;
	data.gpu_id = (uint8_t) device_map[thr_id];
	data.thr_id = (uint8_t) thr_id;
	data.tm_stat = (uint32_t) time(NULL);
	data.height = height;
	data.npool = (uint8_t) cur_pooln;
	data.pool_type = pools[cur_pooln].type;
	data.hashcount = hashcount;
	data.hashfound = found;
	data.hashrate = hashrate;
	data.difficulty = net_diff ? net_diff : stratum_diff;
	if (opt_n_threads == 1 && global_hashrate && uid > 10) {
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
double stats_get_speed(int thr_id, double def_speed)
{
	double speed = 0.0;
	int records = 0;

	std::map<uint64_t, stats_data>::reverse_iterator i = tlastscans.rbegin();
	while (i != tlastscans.rend() && records < opt_statsavg) {
		if (!i->second.ignored)
		if (thr_id == -1 || i->second.thr_id == thr_id) {
			if (i->second.hashcount > 1000) {
				speed += i->second.hashrate;
				records++;
				// applog(LOG_BLUE, "%d %x %.1f", thr_id, i->second.thr_id, i->second.hashrate);
			}
		}
		++i;
	}

	if (records)
		speed /= (double)(records);
	else
		speed = def_speed;

	if (thr_id == -1)
		speed *= (double)(opt_n_threads);

	return speed;
}

/**
 * Get the gpu average speed
 * @param gpu_id int (-1 for all threads)
 */
double stats_get_gpu_speed(int gpu_id)
{
	double speed = 0.0;

	for (int thr_id=0; thr_id<opt_n_threads; thr_id++) {
		int dev_id = device_map[thr_id];
		if (gpu_id == -1 || dev_id == gpu_id)
			speed += stats_get_speed(thr_id, 0.0);
	}

	return speed;
}

/**
 * Export data for api calls
 */
int stats_get_history(int thr_id, struct stats_data *data, int max_records)
{
	int records = 0;

	std::map<uint64_t, stats_data>::reverse_iterator i = tlastscans.rbegin();
	while (i != tlastscans.rend() && records < max_records) {
		if (!i->second.ignored)
			if (thr_id == -1 || i->second.thr_id == thr_id) {
				memcpy(&data[records], &(i->second), sizeof(struct stats_data));
				records++;
			}
		++i;
	}
	return records;
}

/**
 * Remove old entries to reduce memory usage
 */
void stats_purge_old(void)
{
	int deleted = 0;
	uint32_t now = (uint32_t) time(NULL);
	uint32_t sz = (uint32_t) tlastscans.size();
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
void stats_purge_all(void)
{
	tlastscans.clear();
}

/**
 * API meminfo
 */
void stats_getmeminfo(uint64_t *mem, uint32_t *records)
{
	(*records) = (uint32_t) tlastscans.size();
	(*mem) = (*records) * sizeof(stats_data);
}
