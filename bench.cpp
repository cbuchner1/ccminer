/**
 * Made to benchmark and test algo switch
 *
 * 2015 - tpruvot@github
 */

#include "miner.h"
#include "algos.h"

#include <unistd.h>

int bench_algo = -1;

static double algo_hashrates[MAX_GPUS][ALGO_COUNT] = { 0 };
static uint32_t algo_throughput[MAX_GPUS][ALGO_COUNT] = { 0 };
static int algo_mem_used[MAX_GPUS][ALGO_COUNT] = { 0 };
static int device_mem_free[MAX_GPUS] = { 0 };

static pthread_barrier_t miner_barr;
static pthread_barrier_t algo_barr;
static pthread_mutex_t bench_lock = PTHREAD_MUTEX_INITIALIZER;

extern double thr_hashrates[MAX_GPUS];
extern enum sha_algos opt_algo;

void bench_init(int threads)
{
	bench_algo = opt_algo = (enum sha_algos) 0; /* first */
	applog(LOG_BLUE, "Starting benchmark mode with %s", algo_names[opt_algo]);
	pthread_barrier_init(&miner_barr, NULL, threads);
	pthread_barrier_init(&algo_barr, NULL, threads);
}

void bench_free()
{
	pthread_barrier_destroy(&miner_barr);
	pthread_barrier_destroy(&algo_barr);
}

// benchmark all algos (called once per mining thread)
bool bench_algo_switch_next(int thr_id)
{
	int algo = (int) opt_algo;
	int prev_algo = algo;
	int dev_id = device_map[thr_id % MAX_GPUS];
	int mfree, mused;

	algo++;

	// skip some duplicated algos
	if (algo == ALGO_C11) algo++; // same as x11
	if (algo == ALGO_DMD_GR) algo++; // same as groestl
	if (algo == ALGO_WHIRLCOIN) algo++; // same as whirlpool
	// and unwanted ones...
	if (algo == ALGO_LYRA2) algo++; // weird memory leak to fix (uint2 Matrix[96][8] too big)
	if (algo == ALGO_SCRYPT) algo++;
	if (algo == ALGO_SCRYPT_JANE) algo++;

	// we need to wait completion on all cards before the switch
	if (opt_n_threads > 1) {
		pthread_barrier_wait(&miner_barr);
	}

	char rate[32] = { 0 };
	double hashrate = stats_get_speed(thr_id, thr_hashrates[thr_id]);
	format_hashrate(hashrate, rate);
	applog(LOG_NOTICE, "GPU #%d: %s hashrate = %s", dev_id, algo_names[prev_algo], rate);

	// free current algo memory and track mem usage
	mused = cuda_available_memory(thr_id);
	miner_free_device(thr_id);
	mfree = cuda_available_memory(thr_id);

	// check if there is memory leak
	if (device_mem_free[thr_id] > mfree) {
		applog(LOG_WARNING, "GPU #%d, memory leak detected in %s ! %d MB free",
			dev_id, algo_names[prev_algo], mfree);
	}
	// store used memory per algo
	algo_mem_used[thr_id][opt_algo] = device_mem_free[thr_id] - mused;
	device_mem_free[thr_id] = mfree;

	// store to dump a table per gpu later
	algo_hashrates[thr_id][prev_algo] = hashrate;


	// wait the other threads to display logs correctly
	if (opt_n_threads > 1) {
		pthread_barrier_wait(&algo_barr);
	}

	if (algo == ALGO_AUTO)
		return false;

	// mutex primary used for the stats purge
	pthread_mutex_lock(&bench_lock);
	stats_purge_all();

	opt_algo = (enum sha_algos) algo;
	global_hashrate = 0;
	thr_hashrates[thr_id] = 0; // reset for minmax64
	pthread_mutex_unlock(&bench_lock);

	if (thr_id == 0)
		applog(LOG_BLUE, "Benchmark algo %s...", algo_names[algo]);

	return true;
}

void bench_set_throughput(int thr_id, uint32_t throughput)
{
	algo_throughput[thr_id][opt_algo] = throughput;
}

void bench_display_results()
{
	for (int n=0; n < opt_n_threads; n++)
	{
		int dev_id = device_map[n];
		applog(LOG_BLUE, "Benchmark results for GPU #%d - %s:", dev_id, device_name[dev_id]);
		for (int i=0; i < ALGO_COUNT-1; i++) {
			double rate = algo_hashrates[n][i];
			if (rate == 0.0) continue;
			applog(LOG_INFO, "%12s : %12.1f kH/s, %5d MB, %8u thr.", algo_names[i],
				rate / 1024., algo_mem_used[n][i], algo_throughput[n][i]);
		}
	}
}

