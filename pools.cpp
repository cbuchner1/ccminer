/**
 * Functions which handle multiple pools data
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "miner.h"
#include "compat.h"
#include "algos.h"

// to move in miner.h
extern bool allow_gbt;
extern bool allow_mininginfo;
extern bool check_dups;

extern double opt_max_diff;
extern double opt_max_rate;
extern int opt_scantime;
extern int opt_shares_limit;
extern int opt_time_limit;

extern char* rpc_url;
extern char* rpc_user;
extern char* rpc_pass;
extern char* short_url;

extern struct work _ALIGN(64) g_work;
extern struct stratum_ctx stratum;
extern pthread_mutex_t stratum_work_lock;
extern pthread_mutex_t stats_lock;
extern bool get_work(struct thr_info *thr, struct work *work);
extern bool stratum_need_reset;
extern time_t firstwork_time;

extern volatile time_t g_work_time;
extern volatile int pool_switch_count;
extern volatile bool pool_is_switching;
extern uint8_t conditional_state[MAX_GPUS];

extern double thr_hashrates[MAX_GPUS];

extern struct option options[];

#define CFG_NULL 0
#define CFG_POOL 1
struct opt_config_array {
	int cat;
	const char *name;     // json key
	const char *longname; // global opt name if different
} cfg_array_keys[] = {
	{ CFG_POOL, "url", NULL }, /* let this key first, increment pools */
	{ CFG_POOL, "user", NULL },
	{ CFG_POOL, "pass", NULL },
	{ CFG_POOL, "userpass", NULL },
	{ CFG_POOL, "name", "pool-name" },
	{ CFG_POOL, "algo", "pool-algo" },
	{ CFG_POOL, "scantime", "pool-scantime" },
	{ CFG_POOL, "max-diff", "pool-max-diff" },
	{ CFG_POOL, "max-rate", "pool-max-rate" },
	{ CFG_POOL, "disabled", "pool-disabled" },
	{ CFG_POOL, "time-limit", "pool-time-limit" },
	{ CFG_NULL, NULL, NULL }
};

// store current credentials in pools container
void pool_set_creds(int pooln)
{
	struct pool_infos *p = &pools[pooln];

	snprintf(p->url, sizeof(p->url), "%s", rpc_url);
	snprintf(p->short_url, sizeof(p->short_url), "%s", short_url);
	snprintf(p->user, sizeof(p->user), "%s", rpc_user);
	snprintf(p->pass, sizeof(p->pass), "%s", rpc_pass);

	if (!(p->status & POOL_ST_DEFINED)) {
		p->id = pooln;
		p->status |= POOL_ST_DEFINED;
		// init pool options as "unset"
		// until cmdline is fully parsed...
		p->algo = -1;
		p->max_diff = -1.;
		p->max_rate = -1.;
		p->scantime = -1;
		p->shares_limit = -1;
		p->time_limit = -1;

		p->allow_mininginfo = allow_mininginfo;
		p->allow_gbt = allow_gbt;
		p->check_dups = check_dups;

		p->status |= POOL_ST_DEFINED;
	}

	if (strlen(rpc_url)) {
		if (!strncasecmp(rpc_url, "stratum", 7))
			p->type = POOL_STRATUM;
		else /* if (!strncasecmp(rpc_url, "http", 4)) */
			p->type = POOL_GETWORK; // todo: or longpoll
		p->status |= POOL_ST_VALID;
	}
}

// fill the unset pools options with cmdline ones
void pool_init_defaults()
{
	struct pool_infos *p;
	for (int i=0; i<num_pools; i++) {
		p = &pools[i];
		if (p->algo == -1) p->algo = (int) opt_algo;
		if (p->max_diff == -1.) p->max_diff = opt_max_diff;
		if (p->max_rate == -1.) p->max_rate = opt_max_rate;
		if (p->scantime == -1) p->scantime = opt_scantime;
		if (p->shares_limit == -1) p->shares_limit = opt_shares_limit;
		if (p->time_limit == -1) p->time_limit = opt_time_limit;
	}
}

// attributes only set by a json pools config
void pool_set_attr(int pooln, const char* key, char* arg)
{
	struct pool_infos *p = &pools[pooln];
	if (!strcasecmp(key, "name")) {
		snprintf(p->name, sizeof(p->name), "%s", arg);
		return;
	}
	if (!strcasecmp(key, "algo")) {
		p->algo = algo_to_int(arg);
		return;
	}
	if (!strcasecmp(key, "scantime")) {
		p->scantime = atoi(arg);
		return;
	}
	if (!strcasecmp(key, "max-diff")) {
		p->max_diff = atof(arg);
		return;
	}
	if (!strcasecmp(key, "max-rate")) {
		p->max_rate = atof(arg);
		return;
	}
	if (!strcasecmp(key, "shares-limit")) {
		p->shares_limit = atoi(arg);
		return;
	}
	if (!strcasecmp(key, "time-limit")) {
		p->time_limit = atoi(arg);
		return;
	}
	if (!strcasecmp(key, "disabled")) {
		int removed = atoi(arg);
		if (removed) {
			p->status |= POOL_ST_REMOVED;
		}
		return;
	}
}

// pool switching code
bool pool_switch(int thr_id, int pooln)
{
	int prevn = cur_pooln;
	bool algo_switch = false;
	struct pool_infos *prev = &pools[cur_pooln];
	struct pool_infos* p = NULL;

	// save prev stratum connection infos (struct)
	if (prev->type & POOL_STRATUM) {
		// may not be the right moment to free,
		// to check if required on submit...
		stratum_free_job(&stratum);
		prev->stratum = stratum;
	}

	if (pooln < num_pools) {
		cur_pooln = pooln;
		p = &pools[cur_pooln];
	} else {
		applog(LOG_ERR, "Switch to inexistant pool %d!", pooln);
		return false;
	}

	// save global attributes
	prev->allow_mininginfo = allow_mininginfo;
	prev->allow_gbt = allow_gbt;
	prev->check_dups = check_dups;

	pthread_mutex_lock(&stratum_work_lock);

	free(rpc_user); rpc_user = strdup(p->user);
	free(rpc_pass); rpc_pass = strdup(p->pass);
	free(rpc_url);  rpc_url = strdup(p->url);

	short_url = p->short_url; // just a pointer, no alloc

	opt_scantime = p->scantime;
	opt_max_diff = p->max_diff;
	opt_max_rate = p->max_rate;
	opt_shares_limit = p->shares_limit;
	opt_time_limit = p->time_limit;

	want_stratum = have_stratum = (p->type & POOL_STRATUM) != 0;

	// yiimp stats reporting
	opt_stratum_stats = (strstr(p->pass, "stats") != NULL) || (strcmp(p->user, "benchmark") == 0);

	pthread_mutex_unlock(&stratum_work_lock);

	// algo "blind" switch without free, not proper
	// todo: barrier required to free algo resources
	if (p->algo != (int) opt_algo) {

		if (opt_algo != ALGO_AUTO) {

			algo_switch = true;

			pthread_mutex_lock(&stats_lock);
			for (int n=0; n<opt_n_threads; n++)
				thr_hashrates[n] = 0.;
			stats_purge_all();
			if (check_dups)
				hashlog_purge_all();
			pthread_mutex_unlock(&stats_lock);
		}

		opt_algo = (enum sha_algos) p->algo;
	}

	if (prevn != cur_pooln) {

		pool_switch_count++;
		net_diff = 0;
		g_work_time = 0;
		g_work.data[0] = 0;
		pool_is_switching = true;
		stratum_need_reset = true;
		// used to get the pool uptime
		firstwork_time = time(NULL);
		restart_threads();
		// reset wait states
		for (int n=0; n<opt_n_threads; n++)
			conditional_state[n] = false;

		// restore flags
		allow_gbt = p->allow_gbt;
		allow_mininginfo = p->allow_mininginfo;
		check_dups = p->check_dups;

		if (want_stratum) {

			// temporary... until stratum code cleanup
			stratum = p->stratum;
			stratum.pooln = cur_pooln;
			stratum.rpc2 = (p->algo == ALGO_WILDKECCAK || p->algo == ALGO_CRYPTONIGHT);
			stratum.rpc2 |= p->algo == ALGO_CRYPTOLIGHT;

			// unlock the stratum thread
			tq_push(thr_info[stratum_thr_id].q, strdup(rpc_url));
			applog(LOG_BLUE, "Switch to stratum pool %d: %s", cur_pooln,
				strlen(p->name) ? p->name : p->short_url);
		} else {
			applog(LOG_BLUE, "Switch to pool %d: %s", cur_pooln,
				strlen(p->name) ? p->name : p->short_url);
		}

		// will unlock the longpoll thread on /LP url receive
		want_longpoll = (p->type & POOL_LONGPOLL) || !(p->type & POOL_STRATUM);
		if (want_longpoll) {
			pthread_mutex_lock(&stratum_work_lock);
			// will issue a lp_url request to unlock the longpoll thread
			have_longpoll = false;
			get_work(&thr_info[0], &g_work);
			pthread_mutex_unlock(&stratum_work_lock);
		}

	}

	stratum.rpc2 = (p->algo == ALGO_WILDKECCAK || p->algo == ALGO_CRYPTONIGHT);
	stratum.rpc2 |= p->algo == ALGO_CRYPTOLIGHT;

	return true;
}

// search available pool
int pool_get_first_valid(int startfrom)
{
	int next = 0;
	struct pool_infos *p;
	for (int i=0; i<num_pools; i++) {
		int pooln = (startfrom + i) % num_pools;
		p = &pools[pooln];
		if (!(p->status & POOL_ST_VALID))
			continue;
		if (p->status & (POOL_ST_DISABLED | POOL_ST_REMOVED))
			continue;
		next = pooln;
		break;
	}
	return next;
}

// switch to next available pool
bool pool_switch_next(int thr_id)
{
	if (num_pools > 1) {
		int pooln = pool_get_first_valid(cur_pooln+1);
		return pool_switch(thr_id, pooln);
	} else {
		// no switch possible
		if (!opt_quiet)
			applog(LOG_DEBUG, "No other pools to try...");
		return false;
	}
}

// seturl from api remote (deprecated)
bool pool_switch_url(char *params)
{
	int prevn = cur_pooln, nextn;
	parse_arg('o', params);
	// cur_pooln modified by parse_arg('o'), get new pool num
	nextn = cur_pooln;
	// and to handle the "hot swap" from current one...
	cur_pooln = prevn;
	if (nextn == prevn)
		return false;
	return pool_switch(-1, nextn);
}

// Parse pools array in json config
bool parse_pool_array(json_t *obj)
{
	size_t idx;
	json_t *p, *val;

	if (!json_is_array(obj))
		return false;

	// array of objects [ {}, {} ]
	json_array_foreach(obj, idx, p)
	{
		if (!json_is_object(p))
			continue;

		for (int i = 0; i < ARRAY_SIZE(cfg_array_keys); i++)
		{
			int opt = -1;
			char *s = NULL;
			if (cfg_array_keys[i].cat != CFG_POOL)
				continue;

			val = json_object_get(p, cfg_array_keys[i].name);
			if (!val)
				continue;

			for (int k = 0; k < options_count(); k++)
			{
				const char *alias = cfg_array_keys[i].longname;
				if (alias && !strcasecmp(options[k].name, alias)) {
					opt = k;
					break;
				}
				if (!alias && !strcasecmp(options[k].name, cfg_array_keys[i].name)) {
					opt = k;
					break;
				}
			}
			if (opt == -1)
				continue;

			if (json_is_string(val)) {
				s = strdup(json_string_value(val));
				if (!s)
					continue;

				// applog(LOG_DEBUG, "pool key %s '%s'", options[opt].name, s);
				parse_arg(options[opt].val, s);
				free(s);
			} else {
				// numeric or bool
				char buf[32] = { 0 };
				double d = 0.;
				if (json_is_true(val)) d = 1.;
				else if (json_is_integer(val))
					d = 1.0 * json_integer_value(val);
				else if (json_is_real(val))
					d = json_real_value(val);
				snprintf(buf, sizeof(buf)-1, "%f", d);
				// applog(LOG_DEBUG, "pool key %s '%f'", options[opt].name, d);
				parse_arg(options[opt].val, buf);
			}
		}
	}
	return true;
}

// debug stuff
void pool_dump_infos()
{
	struct pool_infos *p;
	if (opt_benchmark) return;
	for (int i=0; i<num_pools; i++) {
		p = &pools[i];
		applog(LOG_DEBUG, "POOL %01d: %s USER %s -s %d", i,
			p->short_url, p->user, p->scantime);
	}
}
