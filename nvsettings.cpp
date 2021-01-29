/**
 * nvidia-settings command line interface for linux - tpruvot 2017
 *
 * Notes: need X setup and running, with an opened X session.
 *        init speed could be improved, running multiple threads
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h> // pid_t

#include "miner.h"
#include "nvml.h"
#include "cuda_runtime.h"

#ifdef __linux__

#define NVS_PATH "/usr/bin/nvidia-settings"

static int8_t nvs_dev_map[MAX_GPUS] = { 0 };
static uint8_t nvs_bus_ids[MAX_GPUS] = { 0 };
static int32_t nvs_clocks_set[MAX_GPUS] = { 0 };

extern int32_t device_mem_offsets[MAX_GPUS];

#if 0 /* complicated exec way and not better in fine */
int nvs_query_fork_int(int nvs_id, const char* field)
{
	pid_t pid;
	int pipes[2] = { 0 };
	if (pipe(pipes) < 0)
		return -1;

	if ((pid = fork()) == -1) {
		close(pipes[0]);
		close(pipes[1]);
		return -1;
	} else if (pid == 0) {
		char gpu_field[128] = { 0 };
		sprintf(gpu_field, "[gpu:%d]/%s", nvs_id, field);

		dup2(pipes[1], STDOUT_FILENO);
		close(pipes[0]);
		//close(pipes[1]);

		if (-1 == execl(NVS_PATH, "nvidia-settings", "-q", gpu_field, "-t", NULL)) {
			exit(-1);
		}
	} else {
		int intval = -1;
		FILE *p = fdopen(pipes[0], "r");
		close(pipes[1]);
		if (!p) {
			applog(LOG_WARNING, "%s: fdopen(%d) failed", __func__, pipes[0]);
			return -1;
		}
		int rc = fscanf(p, "%d", &intval); // BUS 0000:2a:00.0 is read 42
		if (rc > 0) {
			//applog(LOG_BLUE, "%s res=%d", field, intval);
		}
		fclose(p);
		close(pipes[0]);
		return intval;
	}
	return -1;
}
#endif

int nvs_query_int(int nvs_id, const char* field, int showerr)
{
	FILE *fp;
	char command[256] = { 0 };
	sprintf(command, "%s -t -q '[gpu:%d]/%s' 2>&1", NVS_PATH, nvs_id, field);
	fp = popen(command, "r");
	if (fp) {
		int intval = -1;
		if (!showerr) {
			int b = fscanf(fp, "%d", &intval);
			if (!b) {
				pclose(fp);
				return -1;
			}
		} else {
			char msg[512] = { 0 };
			char buf[64] = { 0 };
			ssize_t bytes, len=0, maxlen=sizeof(msg)-1;
			while ((bytes=fscanf(fp, "%s", buf)) > 0) {
				len += snprintf(&msg[len], maxlen-len, "%s ", buf);
				if (len >= maxlen) break;
			}
			if (strstr(msg, "ERROR")) {
				char *xtra = strstr(msg, "; please run");
				if (xtra) *xtra = '\0'; // strip noise
				applog(LOG_INFO, "%s", msg);
				intval = -1;
			} else {
				sscanf(msg, "%d", &intval);
			}
		}
		pclose(fp);
		return intval;
	}
	return -1;
}

int nvs_query_str(int nvs_id, const char* field, char* output, size_t maxlen)
{
	FILE *fp;
	char command[256] = { 0 };
	*output = '\0';
	sprintf(command, "%s -t -q '[gpu:%d]/%s' 2>&1", NVS_PATH, nvs_id, field);
	fp = popen(command, "r");
	if (fp) {
		char buf[256] = { 0 };
		ssize_t len=0;
	        ssize_t bytes=0;
		while ((bytes=fscanf(fp, "%s", buf)) > 0) {
			//applog(LOG_BLUE, "%d %s %d", nvs_id, buf, (int) bytes);
			len += snprintf(&output[len], maxlen-len, "%s ", buf);
			if (len >= maxlen) break;
		}
		pclose(fp);
		if (strstr(output, "ERROR")) {
			char *xtra = strstr(output, "; please run");
			if (xtra) *xtra = '\0'; // strip noise
			applog(LOG_INFO, "%s", output);
			*output='\0';
		}
		return (int) len;
	}
	return -1;
}

int nvs_set_int(int nvs_id, const char* field, int value)
{
	FILE *fp;
	char command[256] = { 0 };
	int res = -1;
	snprintf(command, 256, "%s -a '[gpu:%d]/%s=%d' 2>&1", NVS_PATH, nvs_id, field, value);
	fp = popen(command, "r");
	if (fp) {
		char msg[512] = { 0 };
		char buf[64] = { 0 };
		ssize_t bytes, len=0, maxlen=sizeof(msg)-1;
		while ((bytes=fscanf(fp, "%s", buf)) > 0) {
			len += snprintf(&msg[len], maxlen-len, "%s ", buf);
			if (len >= maxlen) break;
		}
		if (strstr(msg, "ERROR")) {
			char *xtra = strstr(msg, "; please run");
			if (xtra) *xtra = '\0'; // strip noise
			applog(LOG_INFO, "%s", msg);
		} else
			res = 0;
		pclose(fp);
	}
	return res;
}

int8_t nvs_devnum(int dev_id)
{
	return nvs_dev_map[dev_id];
}

int nvs_devid(int8_t nvs_id)
{
	for (int i=0; i < opt_n_threads; i++) {
		int dev_id = device_map[i % MAX_GPUS];
		if (nvs_dev_map[dev_id] == nvs_id)
			return dev_id;
	}
	return 0;
}

int nvs_init()
{
	struct stat info;
	struct timeval tv_start, tv_end, diff;
	int x_devices = 0;
	int n_threads = opt_n_threads;
	if (stat(NVS_PATH, &info))
		return -ENOENT;

	gettimeofday(&tv_start, NULL);

	for (int d = 0; d < MAX_GPUS; d++) {
		// this part can be "slow" (100-200ms per device)
		int res = nvs_query_int(d, "PCIBus", 1);
		if (res < 0) break;
		nvs_bus_ids[d] = 0xFFu & res;
		x_devices++;
	}

	if (opt_debug) {
		gettimeofday(&tv_end, NULL);
		timeval_subtract(&diff, &tv_end, &tv_start);
		applog(LOG_DEBUG, "nvidia-settings pci bus queries took %.2f ms",
			(1000.0 * diff.tv_sec) + (0.001 * diff.tv_usec));
	}

	if (!x_devices)
		return -ENODEV;
	if (!n_threads) n_threads = cuda_num_devices();
	for (int i = 0; i < n_threads; i++) {
		int dev_id = device_map[i % MAX_GPUS];
		cudaDeviceProp props;
		if (cudaGetDeviceProperties(&props, dev_id) == cudaSuccess) {
			for (int8_t d = 0; d < x_devices; d++) {
				if (nvs_bus_ids[d] == (uint8_t) props.pciBusID) {
					gpulog(LOG_DEBUG, i, "matches X gpu:%d by busId %u",
						(int) d, (uint) nvs_bus_ids[d]);
					nvs_dev_map[dev_id] = d;
					/* char buf[1024] = { 0 };
					nvs_query_str(d, "GPUCurrentClockFreqsString", buf, sizeof(buf)-1);
					gpulog(LOG_DEBUG, d, "%s", buf); */
					break;
				}
			}
		}
	}
	return 0;
}

int nvs_set_clocks(int dev_id)
{
	int res;
	int8_t d = nvs_devnum(dev_id);
	if (d < 0) return -ENODEV;
	if (!device_mem_offsets[dev_id] || nvs_clocks_set[d]) return 0;
	res = nvs_set_int(d, "GPUMemoryTransferRateOffsetAllPerformanceLevels", device_mem_offsets[dev_id]*2);
	if (res) nvs_clocks_set[d] = device_mem_offsets[dev_id]*2;
	return res;
}

void nvs_reset_clocks(int dev_id)
{
	int8_t d = nvs_devnum(dev_id);
	if (d < 0 || !nvs_clocks_set[d]) return;
	nvs_set_int(d, "GPUMemoryTransferRateOffsetAllPerformanceLevels", 0);
	nvs_clocks_set[d] = 0;
}

#else
int nvs_init() { return -ENOSYS; }
int nvs_set_clocks(int dev_id) { return -ENOSYS; }
void nvs_reset_clocks(int dev_id) { }
#endif
