/*
 * A trivial little dlopen()-based wrapper library for the
 * NVIDIA NVML library, to allow runtime discovery of NVML on an
 * arbitrary system.  This is all very hackish and simple-minded, but
 * it serves my immediate needs in the short term until NVIDIA provides
 * a static NVML wrapper library themselves, hopefully in
 * CUDA 6.5 or maybe sometime shortly after.
 *
 * This trivial code is made available under the "new" 3-clause BSD license,
 * and/or any of the GPL licenses you prefer.
 * Feel free to use the code and modify as you see fit.
 *
 * John E. Stone - john.stone@gmail.com
 * Tanguy Pruvot - tpruvot@github
 *
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "miner.h"
#include "nvml.h"
#include "cuda_runtime.h"

#ifdef USE_WRAPNVML

extern nvml_handle *hnvml;
extern char driver_version[32];

static uint32_t device_bus_ids[MAX_GPUS] = { 0 };

extern uint32_t device_gpu_clocks[MAX_GPUS];
extern uint32_t device_mem_clocks[MAX_GPUS];
extern int32_t device_mem_offsets[MAX_GPUS];
extern uint8_t device_tlimit[MAX_GPUS];
extern int8_t device_pstate[MAX_GPUS];
extern int32_t device_led[MAX_GPUS];
int32_t device_led_state[MAX_GPUS] = { 0 };
static __thread bool has_rgb_ok = false;

uint32_t clock_prev[MAX_GPUS] = { 0 };
uint32_t clock_prev_mem[MAX_GPUS] = { 0 };
uint32_t limit_prev[MAX_GPUS] = { 0 };

static bool nvml_plimit_set = false;
extern bool need_memclockrst;

/*
 * Wrappers to emulate dlopen() on other systems like Windows
 */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
	static void *wrap_dlopen(const char *filename) {
		HMODULE h = LoadLibrary(filename);
		if (!h && opt_debug) {
			applog(LOG_DEBUG, "dlopen(%d): failed to load %s", 
				GetLastError(), filename);
		}
		return (void*)h;
	}
	static void *wrap_dlsym(void *h, const char *sym) {
		return (void *)GetProcAddress((HINSTANCE)h, sym);
	}
	static int wrap_dlclose(void *h) {
		/* FreeLibrary returns nonzero on success */
		return (!FreeLibrary((HINSTANCE)h));
	}
#else
	/* assume we can use dlopen itself... */
	#include <dlfcn.h>
	#include <errno.h>
	static void *wrap_dlopen(const char *filename) {
		void *h = dlopen(filename, RTLD_NOW);
		if (h == NULL && opt_debug) {
			applog(LOG_DEBUG, "dlopen(%d): failed to load %s", 
				errno, filename);
		}
		return (void*)h;
	}

	static void *wrap_dlsym(void *h, const char *sym) {
		return dlsym(h, sym);
	}
	static int wrap_dlclose(void *h) {
		return dlclose(h);
	}
#endif

nvml_handle * nvml_create()
{
	int i=0;
	nvml_handle *nvmlh = NULL;

#ifdef WIN32
	/* Windows (do not use slashes, else ExpandEnvironmentStrings will mix them) */
#define  libnvidia_ml "%PROGRAMFILES%\\NVIDIA Corporation\\NVSMI\\nvml.dll"
#else
	/* linux assumed */
#define  libnvidia_ml "libnvidia-ml.so"
#endif

	char tmp[512];
#ifdef WIN32
	ExpandEnvironmentStrings(libnvidia_ml, tmp, sizeof(tmp));
#else
	strcpy(tmp, libnvidia_ml);
#endif

	void *nvml_dll = wrap_dlopen(tmp);
	if (nvml_dll == NULL) {
#ifdef WIN32
		nvml_dll = wrap_dlopen("nvml.dll");
		if (nvml_dll == NULL)
#endif
		return NULL;
	}

	nvmlh = (nvml_handle *) calloc(1, sizeof(nvml_handle));

	nvmlh->nvml_dll = nvml_dll;

	nvmlh->nvmlInit = (nvmlReturn_t (*)(void)) wrap_dlsym(nvmlh->nvml_dll, "nvmlInit_v2");
	if (!nvmlh->nvmlInit)
		nvmlh->nvmlInit = (nvmlReturn_t (*)(void)) wrap_dlsym(nvmlh->nvml_dll, "nvmlInit");
	nvmlh->nvmlDeviceGetCount = (nvmlReturn_t (*)(int *)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
	if (!nvmlh->nvmlDeviceGetCount)
		nvmlh->nvmlDeviceGetCount = (nvmlReturn_t (*)(int *)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount");
	nvmlh->nvmlDeviceGetHandleByIndex = (nvmlReturn_t (*)(int, nvmlDevice_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
	nvmlh->nvmlDeviceGetAPIRestriction = (nvmlReturn_t (*)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetAPIRestriction");
	nvmlh->nvmlDeviceSetAPIRestriction = (nvmlReturn_t (*)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceSetAPIRestriction");
	nvmlh->nvmlDeviceGetDefaultApplicationsClock = (nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *clock))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetDefaultApplicationsClock");
	nvmlh->nvmlDeviceGetApplicationsClock = (nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *clocks))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetApplicationsClock");
	nvmlh->nvmlDeviceSetApplicationsClocks = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int mem, unsigned int gpu))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceSetApplicationsClocks");
	nvmlh->nvmlDeviceResetApplicationsClocks = (nvmlReturn_t (*)(nvmlDevice_t))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceResetApplicationsClocks");
	nvmlh->nvmlDeviceGetSupportedGraphicsClocks = (nvmlReturn_t (*)(nvmlDevice_t, uint32_t mem, uint32_t *num, uint32_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetSupportedGraphicsClocks");
	nvmlh->nvmlDeviceGetSupportedMemoryClocks = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *count, unsigned int *clocksMHz))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetSupportedMemoryClocks");
	nvmlh->nvmlDeviceGetClockInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *clock))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetClockInfo");
	nvmlh->nvmlDeviceGetMaxClockInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *clock))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetMaxClockInfo");
	nvmlh->nvmlDeviceGetPciInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo_v2");
	if (!nvmlh->nvmlDeviceGetPciInfo)
		nvmlh->nvmlDeviceGetPciInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *)) wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo");
	nvmlh->nvmlDeviceGetCurrPcieLinkGeneration = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *gen))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCurrPcieLinkGeneration");
	nvmlh->nvmlDeviceGetCurrPcieLinkWidth = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *width))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCurrPcieLinkWidth");
	nvmlh->nvmlDeviceGetMaxPcieLinkGeneration = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *gen))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetMaxPcieLinkGeneration");
	nvmlh->nvmlDeviceGetMaxPcieLinkWidth = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *width))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetMaxPcieLinkWidth");
	nvmlh->nvmlDeviceGetPowerUsage = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
	nvmlh->nvmlDeviceGetPowerManagementDefaultLimit = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *limit))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerManagementDefaultLimit");
	nvmlh->nvmlDeviceGetPowerManagementLimit = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *limit))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerManagementLimit");
	nvmlh->nvmlDeviceGetPowerManagementLimitConstraints = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *min, unsigned int *max))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerManagementLimitConstraints");
	nvmlh->nvmlDeviceSetPowerManagementLimit = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int limit))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceSetPowerManagementLimit");
	nvmlh->nvmlDeviceGetName = (nvmlReturn_t (*)(nvmlDevice_t, char *, int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetName");
	nvmlh->nvmlDeviceGetTemperature = (nvmlReturn_t (*)(nvmlDevice_t, int, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetTemperature");
	nvmlh->nvmlDeviceGetFanSpeed = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFanSpeed");
	nvmlh->nvmlDeviceGetPerformanceState = (nvmlReturn_t (*)(nvmlDevice_t, int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPerformanceState"); /* or nvmlDeviceGetPowerState */
	nvmlh->nvmlDeviceGetSerial = (nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetSerial");
	nvmlh->nvmlDeviceGetUUID = (nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetUUID");
	nvmlh->nvmlDeviceGetVbiosVersion = (nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetVbiosVersion");
	nvmlh->nvmlSystemGetDriverVersion = (nvmlReturn_t (*)(char *, unsigned int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlSystemGetDriverVersion");
	nvmlh->nvmlErrorString = (char* (*)(nvmlReturn_t))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlErrorString");
	nvmlh->nvmlShutdown = (nvmlReturn_t (*)())
		wrap_dlsym(nvmlh->nvml_dll, "nvmlShutdown");
	// v331
	nvmlh->nvmlDeviceGetEnforcedPowerLimit = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *limit))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetEnforcedPowerLimit");
	// v340
#ifdef __linux__
	nvmlh->nvmlDeviceClearCpuAffinity = (nvmlReturn_t (*)(nvmlDevice_t))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceClearCpuAffinity");
	nvmlh->nvmlDeviceGetCpuAffinity = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int sz, unsigned long *cpuSet))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCpuAffinity");
	nvmlh->nvmlDeviceSetCpuAffinity = (nvmlReturn_t (*)(nvmlDevice_t))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceSetCpuAffinity");
#endif
	// v346
	nvmlh->nvmlDeviceGetPcieThroughput = (nvmlReturn_t (*)(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int *value))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPcieThroughput");
	// v36x (API 8 / Pascal)
	nvmlh->nvmlDeviceGetClock = (nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetClock");

	if (nvmlh->nvmlInit == NULL ||
			nvmlh->nvmlShutdown == NULL ||
			nvmlh->nvmlErrorString == NULL ||
			nvmlh->nvmlDeviceGetCount == NULL ||
			nvmlh->nvmlDeviceGetHandleByIndex == NULL ||
			nvmlh->nvmlDeviceGetPciInfo == NULL ||
			nvmlh->nvmlDeviceGetName == NULL)
	{
		if (opt_debug)
			applog(LOG_DEBUG, "Failed to obtain required NVML function pointers");
		wrap_dlclose(nvmlh->nvml_dll);
		free(nvmlh);
		return NULL;
	}

	nvmlh->nvmlInit();
	if (nvmlh->nvmlSystemGetDriverVersion)
		nvmlh->nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version));
	nvmlh->nvmlDeviceGetCount(&nvmlh->nvml_gpucount);

	/* Query CUDA device count, in case it doesn't agree with NVML, since  */
	/* CUDA will only report GPUs with compute capability greater than 1.0 */
	if (cudaGetDeviceCount(&nvmlh->cuda_gpucount) != cudaSuccess) {
		if (opt_debug)
			applog(LOG_DEBUG, "Failed to query CUDA device count!");
		wrap_dlclose(nvmlh->nvml_dll);
		free(nvmlh);
		return NULL;
	}

	nvmlh->devs = (nvmlDevice_t *) calloc(nvmlh->nvml_gpucount, sizeof(nvmlDevice_t));
	nvmlh->nvml_pci_domain_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_bus_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_device_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_vendor_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_subsys_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_cuda_device_id = (int*) calloc(nvmlh->nvml_gpucount, sizeof(int));
	nvmlh->cuda_nvml_device_id = (int*) calloc(nvmlh->cuda_gpucount, sizeof(int));
	nvmlh->app_clocks = (nvmlEnableState_t*) calloc(nvmlh->nvml_gpucount, sizeof(nvmlEnableState_t));

	/* Obtain GPU device handles we're going to need repeatedly... */
	for (i=0; i<nvmlh->nvml_gpucount; i++) {
		nvmlh->nvmlDeviceGetHandleByIndex(i, &nvmlh->devs[i]);
	}

	/* Query PCI info for each NVML device, and build table for mapping of */
	/* CUDA device IDs to NVML device IDs and vice versa                   */
	for (i=0; i<nvmlh->nvml_gpucount; i++) {
		nvmlPciInfo_t pciinfo;

		nvmlh->nvmlDeviceGetPciInfo(nvmlh->devs[i], &pciinfo);
		nvmlh->nvml_pci_domain_id[i] = pciinfo.domain;
		nvmlh->nvml_pci_bus_id[i]    = pciinfo.bus;
		nvmlh->nvml_pci_device_id[i] = pciinfo.device;
		nvmlh->nvml_pci_vendor_id[i] = pciinfo.pci_device_id;
		nvmlh->nvml_pci_subsys_id[i] = pciinfo.pci_subsystem_id;

		nvmlh->app_clocks[i] = NVML_FEATURE_UNKNOWN;
		if (nvmlh->nvmlDeviceSetAPIRestriction) {
			nvmlh->nvmlDeviceSetAPIRestriction(nvmlh->devs[i], NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
				NVML_FEATURE_ENABLED);
			/* there is only this API_SET_APPLICATION_CLOCKS on the 750 Ti (340.58) */
		}
		if (nvmlh->nvmlDeviceGetAPIRestriction) {
			nvmlh->nvmlDeviceGetAPIRestriction(nvmlh->devs[i], NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
				&nvmlh->app_clocks[i]);
		}
	}

	/* build mapping of NVML device IDs to CUDA IDs */
	for (i=0; i<nvmlh->nvml_gpucount; i++) {
		nvmlh->nvml_cuda_device_id[i] = -1;
	}
	for (i=0; i<nvmlh->cuda_gpucount; i++) {
		cudaDeviceProp props;
		nvmlh->cuda_nvml_device_id[i] = -1;

		if (cudaGetDeviceProperties(&props, i) == cudaSuccess) {
			device_bus_ids[i] = props.pciBusID;
			for (int j = 0; j < nvmlh->nvml_gpucount; j++) {
				if ((nvmlh->nvml_pci_domain_id[j] == (uint32_t) props.pciDomainID) &&
				    (nvmlh->nvml_pci_bus_id[j]    == (uint32_t) props.pciBusID) &&
				    (nvmlh->nvml_pci_device_id[j] == (uint32_t) props.pciDeviceID)) {
					if (opt_debug)
						applog(LOG_DEBUG, "CUDA GPU %d matches NVML GPU %d by busId %u",
							i, j, (uint32_t) props.pciBusID);
					nvmlh->nvml_cuda_device_id[j] = i;
					nvmlh->cuda_nvml_device_id[i] = j;
				}
			}
		}
	}

	return nvmlh;
}

/* apply config clocks to an used device */
int nvml_set_clocks(nvml_handle *nvmlh, int dev_id)
{
	nvmlReturn_t rc;
	uint32_t gpu_clk = 0, mem_clk = 0;
	int n = nvmlh->cuda_nvml_device_id[dev_id];
	//if (need_nvsettings) /* prefer later than init time */
	//	nvs_set_clocks(dev_id);
	if (n < 0 || n >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!device_gpu_clocks[dev_id] && !device_mem_clocks[dev_id])
		return 0; // nothing to do

	if (nvmlh->app_clocks[n] != NVML_FEATURE_ENABLED) {
		applog(LOG_WARNING, "GPU #%d: NVML application clock feature is not allowed!", dev_id);
		return -EPERM;
	}

	uint32_t mem_prev = clock_prev_mem[dev_id];
	if (!mem_prev)
		nvmlh->nvmlDeviceGetApplicationsClock(nvmlh->devs[n], NVML_CLOCK_MEM, &mem_prev);
	uint32_t gpu_prev = clock_prev[dev_id];
	if (!gpu_prev)
		nvmlh->nvmlDeviceGetApplicationsClock(nvmlh->devs[n], NVML_CLOCK_GRAPHICS, &gpu_prev);

	nvmlh->nvmlDeviceGetDefaultApplicationsClock(nvmlh->devs[n], NVML_CLOCK_MEM, &mem_clk);
	rc = nvmlh->nvmlDeviceGetDefaultApplicationsClock(nvmlh->devs[n], NVML_CLOCK_GRAPHICS, &gpu_clk);
	if (rc != NVML_SUCCESS) {
		applog(LOG_WARNING, "GPU #%d: unable to query application clocks", dev_id);
		return -EINVAL;
	}

	if (opt_debug)
		applog(LOG_DEBUG, "GPU #%d: default application clocks are %u/%u", dev_id, mem_clk, gpu_clk);

	// get application config values
	if (device_mem_clocks[dev_id]) mem_clk = device_mem_clocks[dev_id];
	if (device_gpu_clocks[dev_id]) gpu_clk = device_gpu_clocks[dev_id];

	// these functions works for the 960 and the 970 (346.72+), and for the 750 Ti with driver ~361+
	uint32_t nclocks = 0, mem_clocks[32] = { 0 };
	nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, NULL);
	nclocks = min(nclocks, 32);
	if (nclocks)
		nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, mem_clocks);
	for (uint8_t u=0; u < nclocks; u++) {
		// ordered by pstate (so highest is first memory clock - P0)
		if (mem_clocks[u] <= mem_clk) {
			mem_clk = mem_clocks[u];
			break;
		}
	}

	uint32_t* gpu_clocks = NULL;
	nclocks = 0;
	nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, NULL);
	if (nclocks) {
		if (opt_debug)
			applog(LOG_DEBUG, "GPU #%d: %u clocks found for mem %u", dev_id, nclocks, mem_clk);
		gpu_clocks = (uint32_t*) calloc(1, sizeof(uint32_t) * nclocks + 4);
		nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, gpu_clocks);
		for (uint8_t u=0; u < nclocks; u++) {
			// ordered desc, so get first
			if (gpu_clocks[u] <= gpu_clk) {
				gpu_clk = gpu_clocks[u];
				break;
			}
		}
		free(gpu_clocks);
	}

	rc = nvmlh->nvmlDeviceSetApplicationsClocks(nvmlh->devs[n], mem_clk, gpu_clk);
	if (rc == NVML_SUCCESS)
		applog(LOG_INFO, "GPU #%d: application clocks set to %u/%u", dev_id, mem_clk, gpu_clk);
	else {
		applog(LOG_WARNING, "GPU #%d: %u/%u - %s (NVML)", dev_id, mem_clk, gpu_clk, nvmlh->nvmlErrorString(rc));
		return -1;
	}

	// store previous clocks for reset on exit (or during wait...)
	clock_prev[dev_id] = gpu_prev;
	clock_prev_mem[dev_id] = mem_prev;
	return 1;
}

/* reset default app clocks and limits on exit */
int nvml_reset_clocks(nvml_handle *nvmlh, int dev_id)
{
	int ret = 0;
	nvmlReturn_t rc;
	uint32_t gpu_clk = 0, mem_clk = 0;
	int n = nvmlh->cuda_nvml_device_id[dev_id];
	if (need_nvsettings)
		nvs_reset_clocks(dev_id);
	if (n < 0 || n >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (clock_prev[dev_id]) {
		rc = nvmlh->nvmlDeviceResetApplicationsClocks(nvmlh->devs[n]);
		if (rc != NVML_SUCCESS) {
			applog(LOG_WARNING, "GPU #%d: unable to reset application clocks", dev_id);
		}
		clock_prev[dev_id] = 0;
		ret = 1;
	}

	if (limit_prev[dev_id]) {
		uint32_t plimit = limit_prev[dev_id];
		if (nvmlh->nvmlDeviceGetPowerManagementDefaultLimit && !plimit) {
			rc = nvmlh->nvmlDeviceGetPowerManagementDefaultLimit(nvmlh->devs[n], &plimit);
		} else if (plimit) {
			rc = NVML_SUCCESS;
		}
		if (rc == NVML_SUCCESS)
			nvmlh->nvmlDeviceSetPowerManagementLimit(nvmlh->devs[n], plimit);
		ret = 1;
	}
	return ret;
}

/**
 * Set power state of a device (9xx)
 * Code is similar as clocks one, which allow the change of the pstate
 */
int nvml_set_pstate(nvml_handle *nvmlh, int dev_id)
{
	nvmlReturn_t rc;
	uint32_t gpu_clk = 0, mem_clk = 0;
	int n = nvmlh->cuda_nvml_device_id[dev_id];
	if (n < 0 || n >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (device_pstate[dev_id] < 0)
		return 0;

	if (nvmlh->app_clocks[n] != NVML_FEATURE_ENABLED) {
		applog(LOG_WARNING, "GPU #%d: NVML app. clock feature is not allowed!", dev_id);
		return -EPERM;
	}

	nvmlh->nvmlDeviceGetDefaultApplicationsClock(nvmlh->devs[n], NVML_CLOCK_MEM, &mem_clk);
	rc = nvmlh->nvmlDeviceGetDefaultApplicationsClock(nvmlh->devs[n], NVML_CLOCK_GRAPHICS, &gpu_clk);
	if (rc != NVML_SUCCESS) {
		applog(LOG_WARNING, "GPU #%d: unable to query application clocks", dev_id);
		return -EINVAL;
	}

	// get application config values
	if (device_mem_clocks[dev_id]) mem_clk = device_mem_clocks[dev_id];
	if (device_gpu_clocks[dev_id]) gpu_clk = device_gpu_clocks[dev_id];

	// these functions works for the 960 and the 970 (346.72+), and for the 750 Ti with driver ~361+
	uint32_t nclocks = 0, mem_clocks[32] = { 0 };
	int8_t wanted_pstate = device_pstate[dev_id];
	nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, NULL);
	nclocks = min(nclocks, 32);
	if (nclocks)
		nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, mem_clocks);
	if ((uint32_t) wanted_pstate+1 > nclocks) {
		applog(LOG_WARNING, "GPU #%d: only %u mem clocks available (p-states)", dev_id, nclocks);
	}
	for (uint8_t u=0; u < nclocks; u++) {
		// ordered by pstate (so highest P0 first)
		if (u == wanted_pstate) {
			mem_clk = mem_clocks[u];
			break;
		}
	}

	uint32_t* gpu_clocks = NULL;
	nclocks = 0;
	nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, NULL);
	if (nclocks) {
		gpu_clocks = (uint32_t*) calloc(1, sizeof(uint32_t) * nclocks + 4);
		rc = nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, gpu_clocks);
		if (rc == NVML_SUCCESS) {
			// ordered desc, get the max app clock (do not limit)
			gpu_clk = gpu_clocks[0];
		}
		free(gpu_clocks);
	}

	rc = nvmlh->nvmlDeviceSetApplicationsClocks(nvmlh->devs[n], mem_clk, gpu_clk);
	if (rc != NVML_SUCCESS) {
		applog(LOG_WARNING, "GPU #%d: pstate P%d (%u/%u) %s", dev_id, (int) wanted_pstate,
			mem_clk, gpu_clk, nvmlh->nvmlErrorString(rc));
		return -1;
	}

	if (!opt_quiet)
		applog(LOG_INFO, "GPU #%d: app clocks set to P%d (%u/%u)", dev_id, (int) wanted_pstate, mem_clk, gpu_clk);

	clock_prev[dev_id] = 1;
	return 1;
}

int nvml_set_plimit(nvml_handle *nvmlh, int dev_id)
{
	nvmlReturn_t rc = NVML_ERROR_UNKNOWN;
	uint32_t gpu_clk = 0, mem_clk = 0;
	int n = nvmlh->cuda_nvml_device_id[dev_id];
	if (n < 0 || n >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!device_plimit[dev_id])
		return 0; // nothing to do

	if (!nvmlh->nvmlDeviceSetPowerManagementLimit)
		return -ENOSYS;

	uint32_t plimit = device_plimit[dev_id] * 1000;
	uint32_t pmin = 1000, pmax = 0, prev_limit = 0;
	if (nvmlh->nvmlDeviceGetPowerManagementLimitConstraints)
		rc = nvmlh->nvmlDeviceGetPowerManagementLimitConstraints(nvmlh->devs[n], &pmin, &pmax);

	if (rc != NVML_SUCCESS) {
		if (!nvmlh->nvmlDeviceGetPowerManagementLimit)
			return -ENOSYS;
	}
	nvmlh->nvmlDeviceGetPowerManagementLimit(nvmlh->devs[n], &prev_limit);
	if (!pmax) pmax = prev_limit;

	plimit = min(plimit, pmax);
	plimit = max(plimit, pmin);
	rc = nvmlh->nvmlDeviceSetPowerManagementLimit(nvmlh->devs[n], plimit);
	if (rc != NVML_SUCCESS) {
#ifndef WIN32
		applog(LOG_WARNING, "GPU #%d: plimit %s", dev_id, nvmlh->nvmlErrorString(rc));
#endif
		return -1;
	} else {
		device_plimit[dev_id] = plimit / 1000;
		nvml_plimit_set = true;
	}

	if (!opt_quiet) {
		applog(LOG_INFO, "GPU #%d: power limit set to %uW (allowed range is %u-%u)",
			dev_id, plimit/1000U, pmin/1000U, pmax/1000U);
	}

	limit_prev[dev_id] = prev_limit;
	return 1;
}

uint32_t nvml_get_plimit(nvml_handle *nvmlh, int dev_id)
{
	uint32_t plimit = 0;
	int n = nvmlh ? nvmlh->cuda_nvml_device_id[dev_id] : -1;
	if (n < 0 || n >= nvmlh->nvml_gpucount)
		return 0;

	if (nvmlh->nvmlDeviceGetPowerManagementLimit) {
		nvmlh->nvmlDeviceGetPowerManagementLimit(nvmlh->devs[n], &plimit);
	}
	return plimit;
}

// ccminer -D -n
#define LSTDEV_PFX "        "
void nvml_print_device_info(int dev_id)
{
	if (!hnvml) return;

	int n = hnvml->cuda_nvml_device_id[dev_id];
	if (n < 0 || n >= hnvml->nvml_gpucount)
		return;

	nvmlReturn_t rc;

	// fprintf(stderr, "------ Hardware ------\n");
	int gvid = hnvml->nvml_pci_vendor_id[n] & 0xFFFF;
	int gpid = hnvml->nvml_pci_vendor_id[n] >> 16;
	int svid = hnvml->nvml_pci_subsys_id[n] & 0xFFFF;
	int spid = hnvml->nvml_pci_subsys_id[n] >> 16;

	fprintf(stderr, LSTDEV_PFX "ID %04x:%04x/%04x:%04x BUS %04x:%02x:%02x.0\n", gvid, gpid, svid, spid,
		(int) hnvml->nvml_pci_domain_id[n], (int) hnvml->nvml_pci_bus_id[n], (int) hnvml->nvml_pci_device_id[n]);

	if (hnvml->nvmlDeviceGetClock) {
		uint32_t gpu_clk = 0, mem_clk = 0;

		// fprintf(stderr, "------- Clocks -------\n");

		hnvml->nvmlDeviceGetClock(hnvml->devs[n], NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, &gpu_clk);
		rc = hnvml->nvmlDeviceGetClock(hnvml->devs[n], NVML_CLOCK_MEM, NVML_CLOCK_ID_APP_CLOCK_DEFAULT, &mem_clk);
		if (rc == NVML_SUCCESS) {
			fprintf(stderr, LSTDEV_PFX "DEFAULT MEM %4u GPU %4u MHz\n", mem_clk, gpu_clk);
		}
		hnvml->nvmlDeviceGetClock(hnvml->devs[n], NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_APP_CLOCK_TARGET, &gpu_clk);
		rc = hnvml->nvmlDeviceGetClock(hnvml->devs[n], NVML_CLOCK_MEM, NVML_CLOCK_ID_APP_CLOCK_TARGET, &mem_clk);
		if (rc == NVML_SUCCESS) {
			fprintf(stderr, LSTDEV_PFX "TARGET  MEM %4u GPU %4u MHz\n", mem_clk, gpu_clk);
		}
		hnvml->nvmlDeviceGetClock(hnvml->devs[n], NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &gpu_clk);
		rc = hnvml->nvmlDeviceGetClock(hnvml->devs[n], NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &mem_clk);
		if (rc == NVML_SUCCESS) {
			fprintf(stderr, LSTDEV_PFX "CURRENT MEM %4u GPU %4u MHz\n", mem_clk, gpu_clk);
		}
	}
}

int nvml_get_gpucount(nvml_handle *nvmlh, int *gpucount)
{
	*gpucount = nvmlh->nvml_gpucount;
	return 0;
}

int cuda_get_gpucount(nvml_handle *nvmlh, int *gpucount)
{
	*gpucount = nvmlh->cuda_gpucount;
	return 0;
}


int nvml_get_gpu_name(nvml_handle *nvmlh, int cudaindex, char *namebuf, int bufsize)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!nvmlh->nvmlDeviceGetName)
		return -ENOSYS;

	if (nvmlh->nvmlDeviceGetName(nvmlh->devs[gpuindex], namebuf, bufsize) != NVML_SUCCESS)
		return -1;

	return 0;
}


int nvml_get_tempC(nvml_handle *nvmlh, int cudaindex, unsigned int *tempC)
{
	nvmlReturn_t rc;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!nvmlh->nvmlDeviceGetTemperature)
		return -ENOSYS;

	rc = nvmlh->nvmlDeviceGetTemperature(nvmlh->devs[gpuindex], 0u /* NVML_TEMPERATURE_GPU */, tempC);
	if (rc != NVML_SUCCESS) {
		return -1;
	}

	return 0;
}


int nvml_get_fanpcnt(nvml_handle *nvmlh, int cudaindex, unsigned int *fanpcnt)
{
	nvmlReturn_t rc;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!nvmlh->nvmlDeviceGetFanSpeed)
		return -ENOSYS;

	rc = nvmlh->nvmlDeviceGetFanSpeed(nvmlh->devs[gpuindex], fanpcnt);
	if (rc != NVML_SUCCESS) {
		return -1;
	}

	return 0;
}


int nvml_get_current_clocks(int cudaindex, unsigned int *graphics_clock, unsigned int *mem_clock)
{
	nvmlReturn_t rc;
	int gpuindex = hnvml->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= hnvml->nvml_gpucount) return -ENODEV;
	if (!hnvml->nvmlDeviceGetClockInfo) return -ENOSYS;

	rc = hnvml->nvmlDeviceGetClockInfo(hnvml->devs[gpuindex], NVML_CLOCK_SM, graphics_clock);
	if (rc != NVML_SUCCESS) return -1;
	rc = hnvml->nvmlDeviceGetClockInfo(hnvml->devs[gpuindex], NVML_CLOCK_MEM, mem_clock);
	if (rc != NVML_SUCCESS) return -1;

	return 0;
}

/* Not Supported on 750Ti 340.23 */
int nvml_get_power_usage(nvml_handle *nvmlh, int cudaindex, unsigned int *milliwatts)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!nvmlh->nvmlDeviceGetPowerUsage)
		return -ENOSYS;

	nvmlReturn_t res = nvmlh->nvmlDeviceGetPowerUsage(nvmlh->devs[gpuindex], milliwatts);
	if (res != NVML_SUCCESS) {
		//if (opt_debug)
		//	applog(LOG_DEBUG, "nvmlDeviceGetPowerUsage: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

/* Not Supported on 750Ti 340.23 */
int nvml_get_pstate(nvml_handle *nvmlh, int cudaindex, int *pstate)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!nvmlh->nvmlDeviceGetPerformanceState)
		return -ENOSYS;

	nvmlReturn_t res = nvmlh->nvmlDeviceGetPerformanceState(nvmlh->devs[gpuindex], pstate);
	if (res != NVML_SUCCESS) {
		//if (opt_debug)
		//	applog(LOG_DEBUG, "nvmlDeviceGetPerformanceState: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

int nvml_get_busid(nvml_handle *nvmlh, int cudaindex, int *busid)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	(*busid) = nvmlh->nvml_pci_bus_id[gpuindex];
	return 0;
}

int nvml_get_serial(nvml_handle *nvmlh, int cudaindex, char *sn, int maxlen)
{
	uint32_t subids = 0;
	char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	nvmlReturn_t res;
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (nvmlh->nvmlDeviceGetSerial) {
		res = nvmlh->nvmlDeviceGetSerial(nvmlh->devs[gpuindex], sn, maxlen);
		if (res == NVML_SUCCESS)
			return 0;
	}

	if (!nvmlh->nvmlDeviceGetUUID)
		return -ENOSYS;

	// nvmlDeviceGetUUID: GPU-f2bd642c-369f-5a14-e0b4-0d22dfe9a1fc
	// use a part of uuid to generate an unique serial
	// todo: check if there is vendor id is inside
	memset(uuid, 0, sizeof(uuid));
	res = nvmlh->nvmlDeviceGetUUID(nvmlh->devs[gpuindex], uuid, sizeof(uuid)-1);
	if (res != NVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetUUID: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}
	strncpy(sn, &uuid[4], min((int) strlen(uuid), maxlen));
	sn[maxlen-1] = '\0';
	return 0;
}

int nvml_get_bios(nvml_handle *nvmlh, int cudaindex, char *desc, int maxlen)
{
	uint32_t subids = 0;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	if (!nvmlh->nvmlDeviceGetVbiosVersion)
		return -ENOSYS;

	nvmlReturn_t res = nvmlh->nvmlDeviceGetVbiosVersion(nvmlh->devs[gpuindex], desc, maxlen);
	if (res != NVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetVbiosVersion: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}
	return 0;
}

int nvml_get_info(nvml_handle *nvmlh, int cudaindex, uint16_t &vid, uint16_t &pid)
{
	uint32_t subids = 0;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -ENODEV;

	subids = nvmlh->nvml_pci_subsys_id[gpuindex];
	if (!subids) subids = nvmlh->nvml_pci_vendor_id[gpuindex];
	pid = subids >> 16;
	vid = subids & 0xFFFF;
	// Colorful and Inno3D
	if (pid == 0) pid = nvmlh->nvml_pci_vendor_id[gpuindex] >> 16;
	return 0;
}

int nvml_destroy(nvml_handle *nvmlh)
{
	nvmlh->nvmlShutdown();

	wrap_dlclose(nvmlh->nvml_dll);

	free(nvmlh->nvml_pci_bus_id);
	free(nvmlh->nvml_pci_device_id);
	free(nvmlh->nvml_pci_domain_id);
	free(nvmlh->nvml_pci_vendor_id);
	free(nvmlh->nvml_pci_subsys_id);
	free(nvmlh->nvml_cuda_device_id);
	free(nvmlh->cuda_nvml_device_id);
	free(nvmlh->app_clocks);
	free(nvmlh->devs);

	free(nvmlh);
	return 0;
}

// ----------------------------------------------------------------------------

/**
 * nvapi alternative for windows x86 binaries
 * nvml api doesn't exists as 32bit dll :///
 */
#ifdef WIN32
#include "nvapi/nvapi_ccminer.h"

static unsigned int nvapi_dev_map[MAX_GPUS] = { 0 };
static NvDisplayHandle hDisplay_a[NVAPI_MAX_PHYSICAL_GPUS * 2] = { 0 };
static NvPhysicalGpuHandle phys[NVAPI_MAX_PHYSICAL_GPUS] = { 0 };
static NvU32 nvapi_dev_cnt = 0;
extern bool nvapi_dll_loaded;

int nvapi_temperature(unsigned int devNum, unsigned int *temperature)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	NV_GPU_THERMAL_SETTINGS thermal;
	thermal.version = NV_GPU_THERMAL_SETTINGS_VER;
	ret = NvAPI_GPU_GetThermalSettings(phys[devNum], 0, &thermal);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI NvAPI_GPU_GetThermalSettings: %s", string);
		return -1;
	}

	(*temperature) = (unsigned int) thermal.sensor[0].currentTemp;

	return 0;
}

int nvapi_fanspeed(unsigned int devNum, unsigned int *speed)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	NvU32 fanspeed = 0;
	ret = NvAPI_GPU_GetTachReading(phys[devNum], &fanspeed);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI NvAPI_GPU_GetTachReading: %s", string);
		return -1;
	}

	(*speed) = (unsigned int) fanspeed;

	return 0;
}

int nvapi_getpstate(unsigned int devNum, unsigned int *pstate)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	NV_GPU_PERF_PSTATE_ID CurrentPstate = NVAPI_GPU_PERF_PSTATE_UNDEFINED; /* 16 */
	ret = NvAPI_GPU_GetCurrentPstate(phys[devNum], &CurrentPstate);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI NvAPI_GPU_GetCurrentPstate: %s", string);
		return -1;
	}
	else {
		// get pstate for the moment... often 0 = P0
		(*pstate) = (unsigned int)CurrentPstate;
	}

	return 0;
}

#define UTIL_DOMAIN_GPU 0
int nvapi_getusage(unsigned int devNum, unsigned int *pct)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	NV_GPU_DYNAMIC_PSTATES_INFO_EX info;
	info.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
	ret = NvAPI_GPU_GetDynamicPstatesInfoEx(phys[devNum], &info);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI GetDynamicPstatesInfoEx: %s", string);
		return -1;
	}
	else {
		if (info.utilization[UTIL_DOMAIN_GPU].bIsPresent)
			(*pct) = info.utilization[UTIL_DOMAIN_GPU].percentage;
	}

	return 0;
}

int nvapi_getinfo(unsigned int devNum, uint16_t &vid, uint16_t &pid)
{
	NvAPI_Status ret;
	NvU32 pDeviceId, pSubSystemId, pRevisionId, pExtDeviceId;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	ret = NvAPI_GPU_GetPCIIdentifiers(phys[devNum], &pDeviceId, &pSubSystemId, &pRevisionId, &pExtDeviceId);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI GetPCIIdentifiers: %s", string);
		return -1;
	}

	pid = pDeviceId >> 16;
	vid = pDeviceId & 0xFFFF;
	if (vid == 0x10DE && pSubSystemId) {
		vid = pSubSystemId & 0xFFFF;
		pid = pSubSystemId >> 16;
		// Colorful and Inno3D
		if (pid == 0) pid = pDeviceId >> 16;
	}

	return 0;
}

int nvapi_getserial(unsigned int devNum, char *serial, unsigned int maxlen)
{
	NvAPI_Status ret;
	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	memset(serial, 0, maxlen);

	if (maxlen < 11)
		return -EINVAL;

	NvAPI_ShortString ser = { 0 };
	ret = NvAPI_DLL_GetSerialNumber(phys[devNum], ser);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI GetSerialNumber: %s", string);
		return -1;
	}

	uint8_t *bytes = (uint8_t*) ser;
	for (int n=0; n<5; n++) sprintf(&serial[n*2], "%02X", bytes[n]);
	return 0;
}

int nvapi_getbios(unsigned int devNum, char *desc, unsigned int maxlen)
{
	NvAPI_Status ret;
	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	if (maxlen < 64) // Short String
		return -1;

	ret = NvAPI_GPU_GetVbiosVersionString(phys[devNum], desc);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI GetVbiosVersionString: %s", string);
		return -1;
	}
	return 0;
}

static int SetAsusRGBLogo(unsigned int devNum, uint32_t RGB, bool ignorePrevState)
{
	NvAPI_Status ret = NVAPI_OK;
	NV_I2C_INFO_EX* i2cInfo;

	int delay1 = 20000;
	int delay2 = 0;

	uchar4 rgb = { 0 };
	memcpy(&rgb, &RGB, 4);
	uchar4 prgb = { 0 };
	int32_t prev = device_led_state[nvapi_devid(devNum)];
	memcpy(&prgb, &prev, 4);

	NV_INIT_STRUCT_ALLOC(NV_I2C_INFO_EX, i2cInfo);
	if (i2cInfo == NULL) return -ENOMEM;

	NvU32 data[5] = { 0 };
	NvU32 datv[2] = { 0, 1 };
	NvU32 datw[2] = { 1, 0 };
	if (rgb.z != prgb.z || ignorePrevState) {
		data[2] = 4; // R:4 G:5 B:6, Mode = 7 (1 static, 2 breath, 3 blink, 4 demo)
		data[3] = 1;
		datv[0] = rgb.z | 0x13384000;

		i2cInfo->i2cDevAddress = 0x52;
		i2cInfo->pbI2cRegAddress = (NvU8*) (&data[2]);
		i2cInfo->regAddrSize = 1;
		i2cInfo->pbData = (NvU8*) datv;
		i2cInfo->cbRead = 5;
		i2cInfo->cbSize = 1;
		i2cInfo->portId = 1;
		i2cInfo->bIsPortIdSet = 1;

		ret = NvAPI_DLL_I2CWriteEx(phys[devNum], i2cInfo, datw);
		usleep(delay1);
		has_rgb_ok = (ret == NVAPI_OK);
	}

	if (rgb.y != prgb.y || ignorePrevState) {
		data[2] = 5;
		data[3] = 1;
		datv[0] = rgb.y | 0x4000;

		i2cInfo->i2cDevAddress = 0x52;
		i2cInfo->pbI2cRegAddress = (NvU8*) (&data[2]);
		i2cInfo->regAddrSize = 1;
		i2cInfo->pbData = (NvU8*) datv;
		i2cInfo->cbRead = 5;
		i2cInfo->cbSize = 1;
		i2cInfo->portId = 1;
		i2cInfo->bIsPortIdSet = 1;

		ret = NvAPI_DLL_I2CWriteEx(phys[devNum], i2cInfo, datw);
		usleep(delay1);
		has_rgb_ok = (ret == NVAPI_OK);
	}

	if (rgb.y != prgb.y || ignorePrevState) {
		data[2] = 6;
		data[3] = 1;
		datv[0] = rgb.x | 0x4000;

		i2cInfo->i2cDevAddress = 0x52;
		i2cInfo->pbI2cRegAddress = (NvU8*) (&data[2]);
		i2cInfo->regAddrSize = 1;
		i2cInfo->pbData = (NvU8*) datv;
		i2cInfo->cbRead = 5;
		i2cInfo->cbSize = 1;
		i2cInfo->portId = 1;
		i2cInfo->bIsPortIdSet = 1;

		ret = NvAPI_DLL_I2CWriteEx(phys[devNum], i2cInfo, datw);
		usleep(delay1);
		has_rgb_ok = (ret == NVAPI_OK);
	}

	if (rgb.w && ignorePrevState) {
		data[2] = 7;
		data[3] = 1;
		datv[0] = rgb.w | 0x4000;

		i2cInfo->i2cDevAddress = 0x52;
		i2cInfo->pbI2cRegAddress = (NvU8*) (&data[2]);
		i2cInfo->regAddrSize = 1;
		i2cInfo->pbData = (NvU8*) datv;
		i2cInfo->cbRead = 5;
		i2cInfo->cbSize = 1;
		i2cInfo->portId = 1;
		i2cInfo->bIsPortIdSet = 1;

		ret = NvAPI_DLL_I2CWriteEx(phys[devNum], i2cInfo, datw);
		usleep(delay1);
		has_rgb_ok = (ret == NVAPI_OK);
	}
	usleep(delay2);
	free(i2cInfo);
	return (int) ret;
}

static int SetGigabyteRGBLogo(unsigned int devNum, uint32_t RGB)
{
	NvAPI_Status ret;
	NV_I2C_INFO_EX* i2cInfo;
	NV_INIT_STRUCT_ALLOC(NV_I2C_INFO_EX, i2cInfo);
	if (i2cInfo == NULL)
		return -ENOMEM;

	NvU32 readBuf[25] = { 0 };
	NvU32 data[5] = { 0 };
	data[0] = 1;
	data[2] = swab32(RGB & 0xfcfcfcU) | 0x40;

	i2cInfo->i2cDevAddress = 0x48 << 1;
	i2cInfo->pbI2cRegAddress = (NvU8*) (&data[2]);
	i2cInfo->regAddrSize = 4; // NVAPI_MAX_SIZEOF_I2C_REG_ADDRESS
	i2cInfo->pbData = (NvU8*) readBuf;
	i2cInfo->cbRead = 2;
	i2cInfo->cbSize = sizeof(readBuf);
	i2cInfo->portId = 1;
	i2cInfo->bIsPortIdSet = 1;

	//ret = NvAPI_DLL_I2CWriteEx(phys[devNum], i2cInfo, data);
	ret = NvAPI_DLL_I2CReadEx(phys[devNum], i2cInfo, data);
	usleep(20000);
	free(i2cInfo);
	return (int) ret;
}

static int SetZotacRGBLogo(unsigned int devNum, uint32_t RGB)
{
	NvAPI_Status ret;
	NV_I2C_INFO* i2cInfo;
	NV_INIT_STRUCT_ALLOC(NV_I2C_INFO, i2cInfo);
	if (i2cInfo == NULL)
		return -ENOMEM;

	NvU32 buf[25] = { 0 };
	NvU32 data[5] = { 0 };

	uint32_t color = 0, level = 0x40;

	uchar4 rgb = { 0 };
	memcpy(&rgb, &RGB, 4);
	level  = rgb.x & 0xF0;
	level |= rgb.y & 0xF0;
	level |= rgb.z & 0xF0;
	//applog(LOG_DEBUG, "R %u G %u B %u", rgb.z, rgb.y, rgb.x);

	// Not really RGB custom, only some basic colors, so convert
	// 0: Red, 1: Yellow, 2: Green, 3: Cyan, 4: Blue, 5: magenta, 6: white
	if ((RGB & 0xFF0000) && (RGB & 0xFF00) && (RGB & 0xFF)) color = 6;
	else if ((RGB & 0xFF0000) && (RGB & 0xFF)) color = 5;
	else if ((RGB & 0xFF00) && (RGB & 0xFF)) color = 3;
	else if ((RGB & 0xFF0000) && (RGB & 0xFF00)) color = 1;
	else if (RGB & 0xFF) color = 4;
	else if (RGB & 0xFF00) color = 2;

	buf[0] = 0xF0; // F0 set colors
	buf[0] |= (color << 8);  // logo
	buf[0] |= (1 << 16); // top
	if (RGB != 0) // level : 0x10 to 0xF0
		buf[0] |= (level << 24);
	else
		buf[0] |= (0x10U << 24);

	// todo: i2c data crc ?

	i2cInfo->displayMask = 1;
	i2cInfo->bIsDDCPort = 1;
	i2cInfo->i2cDevAddress = 0x48 << 1;
	i2cInfo->pbI2cRegAddress = (NvU8*) (&data[2]);
	i2cInfo->regAddrSize = 1;
	i2cInfo->pbData = (NvU8*) buf;
	i2cInfo->cbSize = 4;
	i2cInfo->i2cSpeed = NVAPI_I2C_SPEED_DEPRECATED;
	i2cInfo->i2cSpeedKhz = NVAPI_I2C_SPEED_100KHZ; // 4
	i2cInfo->portId = 1;
	i2cInfo->bIsPortIdSet = 1;

	ret = NvAPI_I2CWrite(phys[devNum], i2cInfo);
	// required to prevent i2c lock
	usleep(20000);

#if 0
	buf[0] = 0xF7; // F7 toggle leds
	if (RGB == 0)
		buf[0] |= (1 << 8);  // 0 logo on, 1 off
	buf[0] |= (1 << 16); // 1 top off
	ret = NvAPI_I2CWrite(phys[devNum], i2cInfo);
	usleep(20000);
#endif
	// other modes:
	// 0xF1 breathing green (0x070202F1)
	// 0xF2 strobe green    (0x070202F2)
	// 0xF3 cycle           (0x000000F3)

	free(i2cInfo);
	return (int) ret;
}

int nvapi_set_led(unsigned int devNum, int RGB, char *device_name)
{
	uint16_t vid = 0, pid = 0;
	NvAPI_Status ret;
	if (strstr(device_name, "Gigabyte GTX 10")) {
		if (opt_debug)
			applog(LOG_DEBUG, "GPU %x: Set RGB led to %06x", (int) phys[devNum], RGB);
		return SetGigabyteRGBLogo(devNum, (uint32_t) RGB);
	} else if (strstr(device_name, "ASUS GTX 10")) {
		if (opt_debug)
			applog(LOG_DEBUG, "GPU %x: Set RGB led to %06x", (int) phys[devNum], RGB);
		return SetAsusRGBLogo(devNum, (uint32_t) RGB, !has_rgb_ok);
	} else if (strstr(device_name, "Zotac GTX 10")) {
		if (opt_debug)
			applog(LOG_DEBUG, "GPU %x: Set RGB led to %06x", (int) phys[devNum], RGB);
		return SetZotacRGBLogo(devNum, (uint32_t) RGB);
	} else {
		NV_GPU_QUERY_ILLUMINATION_SUPPORT_PARM* illu;
		NV_INIT_STRUCT_ALLOC(NV_GPU_QUERY_ILLUMINATION_SUPPORT_PARM, illu);
		illu->hPhysicalGpu = phys[devNum];
		illu->Attribute = NV_GPU_IA_LOGO_BRIGHTNESS;
		ret = NvAPI_GPU_QueryIlluminationSupport(illu);
		if (!ret && illu->bSupported) {
			NV_GPU_GET_ILLUMINATION_PARM *led;
			NV_INIT_STRUCT_ALLOC(NV_GPU_GET_ILLUMINATION_PARM, led);
			led->hPhysicalGpu = phys[devNum];
			led->Attribute = NV_GPU_IA_LOGO_BRIGHTNESS;
			NvAPI_GPU_GetIllumination(led);
			if (opt_debug)
				applog(LOG_DEBUG, "GPU %x: Led level was %d, set to %d", (int) phys[devNum], led->Value, RGB);
			led->Value = (uint32_t) RGB;
			ret = NvAPI_GPU_SetIllumination((NV_GPU_SET_ILLUMINATION_PARM*) led);
			free(led);
		}
		free(illu);
		return ret;
	}
}

int nvapi_pstateinfo(unsigned int devNum)
{
	uint32_t n;
	NvAPI_Status ret;
	uint32_t* mem = (uint32_t*) calloc(1, 0x4000);
	if (!mem)
		return -ENOMEM;

	unsigned int current = 0xFF;
	// useless on init but...
	nvapi_getpstate(devNum, &current);

#if 0
	// try :p
	uint32_t* buf = (uint32_t*) calloc(1, 0x8000);
	for (int i=8; i < 0x8000 && buf; i+=4) {
		buf[0] = 0x10000 + i;
		NV_GPU_PERF_PSTATE_ID pst = NVAPI_GPU_PERF_PSTATE_P0;
		ret = NvAPI_DLL_GetPstateClientLimits(phys[devNum], pst, buf);
		if (ret != NVAPI_INCOMPATIBLE_STRUCT_VERSION) {
			NvAPI_ShortString string;
			NvAPI_GetErrorMessage(ret, string);
			applog(LOG_BLUE, "struct size is %06x : %s", buf[0], string);
			for (int n=0; n < i/32; n++)
				applog_hex(&buf[n*(32/4)], 32);
			break;
		}
	}
	free(buf);
#endif

#if 0
	// Unsure of the meaning of these values
	NVAPI_GPU_POWER_TOPO topo = { 0 };
	topo.version = NVAPI_GPU_POWER_TOPO_VER;
	if ((ret = NvAPI_DLL_ClientPowerTopologyGetStatus(phys[devNum], &topo)) == NVAPI_OK) {
		if (topo.count)
			applog(LOG_RAW, " GPU TDP is %.1f~%.1f W ?",
			(double) topo.entries[0].power/1000, (double) topo.entries[1].power/1000);

	// Ok on 970, not pascal
	NV_GPU_PERF_PSTATES20_INFO_V2 pset2 = { 0 };
	pset2.version = NV_GPU_PERF_PSTATES20_INFO_VER2;
	pset2.ov.numVoltages = 1;
	pset2.ov.voltages[0].voltDelta_uV.value = 3000;  // gpu + 3000 uv;
	ret = NvAPI_DLL_SetPstates20v2(phys[devNum], &pset2);
#endif

	NV_GPU_PERF_PSTATES20_INFO* info;
	NV_INIT_STRUCT_ON(NV_GPU_PERF_PSTATES20_INFO, info, mem);
	if ((ret = NvAPI_GPU_GetPstates20(phys[devNum], info)) != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_RAW, "NVAPI GetPstates20: %s", string);
		return -1;
	}

	for (n=0; n < info->numPstates; n++) {
		NV_GPU_PSTATE20_CLOCK_ENTRY_V1* clocks = info->pstates[n].clocks;
		applog(LOG_RAW, "%sP%d: MEM %4u MHz%s GPU %6.1f MHz%s %4u mV%s \x7F %d/%d",
			info->pstates[n].pstateId == current ? ">":" ", (int) info->pstates[n].pstateId,
			clocks[1].data.single.freq_kHz/1000, clocks[1].bIsEditable ? "*":" ",
			(double) clocks[0].data.single.freq_kHz/1000, clocks[0].bIsEditable ? "*":" ",
			info->pstates[n].baseVoltages[0].volt_uV/1000, info->pstates[n].baseVoltages[0].bIsEditable ? "*": " ",
			info->pstates[n].baseVoltages[0].voltDelta_uV.valueRange.min/1000, // range if editable
			info->pstates[n].baseVoltages[0].voltDelta_uV.valueRange.max/1000);
		if (clocks[1].freqDelta_kHz.value || clocks[0].freqDelta_kHz.value) {
			applog(LOG_RAW, "      OC %+4d MHz      %+6.1f MHz",
				clocks[1].freqDelta_kHz.value/1000, (double) clocks[0].freqDelta_kHz.value/1000);
		}
	}
	// boost over volting (GTX 9xx only ?)
	for (n=0; n < info->ov.numVoltages; n++) {
		applog(LOG_RAW, " OV: %u%+d mV%s \x7F %d/%d",
			info->ov.voltages[n].volt_uV/1000, info->ov.voltages[n].voltDelta_uV.value/1000, info->ov.voltages[n].bIsEditable ? "*":" ",
			info->ov.voltages[n].voltDelta_uV.valueRange.min/1000, info->ov.voltages[n].voltDelta_uV.valueRange.max/1000);
	}

	NV_GPU_CLOCK_FREQUENCIES *freqs;
	NV_INIT_STRUCT_ON(NV_GPU_CLOCK_FREQUENCIES, freqs, mem);
	freqs->ClockType = NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK;
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], freqs);
	applog(LOG_RAW, "     MEM %4.0f MHz  GPU %6.1f MHz     Base Clocks",
		(double) freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency / 1000,
		(double) freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency / 1000);

	freqs->ClockType = NV_GPU_CLOCK_FREQUENCIES_BOOST_CLOCK;
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], freqs);
	applog(LOG_RAW, "     MEM %4.0f MHz  GPU %6.1f MHz     Boost Clocks",
		(double) freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency / 1000,
		(double) freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency / 1000);

	freqs->ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], freqs);
	applog(LOG_RAW, "     MEM %4.0f MHz  GPU %6.1f MHz    >Current",
		(double) freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency / 1000,
		(double) freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency / 1000);

	// Other clock values ??
	NVAPI_GPU_PERF_CLOCKS *pcl;
	NV_INIT_STRUCT_ALLOC(NVAPI_GPU_PERF_CLOCKS, pcl);
	int numClock=0; ret = NVAPI_OK;
	while (ret == NVAPI_OK) {
		if ((ret = NvAPI_DLL_GetPerfClocks(phys[devNum], numClock, pcl)) == NVAPI_OK) {
			applog(LOG_RAW, " C%d: MEM %4.0f MHz  GPU %6.1f MHz [%5.1f/%6.1f]", numClock,
				(double) pcl->memFreq1/1000, (double) pcl->gpuFreq1/1000, (double) pcl->gpuFreqMin/1000, (double) pcl->gpuFreqMax/1000);
		//	ret = NvAPI_DLL_SetPerfClocks(phys[devNum], numClock, pcl); // error
		}
		numClock++;
	}

	// Pascal only
	NVAPI_VOLTBOOST_PERCENT *pvb;
	NV_INIT_STRUCT_ON(NVAPI_VOLTBOOST_PERCENT, pvb, mem);
	if ((ret = NvAPI_DLL_GetCoreVoltageBoostPercent(phys[devNum], pvb)) == NVAPI_OK) {
		NVAPI_VOLTAGE_STATUS *pvdom;
		NV_INIT_STRUCT_ALLOC(NVAPI_VOLTAGE_STATUS, pvdom);
		NvAPI_DLL_GetCurrentVoltage(phys[devNum], pvdom);
		if (pvdom && pvdom->value_uV)
			applog(LOG_RAW, " GPU Voltage is %u mV %+d%% boost", pvdom->value_uV/1000, pvb->percent);
		else if (pvdom)
			applog(LOG_RAW, " GPU Voltage is %u mV", pvdom->value_uV/1000);
		free(pvdom);
	} else {
		// Maxwell 9xx
		NVAPI_VOLT_STATUS *mvdom, *mvstep;
		NV_INIT_STRUCT_ALLOC(NVAPI_VOLT_STATUS, mvdom);
		if (mvdom && (ret = NvAPI_DLL_GetVoltageDomainsStatus(phys[devNum], mvdom)) == NVAPI_OK) {
			NV_INIT_STRUCT_ALLOC(NVAPI_VOLT_STATUS, mvstep);
			NvAPI_DLL_GetVoltageStep(phys[devNum], mvstep);
			if (mvdom->value_uV) applog(LOG_RAW, " GPU Voltage is %.1f mV with %.3f mV resolution",
				(double) mvdom->value_uV/1000, (double) mvstep->value_uV/1000);
			free(mvstep);
		}
		free(mvdom);
	}

	uint32_t plim = nvapi_get_plimit(devNum);
	double min_pw = 0, max_pw = 0; // percent

	NVAPI_GPU_POWER_INFO nfo = { 0 };
	nfo.version = NVAPI_GPU_POWER_INFO_VER;
	ret = NvAPI_DLL_ClientPowerPoliciesGetInfo(phys[devNum], &nfo);
	if (ret == NVAPI_OK && nfo.valid) {
		min_pw = (double)nfo.entries[0].min_power / 1000;
		max_pw = (double)nfo.entries[0].max_power / 1000;
	}
	applog(LOG_RAW, " Power limit is set to %u%%, range [%.0f-%.0f%%]", plim, min_pw, max_pw);

#if 0
	NVAPI_COOLER_SETTINGS *cooler;
	NV_INIT_STRUCT_ON(NVAPI_COOLER_SETTINGS, cooler, mem);
	ret = NvAPI_DLL_GetCoolerSettings(phys[devNum], 7, cooler);
	if (ret == NVAPI_OK) {
		applog(LOG_RAW, " Fan level is set to %u%%", cooler->level); // wrong val, seems 1 (auto ?)
		NVAPI_COOLER_LEVEL *fan;
		NV_INIT_STRUCT_ALLOC(NVAPI_COOLER_LEVEL, fan);
		fan->level = 100;
		fan->count = 1;
		ret = NvAPI_DLL_SetCoolerLevels(phys[devNum], 7, fan);
		free(fan);
		sleep(10);
		ret = NvAPI_DLL_RestoreCoolerSettings(phys[devNum], cooler, 7);
	}
#endif

	NV_GPU_THERMAL_SETTINGS *tset;
	NV_INIT_STRUCT_ON(NV_GPU_THERMAL_SETTINGS, tset, mem);

	NVAPI_GPU_THERMAL_INFO *tnfo;
	NV_INIT_STRUCT_ALLOC(NVAPI_GPU_THERMAL_INFO, tnfo);
	NVAPI_GPU_THERMAL_LIMIT *tlim;
	NV_INIT_STRUCT_ALLOC(NVAPI_GPU_THERMAL_LIMIT, tlim);
	NvAPI_GPU_GetThermalSettings(phys[devNum], 0, tset);
	NvAPI_DLL_ClientThermalPoliciesGetInfo(phys[devNum], tnfo);
	if ((ret = NvAPI_DLL_ClientThermalPoliciesGetLimit(phys[devNum], tlim)) == NVAPI_OK) {
		applog(LOG_RAW, " Thermal limit is set to %u, current Tc %d, range [%u-%u]",
			tlim->entries[0].value >> 8, tset->sensor[0].currentTemp,
			tnfo->entries[0].min_temp >> 8, tnfo->entries[0].max_temp >> 8);
	}
	free(tnfo);
	free(tlim);

#if 1
	// Read pascal Clocks Table, Empty on 9xx
	//NVAPI_CLOCKS_RANGE* ranges;
	//NV_INIT_STRUCT_ON(NVAPI_CLOCKS_RANGE, ranges, mem);
	//ret = NvAPI_DLL_GetClockBoostRanges(phys[devNum], ranges);

	NVAPI_CLOCK_MASKS* boost;
	NV_INIT_STRUCT_ON(NVAPI_CLOCK_MASKS, boost, mem);
	ret = NvAPI_DLL_GetClockBoostMask(phys[devNum], boost);
	int gpuClocks = 0, memClocks = 0;
	for (n=0; n < 80+23; n++) {
		if (boost->clocks[n].memDelta) memClocks++;
		if (boost->clocks[n].gpuDelta) gpuClocks++;
	}

	// PASCAL GTX ONLY
	if (gpuClocks || memClocks) {
		NVAPI_CLOCK_TABLE *table;
		NV_INIT_STRUCT_ALLOC(NVAPI_CLOCK_TABLE, table);
		memcpy(table->mask, boost->mask, 12);
		ret = NvAPI_DLL_GetClockBoostTable(phys[devNum], table);
		gpuClocks = 0, memClocks = 0;
		for (n=0; n < 12; n++) {
			if (table->buf0[n] != 0) applog(LOG_RAW, "boost table 0[%u] not empty (%u)", n, table->buf0[n]);
		}
		for (n=0; n < 80; n++) {
			if (table->gpuDeltas[n].freqDelta) {
				// note: gpu delta value seems to be x2, not the memory
				//applog(LOG_RAW, " Boost gpu clock delta %u set to %d MHz", n, table->gpuDeltas[n].freqDelta/2000);
				gpuClocks++;
			}
		}
		for (n=0; n < 23; n++) {
			if (table->memFilled[n]) {
				//applog(LOG_RAW, " Boost mem clock delta %u set to %d MHz", n, table->memDeltas[n]/1000);
				memClocks++;
			}
		}
		for (n=0; n < 1529; n++) {
			if (table->buf1[n] != 0) applog(LOG_RAW, "boost table 1[%u] not empty (%u)", n, table->buf1[n]);
		}
		applog(LOG_RAW, " Boost table contains %d gpu and %d mem levels.", gpuClocks, memClocks);
		free(table);

		NVAPI_VFP_CURVE *curve;
		NV_INIT_STRUCT_ALLOC(NVAPI_VFP_CURVE, curve);
		memcpy(curve->mask, boost->mask, 12);
		ret = NvAPI_DLL_GetVFPCurve(phys[devNum], curve);
		gpuClocks = 0, memClocks = 0;
		for (n=0; n < 80; n++) {
			if (curve->gpuEntries[n].freq_kHz || curve->gpuEntries[n].volt_uV) {
			//	applog(LOG_RAW, "gpu volt table %2u %4u MHz - %6u mV", n, curve->gpuEntries[n].freq_kHz/1000, curve->gpuEntries[n].volt_uV/1000);
				gpuClocks++;
			}
		}
		for (n=0; n < 23; n++) {
			if (curve->memEntries[n].freq_kHz || curve->memEntries[n].volt_uV) {
			//	applog(LOG_RAW, "mem volt table %2u %4u MHz - %6u mV", n, curve->memEntries[n].freq_kHz/1000, curve->memEntries[n].volt_uV/1000);
				memClocks++;
			}
		}
		for (n=0; n < 1064; n++) {
			if (curve->buf1[n] != 0) applog(LOG_RAW, "volt table buf1[%u] not empty (%u)", n, curve->buf1[n]);
		}
		applog(LOG_RAW, " Volts table contains %d gpu and %d mem levels.", gpuClocks, memClocks);
		free(curve);
	}

	// Maxwell
	else {
		NVAPI_VOLTAGES_TABLE* volts;
		NV_INIT_STRUCT_ALLOC(NVAPI_VOLTAGES_TABLE, volts);
		int entries = 0;
		ret = NvAPI_DLL_GetVoltages(phys[devNum], volts);
		for (n=0; n < 128; n++) {
			if (volts->entries[n].volt_uV)
				entries++;
		}
		applog(LOG_RAW, " Volts table contains %d gpu levels.", entries);
		free(volts);
	}

	NV_DISPLAY_DRIVER_MEMORY_INFO* meminfo;
	NV_INIT_STRUCT_ON(NV_DISPLAY_DRIVER_MEMORY_INFO, meminfo, mem);
	meminfo->version = NV_DISPLAY_DRIVER_MEMORY_INFO_VER;
	if ((ret = NvAPI_GPU_GetMemoryInfo(phys[devNum], meminfo)) == NVAPI_OK) {
		applog(LOG_RAW, " Memory: %u MB, %.1f used", meminfo->dedicatedVideoMemory/1024,
			(double) (meminfo->availableDedicatedVideoMemory - meminfo->curAvailableDedicatedVideoMemory)/1024);
	}
#if 0 /* some undetermined stats */
	NVAPI_GPU_PERF_INFO pi = { 0 };
	pi.version = NVAPI_GPU_PERF_INFO_VER;
	ret = NvAPI_DLL_PerfPoliciesGetInfo(phys[devNum], &pi);

	NVAPI_GPU_PERF_STATUS ps = { 0 };
	ps.version = NVAPI_GPU_PERF_STATUS_VER;
	ret = NvAPI_DLL_PerfPoliciesGetStatus(phys[devNum], &ps);
	applog(LOG_BLUE, "%llx %lld. %lld. %llx %llx %llx", ps.timeRef, ps.val1, ps.val2, ps.values[0], ps.values[1], ps.values[2]);
#endif

#endif
	free(mem);
	return 0;
}

// workaround for buggy driver 378.49
unsigned int nvapi_get_gpu_clock(unsigned int devNum)
{
	NvAPI_Status ret = NVAPI_OK;
	unsigned int freq = 0;
	NV_GPU_CLOCK_FREQUENCIES *freqs;
	NV_INIT_STRUCT_ALLOC(NV_GPU_CLOCK_FREQUENCIES, freqs);
	freqs->ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ;
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], freqs);
	if (ret == NVAPI_OK) {
		freq = freqs->domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency / 1000;
	}
	free(freqs);
	return freq; // in MHz
}

uint8_t nvapi_get_plimit(unsigned int devNum)
{
	NvAPI_Status ret = NVAPI_OK;
	NVAPI_GPU_POWER_STATUS pol = { 0 };
	pol.version = NVAPI_GPU_POWER_STATUS_VER;
	if ((ret = NvAPI_DLL_ClientPowerPoliciesGetStatus(phys[devNum], &pol)) != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI PowerPoliciesGetStatus: %s", string);
		return 0;
	}
	return (uint8_t) (pol.entries[0].power / 1000); // in percent
}

int nvapi_set_plimit(unsigned int devNum, uint16_t percent)
{
	NvAPI_Status ret = NVAPI_OK;
	uint32_t val = percent * 1000;

	NVAPI_GPU_POWER_INFO nfo = { 0 };
	nfo.version = NVAPI_GPU_POWER_INFO_VER;
	ret = NvAPI_DLL_ClientPowerPoliciesGetInfo(phys[devNum], &nfo);
	if (ret == NVAPI_OK) {
		if (val == 0)
			val = nfo.entries[0].def_power;
		else if (val < nfo.entries[0].min_power)
			val = nfo.entries[0].min_power;
		else if (val > nfo.entries[0].max_power)
			val = nfo.entries[0].max_power;
	}

	NVAPI_GPU_POWER_STATUS pol = { 0 };
	pol.version = NVAPI_GPU_POWER_STATUS_VER;
	pol.flags = 1;
	pol.entries[0].power = val;
	if ((ret = NvAPI_DLL_ClientPowerPoliciesSetStatus(phys[devNum], &pol)) != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI PowerPoliciesSetStatus: %s", string);
		return -1;
	}
	return ret;
}

int nvapi_set_tlimit(unsigned int devNum, uint8_t limit)
{
	NvAPI_Status ret;
	uint32_t val = limit;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	NV_GPU_THERMAL_SETTINGS tset = { 0 };
	NVAPI_GPU_THERMAL_INFO tnfo = { 0 };
	NVAPI_GPU_THERMAL_LIMIT tlim = { 0 };
	tset.version = NV_GPU_THERMAL_SETTINGS_VER;
	NvAPI_GPU_GetThermalSettings(phys[devNum], 0, &tset);
	tnfo.version = NVAPI_GPU_THERMAL_INFO_VER;
	NvAPI_DLL_ClientThermalPoliciesGetInfo(phys[devNum], &tnfo);
	tlim.version = NVAPI_GPU_THERMAL_LIMIT_VER;
	if ((ret = NvAPI_DLL_ClientThermalPoliciesGetLimit(phys[devNum], &tlim)) == NVAPI_OK) {
		tlim.entries[0].value = val << 8;
		tlim.flags = 1;
		ret = NvAPI_DLL_ClientThermalPoliciesSetLimit(phys[devNum], &tlim);
		if (ret == NVAPI_OK) {
			applog(LOG_INFO, "GPU #%u: thermal limit set to %u, current Tc %d, range [%u-%u]",
				devNum, val, tset.sensor[0].currentTemp,
				tnfo.entries[0].min_temp >> 8, tnfo.entries[0].max_temp >> 8);
		} else {
			NvAPI_ShortString string;
			NvAPI_GetErrorMessage(ret, string);
			applog(LOG_WARNING, "GPU #%u: thermal limit: %s, valid range is [%u-%u]", devNum, string,
				tnfo.entries[0].min_temp >> 8, tnfo.entries[0].max_temp >> 8);
		}
	}
	return (int) ret;
}

int nvapi_set_gpuclock(unsigned int devNum, uint32_t clock)
{
	NvAPI_Status ret;
	NvS32 delta = 0;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;
#if 0
	// wrong api to get default base clock when modified, cuda props seems fine
	NV_GPU_CLOCK_FREQUENCIES freqs = { 0 };
	freqs.version = NV_GPU_CLOCK_FREQUENCIES_VER;
	freqs.ClockType = NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK;
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], &freqs);
	if (ret == NVAPI_OK)  {
		delta = (clock * 1000) - freqs.domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency;
	}

	NV_GPU_PERF_PSTATES_INFO deffreqs = { 0 };
	deffreqs.version = NV_GPU_PERF_PSTATES_INFO_VER;
	ret = NvAPI_GPU_GetPstatesInfoEx(phys[devNum], &deffreqs, 0); // we want default clock grr!
	if (ret == NVAPI_OK) {
		if (deffreqs.pstates[0].clocks[1].domainId == NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS)
			delta = (clock * 1000) - deffreqs.pstates[0].clocks[1].freq*2;
	}
#endif

	cudaDeviceProp props = { 0 };
	NvU32 busId = 0xFFFF;
	ret = NvAPI_GPU_GetBusId(phys[devNum], &busId);
	for (int d=0; d < (int) nvapi_dev_cnt; d++) {
		 // unsure about devNum, so be safe
		cudaGetDeviceProperties(&props, d);
		if (props.pciBusID == busId) {
			delta = (clock * 1000) - props.clockRate;
			break;
		}
	}

	if (delta == (clock * 1000))
		return ret;

	NV_GPU_PERF_PSTATES20_INFO_V1 pset1 = { 0 };
	pset1.version = NV_GPU_PERF_PSTATES20_INFO_VER1;
	pset1.numPstates = 1;
	pset1.numClocks = 1;
	// Ok on both 1080 and 970
	pset1.pstates[0].clocks[0].domainId = NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS;
	pset1.pstates[0].clocks[0].freqDelta_kHz.value = delta;
	ret = NvAPI_DLL_SetPstates20v1(phys[devNum], &pset1);
	if (ret == NVAPI_OK) {
		applog(LOG_INFO, "GPU #%u: boost gpu clock set to %u (delta %d)", devNum, clock, delta/1000);
	}
	return ret;
}

int nvapi_set_memclock(unsigned int devNum, uint32_t clock)
{
	NvAPI_Status ret;
	NvS32 delta = 0;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	// wrong to get default base clock (when modified) on maxwell (same as cuda props one)
	NV_GPU_CLOCK_FREQUENCIES freqs = { 0 };
	freqs.version = NV_GPU_CLOCK_FREQUENCIES_VER;
	freqs.ClockType = NV_GPU_CLOCK_FREQUENCIES_BASE_CLOCK;
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], &freqs); // wrong base clocks, useless
	if (ret == NVAPI_OK)  {
		delta = (clock * 1000) - freqs.domain[NVAPI_GPU_PUBLIC_CLOCK_MEMORY].frequency;
	}

	// seems ok on maxwell and pascal for the mem clocks
	NV_GPU_PERF_PSTATES_INFO deffreqs = { 0 };
	deffreqs.version = NV_GPU_PERF_PSTATES_INFO_VER;
	ret = NvAPI_GPU_GetPstatesInfoEx(phys[devNum], &deffreqs, 0x1); // deprecated but req for def clocks
	if (ret == NVAPI_OK) {
		if (deffreqs.pstates[0].clocks[0].domainId == NVAPI_GPU_PUBLIC_CLOCK_MEMORY)
			delta = (clock * 1000) - deffreqs.pstates[0].clocks[0].freq;
	}

	if (delta == (clock * 1000))
		return ret;

	// todo: bounds check with GetPstates20

	NV_GPU_PERF_PSTATES20_INFO_V1 pset1 = { 0 };
	pset1.version = NV_GPU_PERF_PSTATES20_INFO_VER1;
	pset1.numPstates = 1;
	pset1.numClocks = 1;
	pset1.pstates[0].clocks[0].domainId = NVAPI_GPU_PUBLIC_CLOCK_MEMORY;
	pset1.pstates[0].clocks[0].freqDelta_kHz.value = delta;
	ret = NvAPI_DLL_SetPstates20v1(phys[devNum], &pset1);
	if (ret == NVAPI_OK) {
		applog(LOG_INFO, "GPU #%u: Boost mem clock set to %u (delta %d)", devNum, clock, delta/1000);
	}
	return ret;
}

static int nvapi_set_memoffset(unsigned int devNum, int32_t delta, bool log=true)
{
	NvAPI_Status ret;
	NvS32 deltaKHz = delta * 1000;

	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	// todo: bounds check with GetPstates20

	NV_GPU_PERF_PSTATES20_INFO_V1 pset1 = { 0 };
	pset1.version = NV_GPU_PERF_PSTATES20_INFO_VER1;
	pset1.numPstates = 1;
	pset1.numClocks = 1;
	pset1.pstates[0].clocks[0].domainId = NVAPI_GPU_PUBLIC_CLOCK_MEMORY;
	pset1.pstates[0].clocks[0].freqDelta_kHz.value = deltaKHz;
	ret = NvAPI_DLL_SetPstates20v1(phys[devNum], &pset1);
	if (ret == NVAPI_OK) {
		if (log) applog(LOG_INFO, "GPU #%u: Memory clock offset set to %+d MHz", devNum, deltaKHz / 1000);
		need_memclockrst = true;
	}
	return ret;
}

// Replacement for WIN32 CUDA 6.5 on pascal
int nvapiMemGetInfo(int dev_id, uint64_t *free, uint64_t *total)
{
	NvAPI_Status ret = NVAPI_OK;
	NV_DISPLAY_DRIVER_MEMORY_INFO mem = { 0 };
	mem.version = NV_DISPLAY_DRIVER_MEMORY_INFO_VER;
	unsigned int devNum = nvapi_dev_map[dev_id % MAX_GPUS];
	if ((ret = NvAPI_GPU_GetMemoryInfo(phys[devNum], &mem)) == NVAPI_OK) {
		*total = (uint64_t) mem.dedicatedVideoMemory;// mem.availableDedicatedVideoMemory;
		*free  = (uint64_t) mem.curAvailableDedicatedVideoMemory;
	}
	return (int) ret;
}

int nvapi_init()
{
	int num_gpus = cuda_num_devices();
	NvAPI_Status ret = NvAPI_Initialize();
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI NvAPI_Initialize: %s", string);
		return -1;
	}

	ret = NvAPI_EnumPhysicalGPUs(phys, &nvapi_dev_cnt);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI NvAPI_EnumPhysicalGPUs: %s", string);
		return -1;
	}

	for (int g = 0; g < num_gpus; g++) {
		cudaDeviceProp props;
		if (cudaGetDeviceProperties(&props, g) == cudaSuccess) {
			device_bus_ids[g] = props.pciBusID;
		}
		nvapi_dev_map[g] = g; // default mapping
	}

	for (NvU8 i = 0; i < nvapi_dev_cnt; i++) {
		NvAPI_ShortString name;
		ret = NvAPI_GPU_GetFullName(phys[i], name);
		if (ret == NVAPI_OK) {
			for (int g = 0; g < num_gpus; g++) {
				NvU32 busId;
				ret = NvAPI_GPU_GetBusId(phys[i], &busId);
				if (ret == NVAPI_OK && busId == device_bus_ids[g]) {
					nvapi_dev_map[g] = i;
					if (opt_debug)
						applog(LOG_DEBUG, "CUDA GPU %d matches NVAPI GPU %d by busId %u",
							g, i, busId);
					break;
				}
			}
		} else {
			NvAPI_ShortString string;
			NvAPI_GetErrorMessage(ret, string);
			applog(LOG_DEBUG, "NVAPI NvAPI_GPU_GetFullName: %s", string);
		}
	}
#if 0
	if (opt_debug) {
		NvAPI_ShortString ver;
		NvAPI_GetInterfaceVersionString(ver);
		applog(LOG_DEBUG, "%s", ver);
	}
#endif

	NvU32 udv;
	NvAPI_ShortString str;
	ret = NvAPI_SYS_GetDriverAndBranchVersion(&udv, str);
	if (ret == NVAPI_OK) {
		sprintf(driver_version,"%d.%02d", udv / 100, udv % 100);
	}

	return 0;
}

int nvapi_init_settings()
{
	// nvapi.dll
	int ret = nvapi_dll_init();
	if (ret != NVAPI_OK)
		return ret;

	if (!opt_n_threads) {
		opt_n_threads = active_gpus;
	}

	for (int n=0; n < opt_n_threads; n++) {
		int dev_id = device_map[n % MAX_GPUS];
		if (device_plimit[dev_id] && !nvml_plimit_set) {
			if (nvapi_set_plimit(nvapi_dev_map[dev_id], device_plimit[dev_id]) == NVAPI_OK) {
				uint32_t res = nvapi_get_plimit(nvapi_dev_map[dev_id]);
				gpulog(LOG_INFO, n, "Power limit is set to %u%%", res);
			}
		}
		if (device_tlimit[dev_id]) {
			nvapi_set_tlimit(nvapi_dev_map[dev_id], device_tlimit[dev_id]);
		}
		if (device_gpu_clocks[dev_id]) {
			ret = nvapi_set_gpuclock(nvapi_dev_map[dev_id], device_gpu_clocks[dev_id]);
			if (ret) {
				NvAPI_ShortString string;
				NvAPI_GetErrorMessage((NvAPI_Status) ret, string);
				gpulog(LOG_WARNING, n, "nvapi_set_gpuclock %s", string);
			}
		}
		if (device_mem_offsets[dev_id]) {
			ret = nvapi_set_memoffset(nvapi_dev_map[dev_id], device_mem_offsets[dev_id]);
			if (ret) {
				NvAPI_ShortString string;
				NvAPI_GetErrorMessage((NvAPI_Status)ret, string);
				gpulog(LOG_WARNING, n, "nvapi_set_memoffset %s", string);
			}
		}
		else if (device_mem_clocks[dev_id]) {
			ret = nvapi_set_memclock(nvapi_dev_map[dev_id], device_mem_clocks[dev_id]);
			if (ret) {
				NvAPI_ShortString string;
				NvAPI_GetErrorMessage((NvAPI_Status) ret, string);
				gpulog(LOG_WARNING, n, "nvapi_set_memclock %s", string);
			}
		}
		if (device_pstate[dev_id]) {
			// dunno how via nvapi or/and pascal
		}
		if (device_led[dev_id] != -1) {
			int err = nvapi_set_led(nvapi_dev_map[dev_id], device_led[dev_id], device_name[dev_id]);
			if (err != 0) {
				gpulog(LOG_WARNING, n, "Unable to set led value (err %d)", err);
			}
			device_led_state[dev_id] = device_led[dev_id];
		}
	}

	return ret;
}

void nvapi_toggle_clocks(int thr_id, bool enable)
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	if (device_mem_offsets[dev_id]) {
		nvapi_set_memoffset(nvapi_dev_map[dev_id], enable ? device_mem_offsets[dev_id] : 0, false);
	}
}

unsigned int nvapi_devnum(int dev_id)
{
	return nvapi_dev_map[dev_id];
}

int nvapi_devid(unsigned int devNum)
{
	for (int i=0; i < opt_n_threads; i++) {
		int dev_id = device_map[i % MAX_GPUS];
		if (nvapi_dev_map[dev_id] = devNum)
			return dev_id;
	}
	return 0;
}

#endif /* WIN32 : Windows specific (nvapi) */

/* api functions -------------------------------------- */

// assume 2500 rpm as default, auto-updated if more
static unsigned int fan_speed_max = 2500;

unsigned int gpu_fanpercent(struct cgpu_info *gpu)
{
	unsigned int pct = 0;
	if (hnvml) {
		nvml_get_fanpcnt(hnvml, gpu->gpu_id, &pct);
	}
#ifdef WIN32
	else {
		unsigned int rpm = 0;
		nvapi_fanspeed(nvapi_dev_map[gpu->gpu_id], &rpm);
		pct = (rpm * 100) / fan_speed_max;
		if (pct > 100) {
			pct = 100;
			fan_speed_max = rpm;
		}
	}
#endif
	return pct;
}

unsigned int gpu_fanrpm(struct cgpu_info *gpu)
{
	unsigned int rpm = 0;
#ifdef WIN32
	nvapi_fanspeed(nvapi_dev_map[gpu->gpu_id], &rpm);
#endif
	return rpm;
}


float gpu_temp(struct cgpu_info *gpu)
{
	float tc = 0.0;
	unsigned int tmp = 0;
	if (hnvml) {
		nvml_get_tempC(hnvml, gpu->gpu_id, &tmp);
		tc = (float)tmp;
	}
#ifdef WIN32
	else {
		nvapi_temperature(nvapi_dev_map[gpu->gpu_id], &tmp);
		tc = (float)tmp;
	}
#endif
	return tc;
}

int gpu_pstate(struct cgpu_info *gpu)
{
	int pstate = -1;
	int support = -1;
	if (hnvml) {
		support = nvml_get_pstate(hnvml, gpu->gpu_id, &pstate);
	}
#ifdef WIN32
	if (support == -1) {
		unsigned int pst = 0;
		nvapi_getpstate(nvapi_dev_map[gpu->gpu_id], &pst);
		pstate = (int) pst;
	}
#endif
	return pstate;
}

int gpu_busid(struct cgpu_info *gpu)
{
	int busid = -1;
	int support = -1;
	if (hnvml) {
		support = nvml_get_busid(hnvml, gpu->gpu_id, &busid);
	}
#ifdef WIN32
	if (support == -1) {
		busid = device_bus_ids[gpu->gpu_id];
	}
#endif
	return busid;
}

unsigned int gpu_power(struct cgpu_info *gpu)
{
	unsigned int mw = 0;
	int support = -1;
	if (hnvml) {
		support = nvml_get_power_usage(hnvml, gpu->gpu_id, &mw);
	}
#ifdef WIN32
	if (support == -1) {
		unsigned int pct = 0;
		nvapi_getusage(nvapi_dev_map[gpu->gpu_id], &pct);
		pct *= nvapi_get_plimit(nvapi_dev_map[gpu->gpu_id]);
		pct /= 100;
		mw = pct; // to fix
	}
#endif
	if (gpu->gpu_power > 0) {
		// average
		mw = (gpu->gpu_power + mw) / 2;
	}
	return mw;
}

unsigned int gpu_plimit(struct cgpu_info *gpu)
{
	unsigned int mw = 0;
	int support = -1;
	if (hnvml) {
		mw = nvml_get_plimit(hnvml, gpu->gpu_id);
		support = (mw > 0);
	}
#ifdef WIN32
	// NVAPI value is in % (< 100 so)
	if (support == -1) {
		mw = nvapi_get_plimit(nvapi_dev_map[gpu->gpu_id]);
	}
#endif
	return mw;
}

static int translate_vendor_id(uint16_t vid, char *vendorname)
{
	struct VENDORS {
		const uint16_t vid;
		const char *name;
	} vendors[] = {
		{ 0x1043, "ASUS" },
		{ 0x1048, "Elsa" },
		{ 0x107D, "Leadtek" },
		{ 0x10B0, "Gainward" },
		// { 0x10DE, "NVIDIA" },
		{ 0x1458, "Gigabyte" },
		{ 0x1462, "MSI" },
		{ 0x154B, "PNY" }, // maybe storage devices
		{ 0x1569, "Palit" },
		{ 0x1682, "XFX" },
		{ 0x196D, "Club3D" },
		{ 0x196E, "PNY" },
		{ 0x19DA, "Zotac" },
		{ 0x19F1, "BFG" },
		{ 0x1ACC, "PoV" },
		{ 0x1B4C, "Galax" }, // KFA2 in EU, to check on Pascal cards
		{ 0x3842, "EVGA" },
		{ 0x7377, "Colorful" },
		{ 0, "" }
	};

	if (!vendorname)
		return -EINVAL;

	for(int v=0; v < ARRAY_SIZE(vendors); v++) {
		if (vid == vendors[v].vid) {
			strcpy(vendorname, vendors[v].name);
			return vid;
		}
	}
	if (opt_debug && vid != 0x10DE)
		applog(LOG_DEBUG, "nvml: Unknown vendor %04x\n", vid);
	return 0;
}

int gpu_vendor(uint8_t pci_bus_id, char *vendorname)
{
	uint16_t vid = 0, pid = 0;
	if (hnvml) { // may not be initialized on start...
		for (int id=0; id < hnvml->nvml_gpucount; id++) {
			if (hnvml->nvml_pci_bus_id[id] == pci_bus_id) {
				int dev_id = hnvml->nvml_cuda_device_id[id];
				nvml_get_info(hnvml, dev_id, vid, pid);
			}
		}
	} else {
#ifdef WIN32
		for (unsigned id = 0; id < nvapi_dev_cnt; id++) {
			if (device_bus_ids[id] == pci_bus_id) {
				nvapi_getinfo(nvapi_dev_map[id], vid, pid);
				break;
			}
		}
#endif
	}
	return translate_vendor_id(vid, vendorname);
}

int gpu_info(struct cgpu_info *gpu)
{
	char vendorname[32] = { 0 };
	int id = gpu->gpu_id;
	uint8_t bus_id = 0;

	gpu->nvml_id = -1;
	gpu->nvapi_id = -1;

	if (id < 0)
		return -1;

	if (hnvml) {
		gpu->nvml_id = (int8_t) hnvml->cuda_nvml_device_id[id];
		nvml_get_info(hnvml, id, gpu->gpu_vid, gpu->gpu_pid);
		nvml_get_serial(hnvml, id, gpu->gpu_sn, sizeof(gpu->gpu_sn));
		nvml_get_bios(hnvml, id, gpu->gpu_desc, sizeof(gpu->gpu_desc));
	}
#ifdef WIN32
	gpu->nvapi_id = (int8_t) nvapi_dev_map[id];
	nvapi_getinfo(nvapi_dev_map[id], gpu->gpu_vid, gpu->gpu_pid);
	nvapi_getserial(nvapi_dev_map[id], gpu->gpu_sn, sizeof(gpu->gpu_sn));
	nvapi_getbios(nvapi_dev_map[id], gpu->gpu_desc, sizeof(gpu->gpu_desc));
#endif
	return 0;
}

#endif /* USE_WRAPNVML */

static int rgb_percent(int RGB, int percent)
{
	uint8_t* comp = (uint8_t*) &RGB;
	int res = ((percent*comp[2]) / 100) << 16;
	res += ((percent*comp[1]) / 100) << 8;
	return res + ((percent*comp[0]) / 100);
}

void gpu_led_on(int dev_id)
{
#if defined(WIN32) && defined(USE_WRAPNVML)
	int value = device_led[dev_id];
	if (device_led_state[dev_id] != value) {
		if (nvapi_set_led(nvapi_dev_map[dev_id], value, device_name[dev_id]) == 0)
			device_led_state[dev_id] = value;
	}
#endif
}

void gpu_led_percent(int dev_id, int percent)
{
#if defined(WIN32) && defined(USE_WRAPNVML)
	int value = rgb_percent(device_led[dev_id], percent);
	if (device_led_state[dev_id] != value) {
		if (nvapi_set_led(nvapi_dev_map[dev_id], value, device_name[dev_id]) == 0)
			device_led_state[dev_id] = value;
	}
#endif
}

void gpu_led_off(int dev_id)
{
#if defined(WIN32) && defined(USE_WRAPNVML)
	if (device_led_state[dev_id]) {
		if (nvapi_set_led(nvapi_dev_map[dev_id], 0, device_name[dev_id]) == 0)
			device_led_state[dev_id] = 0;
	}
#endif
}

#ifdef USE_WRAPNVML
extern double thr_hashrates[MAX_GPUS];
extern bool opt_debug_threads;
extern bool opt_hwmonitor;
extern int num_cpus;

void *monitor_thread(void *userdata)
{
	int thr_id = -1;

	while (!abort_flag && !opt_quiet)
	{
		// This thread monitors card's power lazily during scans, one at a time...
		thr_id = (thr_id + 1) % opt_n_threads;
		struct cgpu_info *cgpu = &thr_info[thr_id].gpu;
		int dev_id = cgpu->gpu_id; cudaSetDevice(dev_id);

		if (hnvml != NULL && cgpu)
		{
			char khw[32] = { 0 };
			uint64_t clock = 0, mem_clock = 0;
			uint32_t fanpercent = 0, power = 0;
			double tempC = 0, khs_per_watt = 0;
			uint32_t counter = 0;
			int max_loops = 1000;

			pthread_cond_wait(&cgpu->monitor.sampling_signal, &cgpu->monitor.lock);

			do {
				unsigned int tmp_clock=0, tmp_memclock=0;
				nvml_get_current_clocks(dev_id, &tmp_clock, &tmp_memclock);
#ifdef WIN32
				if (tmp_clock < 200) {
					// workaround for buggy drivers 378.x (real clock)
					tmp_clock = nvapi_get_gpu_clock(nvapi_dev_map[dev_id]);
				}
#endif
				if (tmp_clock < 200) {
					// some older cards only report a base clock with cuda props.
					if (cuda_gpu_info(cgpu) == 0) {
						tmp_clock = cgpu->gpu_clock/1000;
						tmp_memclock = cgpu->gpu_memclock/1000;
					}
				}
				clock += tmp_clock;
				mem_clock += tmp_memclock;
				tempC += gpu_temp(cgpu);
				fanpercent += gpu_fanpercent(cgpu);
				power += gpu_power(cgpu);
				counter++;

				usleep(50000);
				if (abort_flag) goto abort;

			} while (cgpu->monitor.sampling_flag && (--max_loops));

			cgpu->monitor.gpu_temp = (uint32_t) (tempC/counter);
			cgpu->monitor.gpu_fan = fanpercent/counter;
			cgpu->monitor.gpu_power = power/counter;
			cgpu->monitor.gpu_clock = (uint32_t) (clock/counter);
			cgpu->monitor.gpu_memclock = (uint32_t) (mem_clock/counter);

			if (power) {
				khs_per_watt = stats_get_speed(thr_id, thr_hashrates[thr_id]);
				khs_per_watt = khs_per_watt / ((double)power / counter);
				format_hashrate(khs_per_watt * 1000, khw);
				if (strlen(khw))
					sprintf(&khw[strlen(khw)-1], "W %uW ", cgpu->monitor.gpu_power / 1000);
			}

			if (opt_hwmonitor && (time(NULL) - cgpu->monitor.tm_displayed) > 60) {
				gpulog(LOG_INFO, thr_id, "%u MHz %s%uC FAN %u%%",
					cgpu->monitor.gpu_clock/*, cgpu->monitor.gpu_memclock*/,
					khw, cgpu->monitor.gpu_temp, cgpu->monitor.gpu_fan
				);
				cgpu->monitor.tm_displayed = (uint32_t)time(NULL);
			}

			pthread_mutex_unlock(&cgpu->monitor.lock);
		}
		usleep(500); // safety
	}
abort:
	if (opt_debug_threads)
		applog(LOG_DEBUG, "%s() died", __func__);
	return NULL;
}
#endif
