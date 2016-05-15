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
extern uint32_t device_plimit[MAX_GPUS];
extern int8_t device_pstate[MAX_GPUS];

uint32_t clock_prev[MAX_GPUS] = { 0 };
uint32_t clock_prev_mem[MAX_GPUS] = { 0 };
uint32_t limit_prev[MAX_GPUS] = { 0 };

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

#if defined(WIN32)
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
	/* NVML_ERROR_NOT_SUPPORTED
	nvmlh->nvmlDeviceGetAutoBoostedClocksEnabled = (nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetAutoBoostedClocksEnabled");
	nvmlh->nvmlDeviceSetAutoBoostedClocksEnabled = (nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t enabled))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceSetAutoBoostedClocksEnabled"); */
	// v346
	nvmlh->nvmlDeviceGetPcieThroughput = (nvmlReturn_t (*)(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int *value))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPcieThroughput");

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

	// these functions works for the 960 and the 970 (346.72+), not for the 750 Ti
	uint32_t nclocks = 0, clocks[127] = { 0 };
	nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, NULL);
	nclocks = min(nclocks, 127);
	if (nclocks)
		nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, clocks);
	for (uint8_t u=0; u < nclocks; u++) {
		// ordered by pstate (so highest is first memory clock - P0)
		if (clocks[u] <= mem_clk) {
			mem_clk = clocks[u];
			break;
		}
	}

	nclocks = 0;
	nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, NULL);
	nclocks = min(nclocks, 127);
	if (nclocks)
		nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, clocks);
	for (uint8_t u=0; u < nclocks; u++) {
		// ordered desc, so get first
		if (clocks[u] <= gpu_clk) {
			gpu_clk = clocks[u];
			break;
		}
	}

	rc = nvmlh->nvmlDeviceSetApplicationsClocks(nvmlh->devs[n], mem_clk, gpu_clk);
	if (rc == NVML_SUCCESS)
		applog(LOG_INFO, "GPU #%d: application clocks set to %u/%u", dev_id, mem_clk, gpu_clk);
	else {
		applog(LOG_WARNING, "GPU #%d: %u/%u - %s", dev_id, mem_clk, gpu_clk, nvmlh->nvmlErrorString(rc));
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

	// these functions works for the 960 and the 970 (346.72+), not for the 750 Ti
	uint32_t nclocks = 0, clocks[127] = { 0 };
	int8_t wanted_pstate = device_pstate[dev_id];
	nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, NULL);
	nclocks = min(nclocks, 127);
	if (nclocks)
		nvmlh->nvmlDeviceGetSupportedMemoryClocks(nvmlh->devs[n], &nclocks, clocks);
	for (uint8_t u=0; u < nclocks; u++) {
		// ordered by pstate (so highest P0 first)
		if (u == wanted_pstate) {
			mem_clk = clocks[u];
			break;
		}
	}

	nclocks = 0;
	nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, NULL);
	nclocks = min(nclocks, 127);
	if (nclocks)
		nvmlh->nvmlDeviceGetSupportedGraphicsClocks(nvmlh->devs[n], mem_clk, &nclocks, clocks);
	for (uint8_t u=0; u < nclocks; u++) {
		// ordered desc, so get first
		if (clocks[u] <= gpu_clk) {
			gpu_clk = clocks[u];
			break;
		}
	}

	rc = nvmlh->nvmlDeviceSetApplicationsClocks(nvmlh->devs[n], mem_clk, gpu_clk);
	if (rc != NVML_SUCCESS) {
		applog(LOG_WARNING, "GPU #%d: pstate %s", dev_id, nvmlh->nvmlErrorString(rc));
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
		applog(LOG_WARNING, "GPU #%d: plimit %s", dev_id, nvmlh->nvmlErrorString(rc));
		return -1;
	}

	if (!opt_quiet) {
		applog(LOG_INFO, "GPU #%d: power limit set to %uW (allowed range is %u-%u)",
			dev_id, plimit/1000U, pmin/1000U, pmax/1000U);
	}

	limit_prev[dev_id] = prev_limit;
	return 1;
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
	if (!subids) subids = nvmlh->nvml_pci_device_id[gpuindex];
	pid = subids >> 16;
	vid = subids & 0xFFFF;
	return 0;
}

int nvml_destroy(nvml_handle *nvmlh)
{
	nvmlh->nvmlShutdown();

	wrap_dlclose(nvmlh->nvml_dll);

	free(nvmlh->nvml_pci_bus_id);
	free(nvmlh->nvml_pci_device_id);
	free(nvmlh->nvml_pci_domain_id);
	free(nvmlh->nvml_pci_subsys_id);
	free(nvmlh->nvml_cuda_device_id);
	free(nvmlh->cuda_nvml_device_id);
	free(nvmlh->app_clocks);
	free(nvmlh->devs);

	free(nvmlh);
	return 0;
}

/**
 * nvapi alternative for windows x86 binaries
 * nvml api doesn't exists as 32bit dll :///
 */
#ifdef WIN32
#include "nvapi/nvapi_ccminer.h"

static int nvapi_dev_map[MAX_GPUS] = { 0 };
static NvDisplayHandle hDisplay_a[NVAPI_MAX_PHYSICAL_GPUS * 2] = { 0 };
static NvPhysicalGpuHandle phys[NVAPI_MAX_PHYSICAL_GPUS] = { 0 };
static NvU32 nvapi_dev_cnt = 0;

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

int nvapi_getpstate(unsigned int devNum, unsigned int *power)
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
		(*power) = (unsigned int)CurrentPstate;
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
	}

	return 0;
}

int nvapi_getserial(unsigned int devNum, char *serial, unsigned int maxlen)
{
//	NvAPI_Status ret;
	if (devNum >= nvapi_dev_cnt)
		return -ENODEV;

	sprintf(serial, "");

	if (maxlen < 64) // Short String
		return -1;

#if 0
	ret = NvAPI_GPU_Get..(phys[devNum], serial);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI ...: %s", string);
		return -1;
	}
#endif
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

int nvapi_init()
{
	int num_gpus = cuda_num_devices();
	NvAPI_Status ret = NvAPI_Initialize();
	if (!ret == NVAPI_OK){
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
	NvAPI_ShortString ver;
	NvAPI_GetInterfaceVersionString(ver);
	applog(LOG_DEBUG, "NVAPI Version: %s", ver);
#endif

	NvU32 udv;
	NvAPI_ShortString str;
	ret = NvAPI_SYS_GetDriverAndBranchVersion(&udv, str);
	if (ret == NVAPI_OK) {
		sprintf(driver_version,"%d.%02d", udv / 100, udv % 100);
	}

	return 0;
}
#endif

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

/* not used in api (too much variable) */
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
		mw = pct; // to fix
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
		{ 0x107D, "Leadtek" },
		{ 0x10B0, "Gainward" },
		// { 0x10DE, "NVIDIA" },
		{ 0x1458, "Gigabyte" },
		{ 0x1462, "MSI" },
		{ 0x154B, "PNY" },
		{ 0x1682, "XFX" },
		{ 0x196D, "Club3D" },
		{ 0x19DA, "Zotac" },
		{ 0x19F1, "BFG" },
		{ 0x1ACC, "PoV" },
		{ 0x1B4C, "KFA2" },
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

#ifdef HAVE_PCIDEV
extern "C" {
#include <pci/pci.h>
}
static int linux_gpu_vendor(uint8_t pci_bus_id, char* vendorname, uint16_t &pid)
{
	uint16_t subvendor = 0;
	struct pci_access *pci;
	struct pci_dev *dev;
	uint16_t subdevice;

	if (!vendorname)
		return -EINVAL;

	pci = pci_alloc();
	if (!pci)
		return -ENODEV;

	pci_init(pci);
	pci_scan_bus(pci);

	for(dev = pci->devices; dev; dev = dev->next)
	{
		if (dev->bus == pci_bus_id  && dev->vendor_id == 0x10DE)
		{
			if (!(dev->known_fields & PCI_FILL_CLASS))
				pci_fill_info(dev, PCI_FILL_CLASS);
			if (dev->device_class != PCI_CLASS_DISPLAY_VGA)
				continue;
			subvendor = pci_read_word(dev, PCI_SUBSYSTEM_VENDOR_ID);
			subdevice = pci_read_word(dev, PCI_SUBSYSTEM_ID); // model

			translate_vendor_id(subvendor, vendorname);
		}
	}
	pci_cleanup(pci);
	return (int) subvendor;
}
#endif

int gpu_vendor(uint8_t pci_bus_id, char *vendorname)
{
#ifdef HAVE_PCIDEV
	uint16_t pid = 0;
	return linux_gpu_vendor(pci_bus_id, vendorname, pid);
#else
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
#endif
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
#ifdef HAVE_PCIDEV
		gpu->gpu_vid = linux_gpu_vendor(hnvml->nvml_pci_bus_id[id], vendorname, gpu->gpu_pid);
		if (!gpu->gpu_vid || !gpu->gpu_pid)
			nvml_get_info(hnvml, id, gpu->gpu_vid, gpu->gpu_pid);
#else
		nvml_get_info(hnvml, id, gpu->gpu_vid, gpu->gpu_pid);
#endif
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
