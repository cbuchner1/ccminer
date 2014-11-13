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

#ifdef USE_WRAPNVML

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#ifndef _MSC_VER
#include <libgen.h>
#endif

#include "miner.h"
#include "cuda_runtime.h"
#include "nvml.h"

/*
 * Wrappers to emulate dlopen() on other systems like Windows
 */
#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
	static void *wrap_dlopen(const char *filename) {
		return (void *)LoadLibrary(filename);
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
	static void *wrap_dlopen(const char *filename) {
		return dlopen(filename, RTLD_NOW);
	}
	static void *wrap_dlsym(void *h, const char *sym) {
		return dlsym(h, sym);
	}
	static int wrap_dlclose(void *h) {
		return dlclose(h);
	}
#endif

#if defined(__cplusplus)
extern "C" {
#endif

wrap_nvml_handle * wrap_nvml_create()
{
	int i=0;
	wrap_nvml_handle *nvmlh = NULL;

	/*
	 * We use hard-coded library installation locations for the time being...
	 * No idea where or if libnvidia-ml.so is installed on MacOS X, a
	 * deep scouring of the filesystem on one of the Mac CUDA build boxes
	 * I used turned up nothing, so for now it's not going to work on OSX.
	 */
#if defined(_WIN64)
	/* 64-bit Windows */
#define  libnvidia_ml "%PROGRAMFILES%/NVIDIA Corporation/NVSMI/nvml.dll"
#elif defined(_WIN32) || defined(_MSC_VER)
	/* 32-bit Windows */
#define  libnvidia_ml "%PROGRAMFILES%/NVIDIA Corporation/NVSMI/nvml.dll"
#elif defined(__linux) && (defined(__i386__) || defined(__ARM_ARCH_7A__))
	/* 32-bit linux assumed */
#define  libnvidia_ml "/usr/lib32/libnvidia-ml.so"
#elif defined(__linux)
	/* 64-bit linux assumed */
#define  libnvidia_ml "/usr/lib/libnvidia-ml.so"
#else
#error "Unrecognized platform: need NVML DLL path for this platform..."
#endif

#if WIN32
	char tmp[512];
	ExpandEnvironmentStringsA(libnvidia_ml, tmp, sizeof(tmp));
#else
	char tmp[512] = libnvidia_ml;
#endif

	void *nvml_dll = wrap_dlopen(tmp);
	if (nvml_dll == NULL) {
#ifdef WIN32
		char lib[] = "nvml.dll";
#else
		char lib[64] = { '\0' };
		snprintf(lib, sizeof(lib), "%s", basename(tmp));
		/* try dlopen without path, here /usr/lib/nvidia-340/libnvidia-ml.so */
#endif
		nvml_dll = wrap_dlopen(lib);
		if (opt_debug)
			applog(LOG_DEBUG, "dlopen: %s=%p", lib, nvml_dll);
	}
	if (nvml_dll == NULL) {
		if (opt_debug)
			applog(LOG_DEBUG, "dlopen(%d): failed to load %s", errno, tmp);
		return NULL;
	}

	nvmlh = (wrap_nvml_handle *) calloc(1, sizeof(wrap_nvml_handle));

	nvmlh->nvml_dll = nvml_dll;

	nvmlh->nvmlInit = (wrap_nvmlReturn_t (*)(void))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlInit_v2");
	if (!nvmlh->nvmlInit)
		nvmlh->nvmlInit = (wrap_nvmlReturn_t (*)(void))
			wrap_dlsym(nvmlh->nvml_dll, "nvmlInit");
	nvmlh->nvmlDeviceGetCount = (wrap_nvmlReturn_t (*)(int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
	nvmlh->nvmlDeviceGetHandleByIndex = (wrap_nvmlReturn_t (*)(int, wrap_nvmlDevice_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
	nvmlh->nvmlDeviceGetClockInfo = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, wrap_nvmlClockType_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetClockInfo");
	nvmlh->nvmlDeviceGetPciInfo = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, wrap_nvmlPciInfo_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo");
	nvmlh->nvmlDeviceGetName = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, char *, int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetName");
	nvmlh->nvmlDeviceGetTemperature = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, int, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetTemperature");
	nvmlh->nvmlDeviceGetFanSpeed = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFanSpeed");
	nvmlh->nvmlDeviceGetPerformanceState = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
	nvmlh->nvmlDeviceGetPowerUsage = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
	nvmlh->nvmlErrorString = (char* (*)(wrap_nvmlReturn_t))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlErrorString");
	nvmlh->nvmlShutdown = (wrap_nvmlReturn_t (*)())
		wrap_dlsym(nvmlh->nvml_dll, "nvmlShutdown");

	if (nvmlh->nvmlInit == NULL ||
			nvmlh->nvmlShutdown == NULL ||
			nvmlh->nvmlDeviceGetCount == NULL ||
			nvmlh->nvmlDeviceGetHandleByIndex == NULL ||
			nvmlh->nvmlDeviceGetPciInfo == NULL ||
			nvmlh->nvmlDeviceGetName == NULL ||
			nvmlh->nvmlDeviceGetTemperature == NULL ||
			nvmlh->nvmlDeviceGetFanSpeed == NULL ||
			nvmlh->nvmlDeviceGetPowerUsage == NULL)
	{
		if (opt_debug)
			applog(LOG_DEBUG, "Failed to obtain all required NVML function pointers");
		wrap_dlclose(nvmlh->nvml_dll);
		free(nvmlh);
		return NULL;
	}

	nvmlh->nvmlInit();
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

	nvmlh->devs = (wrap_nvmlDevice_t *) calloc(nvmlh->nvml_gpucount, sizeof(wrap_nvmlDevice_t));
	nvmlh->nvml_pci_domain_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_bus_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_device_id = (unsigned int*) calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_cuda_device_id = (int*) calloc(nvmlh->nvml_gpucount, sizeof(int));
	nvmlh->cuda_nvml_device_id = (int*) calloc(nvmlh->cuda_gpucount, sizeof(int));

	/* Obtain GPU device handles we're going to need repeatedly... */
	for (i=0; i<nvmlh->nvml_gpucount; i++) {
		nvmlh->nvmlDeviceGetHandleByIndex(i, &nvmlh->devs[i]);
	}

	/* Query PCI info for each NVML device, and build table for mapping of */
	/* CUDA device IDs to NVML device IDs and vice versa                   */
	for (i=0; i<nvmlh->nvml_gpucount; i++) {
		wrap_nvmlPciInfo_t pciinfo;
		nvmlh->nvmlDeviceGetPciInfo(nvmlh->devs[i], &pciinfo);
		nvmlh->nvml_pci_domain_id[i] = pciinfo.domain;
		nvmlh->nvml_pci_bus_id[i]    = pciinfo.bus;
		nvmlh->nvml_pci_device_id[i] = pciinfo.device;
	}

	/* build mapping of NVML device IDs to CUDA IDs */
	for (i=0; i<nvmlh->nvml_gpucount; i++) {
		nvmlh->nvml_cuda_device_id[i] = -1;
	}
	for (i=0; i<nvmlh->cuda_gpucount; i++) {
		cudaDeviceProp props;
		nvmlh->cuda_nvml_device_id[i] = -1;

		if (cudaGetDeviceProperties(&props, i) == cudaSuccess) {
			int j;
			for (j=0; j<nvmlh->nvml_gpucount; j++) {
				if ((nvmlh->nvml_pci_domain_id[j] == (uint32_t) props.pciDomainID) &&
				    (nvmlh->nvml_pci_bus_id[j]    == (uint32_t) props.pciBusID) &&
				    (nvmlh->nvml_pci_device_id[j] == (uint32_t) props.pciDeviceID)) {
					if (opt_debug)
						applog(LOG_DEBUG, "CUDA GPU[%d] matches NVML GPU[%d]", i, j);
					nvmlh->nvml_cuda_device_id[j] = i;
					nvmlh->cuda_nvml_device_id[i] = j;
				}
			}
		}
	}

	return nvmlh;
}

int wrap_nvml_get_gpucount(wrap_nvml_handle *nvmlh, int *gpucount)
{
	*gpucount = nvmlh->nvml_gpucount;
	return 0;
}

int wrap_cuda_get_gpucount(wrap_nvml_handle *nvmlh, int *gpucount)
{
	*gpucount = nvmlh->cuda_gpucount;
	return 0;
}

int wrap_nvml_get_gpu_name(wrap_nvml_handle *nvmlh, int cudaindex, char *namebuf, int bufsize)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	if (nvmlh->nvmlDeviceGetName(nvmlh->devs[gpuindex], namebuf, bufsize) != WRAPNVML_SUCCESS)
		return -1;

	return 0;
}


int wrap_nvml_get_tempC(wrap_nvml_handle *nvmlh, int cudaindex, unsigned int *tempC)
{
	wrap_nvmlReturn_t rc;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	rc = nvmlh->nvmlDeviceGetTemperature(nvmlh->devs[gpuindex], 0u /* NVML_TEMPERATURE_GPU */, tempC);
	if (rc != WRAPNVML_SUCCESS) {
		return -1;
	}

	return 0;
}


int wrap_nvml_get_fanpcnt(wrap_nvml_handle *nvmlh, int cudaindex, unsigned int *fanpcnt)
{
	wrap_nvmlReturn_t rc;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	rc = nvmlh->nvmlDeviceGetFanSpeed(nvmlh->devs[gpuindex], fanpcnt);
	if (rc != WRAPNVML_SUCCESS) {
		return -1;
	}

	return 0;
}

/* Not Supported on 750Ti 340.23 */
int wrap_nvml_get_clock(wrap_nvml_handle *nvmlh, int cudaindex, int type, unsigned int *freq)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	wrap_nvmlReturn_t res = nvmlh->nvmlDeviceGetClockInfo(nvmlh->devs[gpuindex], (wrap_nvmlClockType_t) type, freq);
	if (res != WRAPNVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetClockInfo: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

/* Not Supported on 750Ti 340.23 */
int wrap_nvml_get_power_usage(wrap_nvml_handle *nvmlh, int cudaindex, unsigned int *milliwatts)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	wrap_nvmlReturn_t res = nvmlh->nvmlDeviceGetPowerUsage(nvmlh->devs[gpuindex], milliwatts);
	if (res != WRAPNVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetPowerUsage: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

/* Not Supported on 750Ti 340.23 */
int wrap_nvml_get_pstate(wrap_nvml_handle *nvmlh, int cudaindex, int *pstate)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	wrap_nvmlReturn_t res = nvmlh->nvmlDeviceGetPerformanceState(nvmlh->devs[gpuindex], pstate);
	if (res != WRAPNVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetPerformanceState: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

int wrap_nvml_destroy(wrap_nvml_handle *nvmlh)
{
	nvmlh->nvmlShutdown();

	wrap_dlclose(nvmlh->nvml_dll);
	free(nvmlh);
	return 0;
}

/* api functions */

extern wrap_nvml_handle *nvmlh;
extern int device_map[8];

unsigned int gpu_fanpercent(struct cgpu_info *gpu)
{
	unsigned int pct = 0;
	if (nvmlh) {
		wrap_nvml_get_fanpcnt(nvmlh, device_map[gpu->thr_id], &pct);
	}
	return pct;
}

double gpu_temp(struct cgpu_info *gpu)
{
	double tc = 0.0;
	if (nvmlh) {
		unsigned int tmp = 0;
		wrap_nvml_get_tempC(nvmlh, device_map[gpu->thr_id], &tmp);
		tc = (double) tmp;
	}
	return tc;
}

unsigned int gpu_clock(struct cgpu_info *gpu)
{
	unsigned int freq = 0;
	if (nvmlh) {
		wrap_nvml_get_clock(nvmlh, device_map[gpu->thr_id], NVML_CLOCK_SM, &freq);
	}
	return freq;
}

unsigned int gpu_power(struct cgpu_info *gpu)
{
	unsigned int mw = 0;
	if (nvmlh) {
		wrap_nvml_get_power_usage(nvmlh, device_map[gpu->thr_id], &mw);
	}
	return mw;
}

int gpu_pstate(struct cgpu_info *gpu)
{
	int pstate = 0;
	if (nvmlh) {
		wrap_nvml_get_pstate(nvmlh, device_map[gpu->thr_id], &pstate);
		//gpu->gpu_pstate = pstate;
	}
	return pstate;
}

#if defined(__cplusplus)
}
#endif

#endif /* USE_WRAPNVML */

/* strings /usr/lib/nvidia-340/libnvidia-ml.so | grep nvmlDeviceGet | grep -v : | sort | uniq

	nvmlDeviceGetAccountingBufferSize
	nvmlDeviceGetAccountingMode
	nvmlDeviceGetAccountingPids
	nvmlDeviceGetAccountingStats
	nvmlDeviceGetAPIRestriction
	nvmlDeviceGetApplicationsClock
	nvmlDeviceGetAutoBoostedClocksEnabled
	nvmlDeviceGetBAR1MemoryInfo
	nvmlDeviceGetBoardId
	nvmlDeviceGetBrand
	nvmlDeviceGetBridgeChipInfo
*	nvmlDeviceGetClockInfo
	nvmlDeviceGetComputeMode
	nvmlDeviceGetComputeRunningProcesses
	nvmlDeviceGetCount
	nvmlDeviceGetCount_v2
	nvmlDeviceGetCpuAffinity
	nvmlDeviceGetCurrentClocksThrottleReasons
	nvmlDeviceGetCurrPcieLinkGeneration
	nvmlDeviceGetCurrPcieLinkWidth
	nvmlDeviceGetDecoderUtilization
	nvmlDeviceGetDefaultApplicationsClock
	nvmlDeviceGetDetailedEccErrors
	nvmlDeviceGetDisplayActive
	nvmlDeviceGetDisplayMode
	nvmlDeviceGetDriverModel
	nvmlDeviceGetEccMode
	nvmlDeviceGetEncoderUtilization
	nvmlDeviceGetEnforcedPowerLimit
*	nvmlDeviceGetFanSpeed
	nvmlDeviceGetGpuOperationMode
	nvmlDeviceGetHandleByIndex
	nvmlDeviceGetHandleByIndex_v2
	nvmlDeviceGetHandleByPciBusId
	nvmlDeviceGetHandleByPciBusId_v2
	nvmlDeviceGetHandleBySerial
	nvmlDeviceGetHandleByUUID
	nvmlDeviceGetIndex
	nvmlDeviceGetInforomConfigurationChecksum
	nvmlDeviceGetInforomImageVersion
	nvmlDeviceGetInforomVersion
	nvmlDeviceGetMaxClockInfo
	nvmlDeviceGetMaxPcieLinkGeneration
	nvmlDeviceGetMaxPcieLinkWidth
	nvmlDeviceGetMemoryErrorCounter
	nvmlDeviceGetMemoryInfo
	nvmlDeviceGetMinorNumber
	nvmlDeviceGetMultiGpuBoard
	nvmlDeviceGetName
	nvmlDeviceGetPciInfo
	nvmlDeviceGetPciInfo_v2
*	nvmlDeviceGetPerformanceState
	nvmlDeviceGetPersistenceMode
	nvmlDeviceGetPowerManagementDefaultLimit
	nvmlDeviceGetPowerManagementLimit
	nvmlDeviceGetPowerManagementLimitConstraints
	nvmlDeviceGetPowerManagementMode
	nvmlDeviceGetPowerState (deprecated)
*	nvmlDeviceGetPowerUsage
	nvmlDeviceGetRetiredPages
	nvmlDeviceGetRetiredPagesPendingStatus
	nvmlDeviceGetSamples
	nvmlDeviceGetSerial
	nvmlDeviceGetSupportedClocksThrottleReasons
	nvmlDeviceGetSupportedEventTypes
	nvmlDeviceGetSupportedGraphicsClocks
	nvmlDeviceGetSupportedMemoryClocks
	nvmlDeviceGetTemperature
	nvmlDeviceGetTemperatureThreshold
	nvmlDeviceGetTotalEccErrors
	nvmlDeviceGetUtilizationRates
	nvmlDeviceGetUUID
	nvmlDeviceGetVbiosVersion
	nvmlDeviceGetViolationStatus

*/