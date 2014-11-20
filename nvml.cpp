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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _MSC_VER
#include <libgen.h>
#endif

#include "miner.h"
#include "cuda_runtime.h"

#ifdef USE_WRAPNVML

#include "nvml.h"

extern wrap_nvml_handle *hnvml;
extern int num_processors; // gpus

static uint32_t device_bus_ids[8] = { 0 };

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

wrap_nvml_handle * wrap_nvml_create()
{
	int i=0;
	wrap_nvml_handle *nvmlh = NULL;

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

	nvmlh = (wrap_nvml_handle *) calloc(1, sizeof(wrap_nvml_handle));

	nvmlh->nvml_dll = nvml_dll;

	nvmlh->nvmlInit = (wrap_nvmlReturn_t (*)(void))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlInit_v2");
	if (!nvmlh->nvmlInit) {
		nvmlh->nvmlInit = (wrap_nvmlReturn_t (*)(void))
			wrap_dlsym(nvmlh->nvml_dll, "nvmlInit");
	}
	nvmlh->nvmlDeviceGetCount = (wrap_nvmlReturn_t (*)(int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
	nvmlh->nvmlDeviceGetHandleByIndex = (wrap_nvmlReturn_t (*)(int, wrap_nvmlDevice_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
	nvmlh->nvmlDeviceGetApplicationsClock = (wrap_nvmlReturn_t (*)(wrap_nvmlDevice_t, wrap_nvmlClockType_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetApplicationsClock");
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
			nvmlh->nvmlDeviceGetFanSpeed == NULL)
	{
		if (opt_debug)
			applog(LOG_DEBUG, "Failed to obtain required NVML function pointers");
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
	nvmlh->nvml_pci_device_id = (unsigned int*)calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
	nvmlh->nvml_pci_subsys_id = (unsigned int*)calloc(nvmlh->nvml_gpucount, sizeof(unsigned int));
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
		nvmlh->nvml_pci_subsys_id[i] = pciinfo.pci_subsystem_id;
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
			device_bus_ids[i] = props.pciBusID;
			for (j = 0; j<nvmlh->nvml_gpucount; j++) {
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

/* Not Supported on 750Ti 340.23, 346.16 neither */
int wrap_nvml_get_clock(wrap_nvml_handle *nvmlh, int cudaindex, int type, unsigned int *freq)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	// wrap_nvmlReturn_t rc = nvmlh->nvmlDeviceGetApplicationsClock(nvmlh->devs[gpuindex], (wrap_nvmlClockType_t)type, freq);
	wrap_nvmlReturn_t rc = nvmlh->nvmlDeviceGetClockInfo(nvmlh->devs[gpuindex], (wrap_nvmlClockType_t) type, freq);
	if (rc != WRAPNVML_SUCCESS) {
		//if (opt_debug)
		//	applog(LOG_DEBUG, "nvmlDeviceGetClockInfo: %s", nvmlh->nvmlErrorString(rc));
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
		//if (opt_debug)
		//	applog(LOG_DEBUG, "nvmlDeviceGetPerformanceState: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

int wrap_nvml_get_busid(wrap_nvml_handle *nvmlh, int cudaindex, int *busid)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	(*busid) = nvmlh->nvml_pci_bus_id[gpuindex];
	return 0;
}

int wrap_nvml_get_info(wrap_nvml_handle *nvmlh, int cudaindex, uint16_t *vid, uint16_t *pid)
{
	uint32_t subids = 0;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	subids = nvmlh->nvml_pci_subsys_id[gpuindex];
	(*vid) = subids >> 16;
	(*pid) = subids & 0xFFFF;
	return 0;
}

int wrap_nvml_destroy(wrap_nvml_handle *nvmlh)
{
	nvmlh->nvmlShutdown();

	wrap_dlclose(nvmlh->nvml_dll);

	free(nvmlh->nvml_pci_bus_id);
	free(nvmlh->nvml_pci_device_id);
	free(nvmlh->nvml_pci_domain_id);
	free(nvmlh->nvml_pci_subsys_id);
	free(nvmlh->nvml_cuda_device_id);
	free(nvmlh->cuda_nvml_device_id);
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

static int nvapi_dev_map[8] = { 0 };
static NvDisplayHandle hDisplay_a[NVAPI_MAX_PHYSICAL_GPUS * 2] = { 0 };
static NvPhysicalGpuHandle phys[NVAPI_MAX_PHYSICAL_GPUS] = { 0 };
static NvU32 nvapi_dev_cnt = 0;

int nvapi_temperature(unsigned int devNum, unsigned int *temperature)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -1;

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
		return -1;

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

int nvapi_getclock(unsigned int devNum, unsigned int *freq)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -1;

	NV_GPU_CLOCK_FREQUENCIES_V2 clocks;
	clocks.version = NV_GPU_CLOCK_FREQUENCIES_VER_2;
	clocks.ClockType = NV_GPU_CLOCK_FREQUENCIES_CURRENT_FREQ; // CURRENT/BASE/BOOST
	ret = NvAPI_GPU_GetAllClockFrequencies(phys[devNum], &clocks);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI NvAPI_GPU_GetAllClockFrequencies: %s", string);
		return -1;
	} else {
		// GRAPHICS/MEMORY
		(*freq) = (unsigned int)clocks.domain[NVAPI_GPU_PUBLIC_CLOCK_GRAPHICS].frequency;
	}

	return 0;
}

int nvapi_getpstate(unsigned int devNum, unsigned int *power)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -1;

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
		return -1;

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

int nvapi_getinfo(unsigned int devNum, char *desc)
{
	NvAPI_Status ret;

	if (devNum >= nvapi_dev_cnt)
		return -1;

	// bios rev
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

int nvapi_getbusid(unsigned int devNum, int *busid)
{
	if (devNum >= 0 && devNum <= 8) {
		(*busid) = device_bus_ids[devNum];
		return 0;
	}
	return -1;
}

int wrap_nvapi_init()
{
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

	for (int g = 0; g < num_processors; g++) {
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
			for (int g = 0; g < num_processors; g++) {
				NvU32 busId;
				ret = NvAPI_GPU_GetBusId(phys[i], &busId);
				if (ret == NVAPI_OK && busId == device_bus_ids[g]) {
					nvapi_dev_map[g] = i;
					if (opt_debug)
						applog(LOG_DEBUG, "CUDA GPU[%d] matches NVAPI GPU[%d]",
							g, i);
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

	return 0;
}
#endif

/* api functions -------------------------------------- */

// assume 2500 rpm as default, auto-updated if more
static unsigned int fan_speed_max = 2500;

int gpu_fanpercent(struct cgpu_info *gpu)
{
	unsigned int pct = 0;
	if (hnvml) {
		wrap_nvml_get_fanpcnt(hnvml, gpu->gpu_id, &pct);
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
	return (int) pct;
}

float gpu_temp(struct cgpu_info *gpu)
{
	float tc = 0.0;
	unsigned int tmp = 0;
	if (hnvml) {
		wrap_nvml_get_tempC(hnvml, gpu->gpu_id, &tmp);
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
		support = wrap_nvml_get_pstate(hnvml, gpu->gpu_id, &pstate);
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
		support = wrap_nvml_get_busid(hnvml, gpu->gpu_id, &busid);
	}
#ifdef WIN32
	if (support == -1) {
		nvapi_getbusid(nvapi_dev_map[gpu->gpu_id], &busid);
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
		support = wrap_nvml_get_power_usage(hnvml, gpu->gpu_id, &mw);
	}
#ifdef WIN32
	if (support == -1) {
		unsigned int pct = 0;
		nvapi_getusage(nvapi_dev_map[gpu->gpu_id], &pct);
	}
#endif
	return mw;
}

int gpu_info(struct cgpu_info *gpu)
{
	if (hnvml) {
		wrap_nvml_get_info(hnvml, gpu->gpu_id, &gpu->gpu_vid, &gpu->gpu_pid);
	}
#ifdef WIN32
	nvapi_getinfo(nvapi_dev_map[gpu->gpu_id], &gpu->gpu_desc[0]);
#endif
	return 0;
}

#endif /* USE_WRAPNVML */

int gpu_clocks(struct cgpu_info *gpu)
{
	cudaDeviceProp props;
	if (cudaGetDeviceProperties(&props, gpu->gpu_id) == cudaSuccess) {
		gpu->gpu_clock = props.clockRate;
		gpu->gpu_memclock = props.memoryClockRate;
		gpu->gpu_mem = props.totalGlobalMem;
		return 0;
	}
	return -1;
}

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
*	nvmlDeviceGetHandleByIndex_v2
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
*	nvmlDeviceGetName
*	nvmlDeviceGetPciInfo
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
