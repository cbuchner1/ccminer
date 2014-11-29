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
#include "nvml.h"
#include "cuda_runtime.h"

// cuda.cpp
int cuda_num_devices();

#ifdef USE_WRAPNVML

extern nvml_handle *hnvml;
extern char driver_version[32];

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

	nvmlh->nvmlInit = (nvmlReturn_t (*)(void))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlInit_v2");
	if (!nvmlh->nvmlInit) {
		nvmlh->nvmlInit = (nvmlReturn_t (*)(void))
			wrap_dlsym(nvmlh->nvml_dll, "nvmlInit");
	}
	nvmlh->nvmlDeviceGetCount = (nvmlReturn_t (*)(int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetCount_v2");
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
	nvmlh->nvmlDeviceGetClockInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *clock))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetClockInfo");
	nvmlh->nvmlDeviceGetPciInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPciInfo");
	nvmlh->nvmlDeviceGetName = (nvmlReturn_t (*)(nvmlDevice_t, char *, int))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetName");
	nvmlh->nvmlDeviceGetTemperature = (nvmlReturn_t (*)(nvmlDevice_t, int, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetTemperature");
	nvmlh->nvmlDeviceGetFanSpeed = (nvmlReturn_t (*)(nvmlDevice_t, unsigned int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetFanSpeed");
	nvmlh->nvmlDeviceGetPerformanceState = (nvmlReturn_t (*)(nvmlDevice_t, int *))
		wrap_dlsym(nvmlh->nvml_dll, "nvmlDeviceGetPowerUsage");
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

	if (nvmlh->nvmlInit == NULL ||
			nvmlh->nvmlShutdown == NULL ||
			nvmlh->nvmlErrorString == NULL ||
			nvmlh->nvmlSystemGetDriverVersion == NULL ||
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
		nvmlh->nvml_pci_subsys_id[i] = pciinfo.pci_device_id;

		nvmlh->app_clocks[i] = NVML_FEATURE_UNKNOWN;
		if (nvmlh->nvmlDeviceSetAPIRestriction) {
			nvmlh->nvmlDeviceSetAPIRestriction(nvmlh->devs[i], NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
				NVML_FEATURE_ENABLED);
			/* there is only this API_SET_APPLICATION_CLOCKS on the 750 Ti (340.58) */
		}
		if (nvmlh->nvmlDeviceGetAPIRestriction) {
			nvmlh->nvmlDeviceGetAPIRestriction(nvmlh->devs[i], NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
				&nvmlh->app_clocks[i]);
			if (nvmlh->app_clocks[i] == NVML_FEATURE_ENABLED && opt_debug) {
				applog(LOG_DEBUG, "NVML application clock feature is allowed");
#if 0
				uint32_t mem;
				nvmlReturn_t rc;
				rc = nvmlh->nvmlDeviceGetDefaultApplicationsClock(nvmlh->devs[i], NVML_CLOCK_MEM, &mem);
				if (rc == NVML_SUCCESS)
					applog(LOG_DEBUG, "nvmlDeviceGetDefaultApplicationsClock: mem %u", mem);
				else
					applog(LOG_DEBUG, "nvmlDeviceGetDefaultApplicationsClock: %s", nvmlh->nvmlErrorString(rc));
				rc = nvmlh->nvmlDeviceSetApplicationsClocks(nvmlh->devs[i], mem, 1228000);
				if (rc != NVML_SUCCESS)
					applog(LOG_DEBUG, "nvmlDeviceSetApplicationsClocks: %s", nvmlh->nvmlErrorString(rc));
#endif
			}
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
						applog(LOG_DEBUG, "CUDA GPU#%d matches NVML GPU %d by busId %u",
							i, j, (uint32_t) props.pciBusID);
					nvmlh->nvml_cuda_device_id[j] = i;
					nvmlh->cuda_nvml_device_id[i] = j;
				}
			}
		}
	}

	return nvmlh;
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
		return -1;

	if (nvmlh->nvmlDeviceGetName(nvmlh->devs[gpuindex], namebuf, bufsize) != NVML_SUCCESS)
		return -1;

	return 0;
}


int nvml_get_tempC(nvml_handle *nvmlh, int cudaindex, unsigned int *tempC)
{
	nvmlReturn_t rc;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

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
		return -1;

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
		return -1;

	nvmlReturn_t res = nvmlh->nvmlDeviceGetPowerUsage(nvmlh->devs[gpuindex], milliwatts);
	if (res != NVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetPowerUsage: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}

	return 0;
}

/* Not Supported on 750Ti 340.23 */
int nvml_get_pstate(nvml_handle *nvmlh, int cudaindex, int *pstate)
{
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

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
		return -1;

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
		return -1;

	res = nvmlh->nvmlDeviceGetSerial(nvmlh->devs[gpuindex], sn, maxlen);
	if (res == NVML_SUCCESS) {
		return 0;
	}

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
		return -1;

	nvmlReturn_t res = nvmlh->nvmlDeviceGetVbiosVersion(nvmlh->devs[gpuindex], desc, maxlen);
	if (res != NVML_SUCCESS) {
		if (opt_debug)
			applog(LOG_DEBUG, "nvmlDeviceGetVbiosVersion: %s", nvmlh->nvmlErrorString(res));
		return -1;
	}
	return 0;
}

int nvml_get_info(nvml_handle *nvmlh, int cudaindex, uint16_t *vid, uint16_t *pid)
{
	uint32_t subids = 0;
	int gpuindex = nvmlh->cuda_nvml_device_id[cudaindex];
	if (gpuindex < 0 || gpuindex >= nvmlh->nvml_gpucount)
		return -1;

	subids = nvmlh->nvml_pci_subsys_id[gpuindex];
	(*pid) = subids >> 16;
	(*vid) = subids & 0xFFFF;
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

int nvapi_getinfo(unsigned int devNum, uint16_t *vid, uint16_t *pid)
{
	NvAPI_Status ret;
	NvU32 pDeviceId, pSubSystemId, pRevisionId, pExtDeviceId;

	if (devNum >= nvapi_dev_cnt)
		return -1;

	ret = NvAPI_GPU_GetPCIIdentifiers(phys[devNum], &pDeviceId, &pSubSystemId, &pRevisionId, &pExtDeviceId);
	if (ret != NVAPI_OK) {
		NvAPI_ShortString string;
		NvAPI_GetErrorMessage(ret, string);
		if (opt_debug)
			applog(LOG_DEBUG, "NVAPI GetPCIIdentifiers: %s", string);
		return -1;
	}

	(*pid) = pDeviceId >> 16;
	(*vid) = pDeviceId & 0xFFFF;

	return 0;
}

int nvapi_getserial(unsigned int devNum, char *serial, unsigned int maxlen)
{
//	NvAPI_Status ret;
	if (devNum >= nvapi_dev_cnt)
		return -1;

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
		return -1;

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
						applog(LOG_DEBUG, "CUDA GPU#%d matches NVAPI GPU %d by busId %u",
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
		sprintf(driver_version,"%d.%d", udv/100, udv % 100);
	}

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
	return (int) pct;
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

int gpu_info(struct cgpu_info *gpu)
{
	int id = gpu->gpu_id;

	gpu->nvml_id = -1;
	gpu->nvapi_id = -1;

	if (id < 0)
		return -1;

	if (hnvml) {
		gpu->nvml_id = (int8_t) hnvml->cuda_nvml_device_id[id];
		nvml_get_info(hnvml, id, &gpu->gpu_vid, &gpu->gpu_pid);
		nvml_get_serial(hnvml, id, gpu->gpu_sn, sizeof(gpu->gpu_sn));
		nvml_get_bios(hnvml, id, gpu->gpu_desc, sizeof(gpu->gpu_desc));
	}
#ifdef WIN32
	gpu->nvapi_id = (int8_t) nvapi_dev_map[id];
	nvapi_getinfo(nvapi_dev_map[id], &gpu->gpu_vid, &gpu->gpu_pid);
	nvapi_getserial(nvapi_dev_map[id], gpu->gpu_sn, sizeof(gpu->gpu_sn));
	nvapi_getbios(nvapi_dev_map[id], gpu->gpu_desc, sizeof(gpu->gpu_desc));
#endif
	return 0;
}

#endif /* USE_WRAPNVML */
