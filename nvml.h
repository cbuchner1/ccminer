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
 *
 */
#ifdef USE_WRAPNVML

#include "miner.h"

typedef void * nvmlDevice_t;

/* our own version of the PCI info struct */
typedef struct {
	char bus_id_str[16];             /* string form of bus info */
	unsigned int domain;
	unsigned int bus;
	unsigned int device;
	unsigned int pci_device_id;      /* combined device and vendor id */
	unsigned int pci_subsystem_id;
	unsigned int res0;               /* NVML internal use only */
	unsigned int res1;
	unsigned int res2;
	unsigned int res3;
} nvmlPciInfo_t;

enum nvmlEnableState_t {
	NVML_FEATURE_DISABLED = 0,
	NVML_FEATURE_ENABLED = 1,
	NVML_FEATURE_UNKNOWN = 2
};

enum nvmlRestrictedAPI_t {
	NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS = 0,
	NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1,
	NVML_RESTRICTED_API_COUNT = 2
};

enum nvmlReturn_t {
	NVML_SUCCESS = 0,
	NVML_ERROR_UNINITIALIZED = 1,
	NVML_ERROR_INVALID_ARGUMENT = 2,
	NVML_ERROR_NOT_SUPPORTED = 3,
	NVML_ERROR_NO_PERMISSION = 4,
	NVML_ERROR_ALREADY_INITIALIZED = 5,
	NVML_ERROR_NOT_FOUND = 6,
	NVML_ERROR_INSUFFICIENT_SIZE = 7,
	NVML_ERROR_INSUFFICIENT_POWER = 8,
	NVML_ERROR_DRIVER_NOT_LOADED = 9,
	NVML_ERROR_TIMEOUT = 10,
	NVML_ERROR_UNKNOWN = 999
};

enum nvmlClockType_t {
	NVML_CLOCK_GRAPHICS = 0,
	NVML_CLOCK_SM = 1,
	NVML_CLOCK_MEM = 2
};

enum nvmlPcieUtilCounter_t {
	NVML_PCIE_UTIL_TX_BYTES = 0,
	NVML_PCIE_UTIL_RX_BYTES = 1,
	NVML_PCIE_UTIL_COUNT
};

enum nvmlValueType_t {
	NVML_VALUE_TYPE_DOUBLE = 0,
	NVML_VALUE_TYPE_UNSIGNED_INT = 1,
	NVML_VALUE_TYPE_UNSIGNED_LONG = 2,
	NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
	NVML_VALUE_TYPE_COUNT
};

#define NVML_DEVICE_SERIAL_BUFFER_SIZE 30
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE 32

/*
 * Handle to hold the function pointers for the entry points we need,
 * and the shared library itself.
 */
typedef struct {
	void *nvml_dll;
	int nvml_gpucount;
	int cuda_gpucount;
	unsigned int *nvml_pci_domain_id;
	unsigned int *nvml_pci_bus_id;
	unsigned int *nvml_pci_device_id;
	unsigned int *nvml_pci_subsys_id;
	int *nvml_cuda_device_id;          /* map NVML dev to CUDA dev */
	int *cuda_nvml_device_id;          /* map CUDA dev to NVML dev */
	nvmlDevice_t *devs;
	nvmlEnableState_t *app_clocks;
	nvmlReturn_t (*nvmlInit)(void);
	nvmlReturn_t (*nvmlDeviceGetCount)(int *);
	nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(int, nvmlDevice_t *);
	nvmlReturn_t (*nvmlDeviceGetAPIRestriction)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t *);
	nvmlReturn_t (*nvmlDeviceSetAPIRestriction)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t);
	nvmlReturn_t (*nvmlDeviceGetDefaultApplicationsClock)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
	nvmlReturn_t (*nvmlDeviceGetApplicationsClock)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
	nvmlReturn_t (*nvmlDeviceSetApplicationsClocks)(nvmlDevice_t, unsigned int, unsigned int);
	nvmlReturn_t (*nvmlDeviceResetApplicationsClocks)(nvmlDevice_t);
	nvmlReturn_t (*nvmlDeviceGetSupportedGraphicsClocks)(nvmlDevice_t, uint32_t mem, uint32_t *num, uint32_t *arr);
	nvmlReturn_t (*nvmlDeviceGetSupportedMemoryClocks)(nvmlDevice_t, unsigned int *count, unsigned int *clocksMHz);
	nvmlReturn_t (*nvmlDeviceGetClockInfo)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
	nvmlReturn_t (*nvmlDeviceGetMaxClockInfo)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
	nvmlReturn_t (*nvmlDeviceGetPowerManagementDefaultLimit)(nvmlDevice_t, unsigned int *limit);
	nvmlReturn_t (*nvmlDeviceGetPowerManagementLimit)(nvmlDevice_t, unsigned int *limit);
	nvmlReturn_t (*nvmlDeviceGetPowerManagementLimitConstraints)(nvmlDevice_t, unsigned int *min, unsigned int *max);
	nvmlReturn_t (*nvmlDeviceSetPowerManagementLimit)(nvmlDevice_t device, unsigned int limit);
	nvmlReturn_t (*nvmlDeviceGetPciInfo)(nvmlDevice_t, nvmlPciInfo_t *);
	nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkGeneration)(nvmlDevice_t device, unsigned int *gen);
	nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkWidth)(nvmlDevice_t device, unsigned int *width);
	nvmlReturn_t (*nvmlDeviceGetMaxPcieLinkGeneration)(nvmlDevice_t device, unsigned int *gen);
	nvmlReturn_t (*nvmlDeviceGetMaxPcieLinkWidth)(nvmlDevice_t device, unsigned int *width);
	nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t, char *, int);
	nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int *);
	nvmlReturn_t (*nvmlDeviceGetFanSpeed)(nvmlDevice_t, unsigned int *);
	nvmlReturn_t (*nvmlDeviceGetPerformanceState)(nvmlDevice_t, int *); /* enum */
	nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int *);
	nvmlReturn_t (*nvmlDeviceGetSerial)(nvmlDevice_t, char *serial, unsigned int len);
	nvmlReturn_t (*nvmlDeviceGetUUID)(nvmlDevice_t, char *uuid, unsigned int len);
	nvmlReturn_t (*nvmlDeviceGetVbiosVersion)(nvmlDevice_t, char *version, unsigned int len);
	nvmlReturn_t (*nvmlSystemGetDriverVersion)(char *version, unsigned int len);
	char* (*nvmlErrorString)(nvmlReturn_t);
	nvmlReturn_t (*nvmlShutdown)(void);
	// v331
	nvmlReturn_t (*nvmlDeviceGetEnforcedPowerLimit)(nvmlDevice_t, unsigned int *limit);
	// v340
	//nvmlReturn_t (*nvmlDeviceGetCpuAffinity)(nvmlDevice_t, unsigned int cpuSetSize, unsigned long* cpuSet);
	//nvmlReturn_t (*nvmlDeviceSetCpuAffinity)(nvmlDevice_t);
	//nvmlReturn_t (*nvmlDeviceGetAutoBoostedClocksEnabled)(nvmlDevice_t, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled);
	//nvmlReturn_t (*nvmlDeviceSetAutoBoostedClocksEnabled)(nvmlDevice_t, nvmlEnableState_t enabled);
	// v346
	nvmlReturn_t (*nvmlDeviceGetPcieThroughput)(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int *value);
} nvml_handle;


nvml_handle * nvml_create();
int nvml_destroy(nvml_handle *nvmlh);

/*
 * Query the number of GPUs seen by NVML
 */
int nvml_get_gpucount(nvml_handle *nvmlh, int *gpucount);

int nvml_set_plimit(nvml_handle *nvmlh, int dev_id);
int nvml_set_pstate(nvml_handle *nvmlh, int dev_id);

int nvml_set_clocks(nvml_handle *nvmlh, int dev_id);
int nvml_reset_clocks(nvml_handle *nvmlh, int dev_id);

/* api functions */

unsigned int gpu_fanpercent(struct cgpu_info *gpu);
unsigned int gpu_fanrpm(struct cgpu_info *gpu);
float gpu_temp(struct cgpu_info *gpu);
unsigned int gpu_power(struct cgpu_info *gpu);
int gpu_pstate(struct cgpu_info *gpu);
int gpu_busid(struct cgpu_info *gpu);

/* pid/vid, sn and bios rev */
int gpu_info(struct cgpu_info *gpu);
int gpu_vendor(uint8_t pci_bus_id, char *vendorname);


/* nvapi functions */
#ifdef WIN32
int nvapi_init();
#endif

#endif /* USE_WRAPNVML */
