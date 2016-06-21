#pragma once

#include "nvapi.h"

NvAPI_Status nvapi_dll_init();

typedef struct {
	NvU32 version;
	NvU32 flags;
	struct
	{
		NvU32 pstate; // Assumption
		NvU32 unknown1[2];
		NvU32 min_power;
		NvU32 unknown2[2];
		NvU32 def_power;
		NvU32 unknown3[2];
		NvU32 max_power;
		NvU32 unknown4; // 0
	} entries[4];
} NVAPI_GPU_POWER_INFO;

typedef struct {
	NvU32 version;
	NvU32 flags;
	struct {
		NvU32 unknown1;
		NvU32 unknown2;
		NvU32 power; // percent * 1000
		NvU32 unknown4;
	} entries[4];
} NVAPI_GPU_POWER_STATUS;

#define NVAPI_GPU_POWER_STATUS_VER MAKE_NVAPI_VERSION(NVAPI_GPU_POWER_STATUS, 1)
#define NVAPI_GPU_POWER_INFO_VER MAKE_NVAPI_VERSION(NVAPI_GPU_POWER_INFO, 1)

NvAPI_Status NvAPI_DLL_GetInterfaceVersionString(NvAPI_ShortString string);
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetInfo(NvPhysicalGpuHandle hPhysicalGpu, NVAPI_GPU_POWER_INFO* pInfo);
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetStatus(NvPhysicalGpuHandle hPhysicalGpu, NVAPI_GPU_POWER_STATUS* pPolicies);
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesSetStatus(NvPhysicalGpuHandle hPhysicalGpu, NVAPI_GPU_POWER_STATUS* pPolicies);

NvAPI_Status NvAPI_DLL_Unload();

#define NV_ASSERT(x) { NvAPI_Status ret = x; if(ret != NVAPI_OK) return ret; }
