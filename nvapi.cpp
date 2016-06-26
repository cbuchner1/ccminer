/**
 * Wrapper to nvapi.dll to query informations missing for x86 binaries (there is no nvml x86)
 * based on the work of https://github.com/ircubic/lib_gpu
 *
 * tpruvot@ccminer.org 06-2016
 */

#ifdef _WIN32

#include <windows.h>
#include <memory>
#include <stdexcept>

#include "compat/nvapi/nvapi_ccminer.h"

class NvAPILibraryHandle
{
	typedef void *(*QueryPtr)(uint32_t);

private:
	HMODULE library;
	QueryPtr nvidia_query;

public:
	NvAPILibraryHandle()
	{
		bool success = false;
#ifdef _WIN64
		library = LoadLibrary("nvapi64.dll");
#else
		library = LoadLibrary("nvapi.dll");
#endif
		if (library != NULL) {
			nvidia_query = reinterpret_cast<QueryPtr>(GetProcAddress(library, "nvapi_QueryInterface"));
			if (nvidia_query != NULL) {
				const uint32_t NVAPI_ID_INIT = 0x0150E828;
				auto init = static_cast<NvAPI_Status(*)()>(nvidia_query(NVAPI_ID_INIT));
				NvAPI_Status ret = init();
				success = (ret == NVAPI_OK);
			}
		}

		if (!success) {
			throw std::runtime_error("Unable to locate NVAPI library!");
		}
	}

	~NvAPILibraryHandle()
	{
		NvAPI_DLL_Unload();
		FreeLibrary(library);
	}

	void *query(uint32_t ID)
	{
		return nvidia_query(ID);
	}

};

static std::unique_ptr<NvAPILibraryHandle> nvidia_handle;
bool nvapi_dll_loaded = false;

NvAPI_Status nvapi_dll_init()
{
	try {
		if (!nvapi_dll_loaded) {
			nvidia_handle = std::make_unique<NvAPILibraryHandle>();
			nvapi_dll_loaded = true;
		}
	}
	catch (std::runtime_error) {
		nvapi_dll_loaded = false;
		return NVAPI_ERROR;
	}

	return NVAPI_OK;
}

// Hidden nvapi.dll functions

#define NVAPI_ID_IFVERSION 0x01053FA5
NvAPI_Status NvAPI_DLL_GetInterfaceVersionString(NvAPI_ShortString string) {
	static NvAPI_Status (*pointer)(NvAPI_ShortString string) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvAPI_ShortString))nvidia_handle->query(NVAPI_ID_IFVERSION);
	}
	return (*pointer)(string);
}

#define NVAPI_ID_PERF_INFO 0x409D9841
NvAPI_Status NvAPI_DLL_PerfPoliciesGetInfo(NvPhysicalGpuHandle handle, NVAPI_GPU_PERF_INFO* pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_INFO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_INFO*))nvidia_handle->query(NVAPI_ID_PERF_INFO);
	}
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_PERF_STATS 0x3D358A0C
NvAPI_Status NvAPI_DLL_PerfPoliciesGetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_PERF_STATUS* pStatus) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_STATUS*))nvidia_handle->query(NVAPI_ID_PERF_STATS);
	}
	return (*pointer)(handle, pStatus);
}

#define NVAPI_ID_POWER_INFO 0x34206D86
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetInfo(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_INFO* pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_INFO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_INFO*))nvidia_handle->query(NVAPI_ID_POWER_INFO);
	}
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_POWERPOL_GET 0x70916171
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_STATUS* pPolicies) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*))nvidia_handle->query(NVAPI_ID_POWERPOL_GET);
	}
	return (*pointer)(handle, pPolicies);
}

#define NVAPI_ID_POWERPOL_SET 0xAD95F5ED
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesSetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_STATUS* pPolicies) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*))nvidia_handle->query(NVAPI_ID_POWERPOL_SET);
	}
	return (*pointer)(handle, pPolicies);
}

#define NVAPI_ID_POWERTOPO_GET 0xEDCF624E
NvAPI_Status NvAPI_DLL_ClientPowerTopologyGetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_TOPO* topo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_TOPO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_TOPO*))nvidia_handle->query(NVAPI_ID_POWERTOPO_GET);
	}
	return (*pointer)(handle, topo);
}

#define NVAPI_ID_THERMAL_INFO 0x0D258BB5
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesGetInfo(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_INFO* pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_INFO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_INFO*))nvidia_handle->query(NVAPI_ID_THERMAL_INFO);
	}
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_TLIMIT_GET 0xE9C425A1
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesGetLimit(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_LIMIT* pLimit) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*))nvidia_handle->query(NVAPI_ID_TLIMIT_GET);
	}
	return (*pointer)(handle, pLimit);
}

#define NVAPI_ID_TLIMIT_SET 0x34C0B13D
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesSetLimit(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_LIMIT* pLimit) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*))nvidia_handle->query(NVAPI_ID_TLIMIT_SET);
	}
	return (*pointer)(handle, pLimit);
}

#define NVAPI_ID_SERIALNUM_GET 0x14B83A5F
NvAPI_Status NvAPI_DLL_GetSerialNumber(NvPhysicalGpuHandle handle, NvAPI_ShortString serial) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NvAPI_ShortString) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NvAPI_ShortString))nvidia_handle->query(NVAPI_ID_SERIALNUM_GET);
	}
	return (*pointer)(handle, serial);
}

#define NVAPI_ID_VOLTAGE_GET 0x465F9BCF
NvAPI_Status NvAPI_DLL_GetCurrentVoltage(NvPhysicalGpuHandle handle, NVAPI_VOLTAGE_STATUS* status) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_VOLTAGE_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLTAGE_STATUS*))nvidia_handle->query(NVAPI_ID_VOLTAGE_GET);
	}
	return (*pointer)(handle, status);
}

#define NVAPI_ID_VOLT_STATUS_GET 0xC16C7E2C
NvAPI_Status NvAPI_DLL_GetVoltageDomainsStatus(NvPhysicalGpuHandle handle, NVAPI_VOLT_STATUS* data) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_VOLT_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*))nvidia_handle->query(NVAPI_ID_VOLT_STATUS_GET);
	}
	return (*pointer)(handle, data);
}

#define NVAPI_ID_CLK_RANGE_GET 0x64B43A6A // Pascal
NvAPI_Status NvAPI_DLL_GetClockBoostRanges(NvPhysicalGpuHandle handle, NVAPI_CLOCKS_RANGE* range) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_CLOCKS_RANGE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCKS_RANGE*))nvidia_handle->query(NVAPI_ID_CLK_RANGE_GET);
	}
	return (*pointer)(handle, range);
}

#define NVAPI_ID_CLK_BOOST_MASK 0x507B4B59 // Pascal
NvAPI_Status NvAPI_DLL_GetClockBoostMask(NvPhysicalGpuHandle handle, NVAPI_CLOCK_MASKS* range) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_CLOCK_MASKS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCK_MASKS*))nvidia_handle->query(NVAPI_ID_CLK_BOOST_MASK);
	}
	return (*pointer)(handle, range);
}

#define NVAPI_ID_CLK_BOOST_TABLE_GET 0x23F1B133 // Pascal
NvAPI_Status NvAPI_DLL_GetClockBoostTable(NvPhysicalGpuHandle handle, NVAPI_CLOCK_TABLE* table) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_CLOCK_TABLE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCK_TABLE*))nvidia_handle->query(NVAPI_ID_CLK_BOOST_TABLE_GET);
	}
	return (*pointer)(handle, table);
}

#define NVAPI_ID_CLK_BOOST_TABLE_SET 0x0733E009 // Pascal
NvAPI_Status NvAPI_DLL_SetClockBoostTable(NvPhysicalGpuHandle handle, NVAPI_CLOCK_TABLE* table) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_CLOCK_TABLE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCK_TABLE*))nvidia_handle->query(NVAPI_ID_CLK_BOOST_TABLE_SET);
	}
	return (*pointer)(handle, table);
}

#define NVAPI_ID_VFP_CURVE_GET 0x21537AD4 // Pascal 39442CFB to check also, Set ?
NvAPI_Status NvAPI_DLL_GetVFPCurve(NvPhysicalGpuHandle handle, NVAPI_VFP_CURVE* curve) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_VFP_CURVE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VFP_CURVE*))nvidia_handle->query(NVAPI_ID_VFP_CURVE_GET);
	}
	return (*pointer)(handle, curve);
}

#define NVAPI_ID_PERFCLOCKS_GET 0x1EA54A3B
NvAPI_Status NvAPI_DLL_GetPerfClocks(NvPhysicalGpuHandle handle, void* pFreqs){
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, void*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, void*))nvidia_handle->query(NVAPI_ID_PERFCLOCKS_GET);
	}
	return (*pointer)(handle, pFreqs);
}

#define NVAPI_ID_PSTATE20_SET 0x0F4DAE6B
// allow to set gpu/mem core freq delta
NvAPI_Status NvAPI_DLL_SetPstates20v1(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V1 *pSet) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V1*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V1*))nvidia_handle->query(NVAPI_ID_PSTATE20_SET);
	}
	return (*pointer)(handle, pSet);
}

// allow to set gpu core voltage delta
NvAPI_Status NvAPI_DLL_SetPstates20v2(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V2 *pSet) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V2*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V2*))nvidia_handle->query(NVAPI_ID_PSTATE20_SET);
	}
	return (*pointer)(handle, pSet);
}

// maxwell voltage table
#define NVAPI_ID_VOLTAGES 0x7D656244 // 1-40cc
NvAPI_Status NvAPI_DLL_GetVoltages(NvPhysicalGpuHandle handle, NVAPI_VOLTAGES_TABLE *pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_VOLTAGES_TABLE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLTAGES_TABLE*))nvidia_handle->query(NVAPI_ID_VOLTAGES);
	}
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_UNLOAD 0xD22BDD7E
NvAPI_Status NvAPI_DLL_Unload() {
	static NvAPI_Status (*pointer)() = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)())nvidia_handle->query(NVAPI_ID_UNLOAD);
	}
	return (*pointer)();
}

#endif