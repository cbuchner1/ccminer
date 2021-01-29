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
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(string);
}

#define NVAPI_ID_PERF_INFO 0x409D9841
NvAPI_Status NvAPI_DLL_PerfPoliciesGetInfo(NvPhysicalGpuHandle handle, NVAPI_GPU_PERF_INFO* pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_INFO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_INFO*))nvidia_handle->query(NVAPI_ID_PERF_INFO);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_PERF_STATS 0x3D358A0C
NvAPI_Status NvAPI_DLL_PerfPoliciesGetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_PERF_STATUS* pStatus) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_PERF_STATUS*))nvidia_handle->query(NVAPI_ID_PERF_STATS);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pStatus);
}

#define NVAPI_ID_POWER_INFO 0x34206D86
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetInfo(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_INFO* pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_INFO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_INFO*))nvidia_handle->query(NVAPI_ID_POWER_INFO);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_POWERPOL_GET 0x70916171
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_STATUS* pPolicies) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*))nvidia_handle->query(NVAPI_ID_POWERPOL_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pPolicies);
}

#define NVAPI_ID_POWERPOL_SET 0xAD95F5ED
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesSetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_STATUS* pPolicies) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*))nvidia_handle->query(NVAPI_ID_POWERPOL_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pPolicies);
}

#define NVAPI_ID_POWERTOPO_GET 0xEDCF624E
NvAPI_Status NvAPI_DLL_ClientPowerTopologyGetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_TOPO* topo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_TOPO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_TOPO*))nvidia_handle->query(NVAPI_ID_POWERTOPO_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, topo);
}

#define NVAPI_ID_THERMAL_INFO 0x0D258BB5
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesGetInfo(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_INFO* pInfo) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_INFO*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_INFO*))nvidia_handle->query(NVAPI_ID_THERMAL_INFO);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_TLIMIT_GET 0xE9C425A1
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesGetLimit(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_LIMIT* pLimit) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*))nvidia_handle->query(NVAPI_ID_TLIMIT_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pLimit);
}

#define NVAPI_ID_TLIMIT_SET 0x34C0B13D
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesSetLimit(NvPhysicalGpuHandle handle, NVAPI_GPU_THERMAL_LIMIT* pLimit) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*))nvidia_handle->query(NVAPI_ID_TLIMIT_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pLimit);
}

#define NVAPI_ID_SERIALNUM_GET 0x14B83A5F
NvAPI_Status NvAPI_DLL_GetSerialNumber(NvPhysicalGpuHandle handle, NvAPI_ShortString serial) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NvAPI_ShortString) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NvAPI_ShortString))nvidia_handle->query(NVAPI_ID_SERIALNUM_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, serial);
}

#define NVAPI_ID_VOLTAGE_GET 0x465F9BCF
NvAPI_Status NvAPI_DLL_GetCurrentVoltage(NvPhysicalGpuHandle handle, NVAPI_VOLTAGE_STATUS* status) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_VOLTAGE_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLTAGE_STATUS*))nvidia_handle->query(NVAPI_ID_VOLTAGE_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, status);
}

#define NVAPI_ID_VOLT_STATUS_GET 0xC16C7E2C // Maxwell
NvAPI_Status NvAPI_DLL_GetVoltageDomainsStatus(NvPhysicalGpuHandle handle, NVAPI_VOLT_STATUS* data) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*))nvidia_handle->query(NVAPI_ID_VOLT_STATUS_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, data);
}

#define NVAPI_ID_VOLTAGE 0x28766157 // Maxwell 1-008c Real func name is unknown
NvAPI_Status NvAPI_DLL_GetVoltageStep(NvPhysicalGpuHandle handle, NVAPI_VOLT_STATUS* data) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*))nvidia_handle->query(NVAPI_ID_VOLTAGE);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, data);
}

#define NVAPI_ID_CLK_RANGE_GET 0x64B43A6A // Pascal
NvAPI_Status NvAPI_DLL_GetClockBoostRanges(NvPhysicalGpuHandle handle, NVAPI_CLOCKS_RANGE* range) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_CLOCKS_RANGE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCKS_RANGE*))nvidia_handle->query(NVAPI_ID_CLK_RANGE_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, range);
}

#define NVAPI_ID_CLK_BOOST_MASK 0x507B4B59 // Pascal
NvAPI_Status NvAPI_DLL_GetClockBoostMask(NvPhysicalGpuHandle handle, NVAPI_CLOCK_MASKS* range) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_CLOCK_MASKS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCK_MASKS*))nvidia_handle->query(NVAPI_ID_CLK_BOOST_MASK);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, range);
}

#define NVAPI_ID_CLK_BOOST_TABLE_GET 0x23F1B133 // Pascal
NvAPI_Status NvAPI_DLL_GetClockBoostTable(NvPhysicalGpuHandle handle, NVAPI_CLOCK_TABLE* table) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_CLOCK_TABLE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCK_TABLE*))nvidia_handle->query(NVAPI_ID_CLK_BOOST_TABLE_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, table);
}

#define NVAPI_ID_CLK_BOOST_TABLE_SET 0x0733E009 // Pascal
NvAPI_Status NvAPI_DLL_SetClockBoostTable(NvPhysicalGpuHandle handle, NVAPI_CLOCK_TABLE* table) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_CLOCK_TABLE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_CLOCK_TABLE*))nvidia_handle->query(NVAPI_ID_CLK_BOOST_TABLE_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, table);
}

#define NVAPI_ID_VFP_CURVE_GET 0x21537AD4 // Pascal
NvAPI_Status NvAPI_DLL_GetVFPCurve(NvPhysicalGpuHandle handle, NVAPI_VFP_CURVE* curve) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_VFP_CURVE*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VFP_CURVE*))nvidia_handle->query(NVAPI_ID_VFP_CURVE_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, curve);
}

#define NVAPI_ID_CURVE_GET 0xE440B867 // Pascal 2-030c struct 0C 03 02 00 00 00 00 00 01 00 00 00 06 00 00 00
#define NVAPI_ID_CURVE_SET 0x39442CFB // Pascal 2-030c struct 0C 03 02 00 00 00 00 00 01 00 00 00 06 00 00 00

#define NVAPI_ID_VOLTBOOST_GET 0x9DF23CA1 // Pascal 1-0028
NvAPI_Status NvAPI_DLL_GetCoreVoltageBoostPercent(NvPhysicalGpuHandle handle, NVAPI_VOLTBOOST_PERCENT* boost) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_VOLTBOOST_PERCENT*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLTBOOST_PERCENT*))nvidia_handle->query(NVAPI_ID_VOLTBOOST_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, boost);
}
#define NVAPI_ID_VOLTBOOST_SET 0xB9306D9B // Pascal 1-0028
NvAPI_Status NvAPI_DLL_SetCoreVoltageBoostPercent(NvPhysicalGpuHandle handle, NVAPI_VOLTBOOST_PERCENT* boost) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle,  NVAPI_VOLTBOOST_PERCENT*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_VOLTBOOST_PERCENT*))nvidia_handle->query(NVAPI_ID_VOLTBOOST_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, boost);
}

#define NVAPI_ID_PERFCLOCKS_GET 0x1EA54A3B
NvAPI_Status NvAPI_DLL_GetPerfClocks(NvPhysicalGpuHandle handle, uint32_t num, NVAPI_GPU_PERF_CLOCKS* pClocks) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, uint32_t, NVAPI_GPU_PERF_CLOCKS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, uint32_t, NVAPI_GPU_PERF_CLOCKS*))nvidia_handle->query(NVAPI_ID_PERFCLOCKS_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, num, pClocks);
}

#define NVAPI_ID_PERFCLOCKS_SET 0x07BCF4AC // error
NvAPI_Status NvAPI_DLL_SetPerfClocks(NvPhysicalGpuHandle handle, uint32_t num, NVAPI_GPU_PERF_CLOCKS* pClocks) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, uint32_t, NVAPI_GPU_PERF_CLOCKS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, uint32_t, NVAPI_GPU_PERF_CLOCKS*))nvidia_handle->query(NVAPI_ID_PERFCLOCKS_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, num, pClocks);
}

#define NVAPI_ID_PSTATELIMITS_GET 0x88C82104 // wrong prototype or missing struct data ?
NvAPI_Status NvAPI_DLL_GetPstateClientLimits(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATE_ID pst, uint32_t* pLimits) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATE_ID, uint32_t*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATE_ID, uint32_t*))nvidia_handle->query(NVAPI_ID_PSTATELIMITS_GET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pst, pLimits);
}

#define NVAPI_ID_PSTATELIMITS_SET 0xFDFC7D49 // wrong prototype or missing struct data ?
NvAPI_Status NvAPI_DLL_SetPstateClientLimits(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATE_ID pst, uint32_t* pLimits) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATE_ID, uint32_t*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATE_ID, uint32_t*))nvidia_handle->query(NVAPI_ID_PSTATELIMITS_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pst, pLimits);
}

#define NVAPI_ID_PSTATE20_SET 0x0F4DAE6B
// allow to set gpu/mem core freq delta
NvAPI_Status NvAPI_DLL_SetPstates20v1(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V1 *pSet) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V1*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V1*))nvidia_handle->query(NVAPI_ID_PSTATE20_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pSet);
}

// allow to set gpu core voltage delta
NvAPI_Status NvAPI_DLL_SetPstates20v2(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V2 *pSet) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V2*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATES20_INFO_V2*))nvidia_handle->query(NVAPI_ID_PSTATE20_SET);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
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
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pInfo);
}

#define NVAPI_ID_COOLERSETTINGS 0xDA141340 // 4-0558
NvAPI_Status NvAPI_DLL_GetCoolerSettings(NvPhysicalGpuHandle handle, uint32_t id, NVAPI_COOLER_SETTINGS* pSettings) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, uint32_t, NVAPI_COOLER_SETTINGS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, uint32_t, NVAPI_COOLER_SETTINGS*))nvidia_handle->query(NVAPI_ID_COOLERSETTINGS);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, id, pSettings);
}

#define NVAPI_ID_COOLER_SETLEVELS 0x891FA0AE // 1-00A4
NvAPI_Status NvAPI_DLL_SetCoolerLevels(NvPhysicalGpuHandle handle, uint32_t id, NVAPI_COOLER_LEVEL* pLevel) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, uint32_t, NVAPI_COOLER_LEVEL*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, uint32_t, NVAPI_COOLER_LEVEL*))nvidia_handle->query(NVAPI_ID_COOLER_SETLEVELS);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, id, pLevel);
}

#define NVAPI_ID_COOLER_RESTORE 0x8F6ED0FB
NvAPI_Status NvAPI_DLL_RestoreCoolerSettings(NvPhysicalGpuHandle handle, NVAPI_COOLER_SETTINGS* pSettings, uint32_t id) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_COOLER_SETTINGS*, uint32_t) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_COOLER_SETTINGS*, uint32_t))nvidia_handle->query(NVAPI_ID_COOLER_RESTORE);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, pSettings, id);
}

#define NVAPI_ID_I2CREADEX 0x4D7B0709 // 3-002c
NvAPI_Status NvAPI_DLL_I2CReadEx(NvPhysicalGpuHandle handle, NV_I2C_INFO_EX *i2c, NvU32 *exData) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_I2C_INFO_EX*, NvU32*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_I2C_INFO_EX*, NvU32*))nvidia_handle->query(NVAPI_ID_I2CREADEX);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, i2c, exData);
}

#define NVAPI_ID_I2CWRITEEX 0x283AC65A
NvAPI_Status NvAPI_DLL_I2CWriteEx(NvPhysicalGpuHandle handle, NV_I2C_INFO_EX *i2c, NvU32 *exData) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NV_I2C_INFO_EX*, NvU32 *exData) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NV_I2C_INFO_EX*, NvU32 *exData))nvidia_handle->query(NVAPI_ID_I2CWRITEEX);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)(handle, i2c, exData);
}

#define NVAPI_ID_UNLOAD 0xD22BDD7E
NvAPI_Status NvAPI_DLL_Unload() {
	static NvAPI_Status (*pointer)() = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)())nvidia_handle->query(NVAPI_ID_UNLOAD);
	}
	if(!pointer) return NVAPI_NO_IMPLEMENTATION;
	return (*pointer)();
}

#endif