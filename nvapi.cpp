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
		library = LoadLibrary("nvapi.dll");
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

#define NVAPI_ID_POWERPOL_SET 0x0AD95F5ED
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesSetStatus(NvPhysicalGpuHandle handle, NVAPI_GPU_POWER_STATUS* pPolicies) {
	static NvAPI_Status (*pointer)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*) = NULL;
	if(!nvapi_dll_loaded) return NVAPI_API_NOT_INITIALIZED;
	if(!pointer) {
		pointer = (NvAPI_Status (*)(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*))nvidia_handle->query(NVAPI_ID_POWERPOL_SET);
	}
	return (*pointer)(handle, pPolicies);
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