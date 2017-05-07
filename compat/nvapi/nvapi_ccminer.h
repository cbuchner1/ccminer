#pragma once

#define NVAPI_INTERNAL
#include "nvapi.h"

NvAPI_Status nvapi_dll_init();

typedef struct {
	NvU32 version;
	NvU8  valid;
	NvU8  count;
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
#define NVAPI_GPU_POWER_INFO_VER MAKE_NVAPI_VERSION(NVAPI_GPU_POWER_INFO, 1)

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

typedef struct {
	NvU32 version;
	NvU32 count;
	struct {
		NvU32 unknown1;
		NvU32 unknown2;
		NvU32 power; // unsure ?? 85536 to 95055 on 1080, 104825+ on 970
		NvU32 unknown4;
	} entries[4];
} NVAPI_GPU_POWER_TOPO;
#define NVAPI_GPU_POWER_TOPO_VER MAKE_NVAPI_VERSION(NVAPI_GPU_POWER_TOPO, 1)

typedef struct {
	NvU32 version;
	NvU32 flags;
	struct {
		NvU32 controller;
		NvU32 unknown;
		NvS32 min_temp;
		NvS32 def_temp;
		NvS32 max_temp;
		NvU32 defaultFlags;
	} entries[4];
} NVAPI_GPU_THERMAL_INFO;
#define NVAPI_GPU_THERMAL_INFO_VER MAKE_NVAPI_VERSION(NVAPI_GPU_THERMAL_INFO, 2)

typedef struct {
	NvU32 version;
	NvU32 flags;
	struct {
		NvU32 controller;
		NvU32 value;
		NvU32 flags;
	} entries[4];
} NVAPI_GPU_THERMAL_LIMIT;
#define NVAPI_GPU_THERMAL_LIMIT_VER MAKE_NVAPI_VERSION(NVAPI_GPU_THERMAL_LIMIT, 2)

// Maxwell gpu core voltage reading
typedef struct {
	NvU32 version;
	NvU32 flags;
	NvU32 count; // unsure
	NvU32 unknown;
	NvU32 value_uV;
	NvU32 buf1[30];
} NVAPI_VOLT_STATUS; // 140 bytes (1-008c)
#define NVAPI_VOLT_STATUS_VER MAKE_NVAPI_VERSION(NVAPI_VOLT_STATUS, 1)

// Pascal gpu core voltage reading
typedef struct {
	NvU32 version;
	NvU32 flags;
	NvU32 nul[8];
	NvU32 value_uV;
	NvU32 buf1[8];
} NVAPI_VOLTAGE_STATUS; // 76 bytes (1-004c)
#define NVAPI_VOLTAGE_STATUS_VER MAKE_NVAPI_VERSION(NVAPI_VOLTAGE_STATUS, 1)

typedef struct {
	NvU32 version;
	NvU32 numClocks; // unsure
	NvU32 nul[8];
	struct {
		NvU32 a;
		NvU32 clockType;
		NvU32 c;
		NvU32 d;
		NvU32 e;
		NvU32 f;
		NvU32 g;
		NvU32 h;
		NvU32 i;
		NvU32 j;
		NvS32 rangeMax;
		NvS32 rangeMin;
		NvS32 tempMax; // ? unsure
		NvU32 n;
		NvU32 o;
		NvU32 p;
		NvU32 q;
		NvU32 r;
	} entries[32]; // NVAPI_MAX_GPU_CLOCKS ?
} NVAPI_CLOCKS_RANGE; // 2344 bytes
#define NVAPI_CLOCKS_RANGE_VER MAKE_NVAPI_VERSION(NVAPI_CLOCKS_RANGE, 1)

// seems to return a clock table mask
typedef struct {
	NvU32 version;
	NvU32 mask[4]; // 80 bits mask
	NvU32 buf0[8];
	struct {
		NvU32 a;
		NvU32 b;
		NvU32 c;
		NvU32 d;
		NvU32 memDelta; // 1 for mem
		NvU32 gpuDelta; // 1 for gpu
	} clocks[80 + 23];
	NvU32 buf1[916];
} NVAPI_CLOCK_MASKS; // 6188 bytes
#define NVAPI_CLOCK_MASKS_VER MAKE_NVAPI_VERSION(NVAPI_CLOCK_MASKS, 1)

// contains the gpu/mem clocks deltas
typedef struct {
	NvU32 version;
	NvU32 mask[4]; // 80 bits mask (could be 8x 32bits)
	NvU32 buf0[12];
	struct {
		NvU32 a;
		NvU32 b;
		NvU32 c;
		NvU32 d;
		NvU32 e;
		NvS32 freqDelta; // 84000 = +84MHz
		NvU32 g;
		NvU32 h;
		NvU32 i;
	} gpuDeltas[80];
	NvU32 memFilled[23]; // maybe only 4 max
	NvS32 memDeltas[23];
	NvU32 buf1[1529];
} NVAPI_CLOCK_TABLE; // 9248 bytes
#define NVAPI_CLOCK_TABLE_VER MAKE_NVAPI_VERSION(NVAPI_CLOCK_TABLE, 1)

typedef struct {
	NvU32 version;
	NvU32 mask[4]; // 80 bits mask
	NvU32 buf0[12];
	struct {
		NvU32 a; // 0
		NvU32 freq_kHz;
		NvU32 volt_uV;
		NvU32 d;
		NvU32 e;
		NvU32 f;
		NvU32 g;
	} gpuEntries[80];
	struct {
		NvU32 a;  // 1 for idle values ?
		NvU32 freq_kHz;
		NvU32 volt_uV;
		NvU32 d;
		NvU32 e;
		NvU32 f;
		NvU32 g;
	} memEntries[23];
	NvU32 buf1[1064];
} NVAPI_VFP_CURVE; // 7208 bytes (1-1c28)
#define NVAPI_VFP_CURVE_VER MAKE_NVAPI_VERSION(NVAPI_VFP_CURVE, 1)

typedef struct {
	NvU32 version;
	NvS32 percent;
	NvU32 pad[8];
} NVAPI_VOLTBOOST_PERCENT; // 40 bytes (1-0028)
#define NVAPI_VOLTBOOST_PERCENT_VER MAKE_NVAPI_VERSION(NVAPI_VOLTBOOST_PERCENT, 1)

typedef struct {
	NvU32 version;
	NvU32 flags;
	NvU32 filled; // 1
	struct {
		NvU32 volt_uV;
		NvU32 unknown;
	} entries[128];
	// some empty tables then...
	NvU32 buf1[3888];
} NVAPI_VOLTAGES_TABLE; // 16588 bytes (1-40cc)
#define NVAPI_VOLTAGES_TABLE_VER MAKE_NVAPI_VERSION(NVAPI_VOLTAGES_TABLE, 1)

typedef struct {
	NvU32 version;
	NvU32 val1; // 7
	NvU32 val2; // 0x3F (63.)
	NvU32 pad[16];
} NVAPI_GPU_PERF_INFO; // 76 bytes (1-004c)
#define NVAPI_GPU_PERF_INFO_VER MAKE_NVAPI_VERSION(NVAPI_GPU_PERF_INFO, 1)

typedef struct {
	NvU32 version;
	NvU32 flags;     // 0
	NvU64 timeRef;   // increment with time
	NvU64 val1;      // seen 1 4 5 while mining, 16 else
	NvU64 val2;      // seen 7 and 3
	NvU64 values[3]; // increment with time
	NvU32 pad[326];  // empty
} NVAPI_GPU_PERF_STATUS; // 1360 bytes (1-0550)
#define NVAPI_GPU_PERF_STATUS_VER MAKE_NVAPI_VERSION(NVAPI_GPU_PERF_STATUS, 1)

typedef struct {
	NvU32 version;
	NvU32 val1;      // 4
	NvU32 val2;      // 2 or 0
	NvU32 val3;      // 2
	NvU32 val4;      // 3
	NV_GPU_PERF_PSTATE_ID pStateId;
	NvU32 val6;      // 0 or 2
	NvU32 val7;      // 4
	NvU32 val8;      // 0
	NvU32 memFreq1;  // 405000.
	NvU32 memFreq2;  // 405000.
	NvU32 memFreqMin;// 101250.
	NvU32 memFreqMax;// 486000.
	NvU32 zeros[3];
	NvU32 gpuFreq1;  // 696000. Unsure about those
	NvU32 gpuFreq2;  // 696000.
	NvU32 gpuFreqMin;// 174000.
	NvU32 gpuFreqMax;// 658000.
	NvU32 pad[2697];
} NVAPI_GPU_PERF_CLOCKS; // 10868 bytes (2-2a74)
#define NVAPI_GPU_PERF_CLOCKS_VER MAKE_NVAPI_VERSION(NVAPI_GPU_PERF_CLOCKS, 2)

typedef struct {
	NvU32 version;
	NvU32 level;
	NvU32 count;
	NvU32 pad[339]; // (4-0558)
} NVAPI_COOLER_SETTINGS;
#define NVAPI_COOLER_SETTINGS_VER MAKE_NVAPI_VERSION(NVAPI_COOLER_SETTINGS, 4)

typedef struct {
	NvU32 version;
	NvU32 level;   // 0 = auto ?
	NvU32 count;   // 1
	NvU32 pad[38]; // (1-00a4)
} NVAPI_COOLER_LEVEL;
#define NVAPI_COOLER_LEVEL_VER MAKE_NVAPI_VERSION(NVAPI_COOLER_LEVEL, 1)

NvAPI_Status NvAPI_DLL_GetInterfaceVersionString(NvAPI_ShortString string);

NvAPI_Status NvAPI_DLL_PerfPoliciesGetInfo(NvPhysicalGpuHandle, NVAPI_GPU_PERF_INFO*); // 409D9841 1-004c
NvAPI_Status NvAPI_DLL_PerfPoliciesGetStatus(NvPhysicalGpuHandle, NVAPI_GPU_PERF_STATUS*); // 3D358A0C 1-0550

NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetInfo(NvPhysicalGpuHandle, NVAPI_GPU_POWER_INFO*);
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesGetStatus(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*);
NvAPI_Status NvAPI_DLL_ClientPowerPoliciesSetStatus(NvPhysicalGpuHandle, NVAPI_GPU_POWER_STATUS*);
NvAPI_Status NvAPI_DLL_ClientPowerTopologyGetStatus(NvPhysicalGpuHandle, NVAPI_GPU_POWER_TOPO*); // EDCF624E 1-0048

NvAPI_Status NvAPI_DLL_ClientThermalPoliciesGetInfo(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_INFO*);
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesGetLimit(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*);
NvAPI_Status NvAPI_DLL_ClientThermalPoliciesSetLimit(NvPhysicalGpuHandle, NVAPI_GPU_THERMAL_LIMIT*);

// Pascal GTX only
NvAPI_Status NvAPI_DLL_GetClockBoostRanges(NvPhysicalGpuHandle, NVAPI_CLOCKS_RANGE*);
NvAPI_Status NvAPI_DLL_GetClockBoostMask(NvPhysicalGpuHandle, NVAPI_CLOCK_MASKS*);  // 0x507B4B59
NvAPI_Status NvAPI_DLL_GetClockBoostTable(NvPhysicalGpuHandle, NVAPI_CLOCK_TABLE*); // 0x23F1B133
NvAPI_Status NvAPI_DLL_SetClockBoostTable(NvPhysicalGpuHandle, NVAPI_CLOCK_TABLE*); // 0x0733E009
NvAPI_Status NvAPI_DLL_GetVFPCurve(NvPhysicalGpuHandle, NVAPI_VFP_CURVE*); // 0x21537AD4
NvAPI_Status NvAPI_DLL_GetCurrentVoltage(NvPhysicalGpuHandle, NVAPI_VOLTAGE_STATUS*);   // 0x465F9BCF 1-004c
NvAPI_Status NvAPI_DLL_GetCoreVoltageBoostPercent(NvPhysicalGpuHandle, NVAPI_VOLTBOOST_PERCENT*);
NvAPI_Status NvAPI_DLL_SetCoreVoltageBoostPercent(NvPhysicalGpuHandle, NVAPI_VOLTBOOST_PERCENT*);

// Maxwell only
NvAPI_Status NvAPI_DLL_GetVoltageDomainsStatus(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*); // 0xC16C7E2C
NvAPI_Status NvAPI_DLL_GetVoltages(NvPhysicalGpuHandle, NVAPI_VOLTAGES_TABLE*); // 0x7D656244 1-40CC
NvAPI_Status NvAPI_DLL_GetVoltageStep(NvPhysicalGpuHandle, NVAPI_VOLT_STATUS*); // 0x28766157 1-008C unsure of the name

NvAPI_Status NvAPI_DLL_GetCoolerSettings(NvPhysicalGpuHandle, uint32_t, NVAPI_COOLER_SETTINGS*); // 0xDA141340 4-0558
NvAPI_Status NvAPI_DLL_SetCoolerLevels(NvPhysicalGpuHandle, uint32_t, NVAPI_COOLER_LEVEL*); // 0x891FA0AE 1-00A4
NvAPI_Status NvAPI_DLL_RestoreCoolerSettings(NvPhysicalGpuHandle, NVAPI_COOLER_SETTINGS*, uint32_t);

NvAPI_Status NvAPI_DLL_GetSerialNumber(NvPhysicalGpuHandle, NvAPI_ShortString serial);

NvAPI_Status NvAPI_DLL_GetPerfClocks(NvPhysicalGpuHandle, uint32_t num, NVAPI_GPU_PERF_CLOCKS* pClocks); // 2-2A74
//NvAPI_Status NvAPI_DLL_SetPerfClocks(NvPhysicalGpuHandle, uint32_t num, NVAPI_GPU_PERF_CLOCKS* pClocks); // error

//invalid..
//NvAPI_Status NvAPI_DLL_GetPstateClientLimits(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATE_ID, uint32_t* pLimits);
//NvAPI_Status NvAPI_DLL_SetPstateClientLimits(NvPhysicalGpuHandle, NV_GPU_PERF_PSTATE_ID, uint32_t* pLimits);

NvAPI_Status NvAPI_DLL_SetPstates20v1(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V1 *pSet);
NvAPI_Status NvAPI_DLL_SetPstates20v2(NvPhysicalGpuHandle handle, NV_GPU_PERF_PSTATES20_INFO_V2 *pSet);

NvAPI_Status NvAPI_DLL_Unload();

#define NV_ASSERT(x) { NvAPI_Status ret = x; if(ret != NVAPI_OK) return ret; }

// to reduce stack size, allow to reuse a mem buffer
#define NV_INIT_STRUCT_ON(TYPE, var, mem) { \
	var = (TYPE*) mem; \
	memset(var, 0, sizeof(TYPE)); \
	var->version = TYPE##_VER; \
}

// alloc a struct, need free(var)
#define NV_INIT_STRUCT_ALLOC(TYPE, var) { \
	var = (TYPE*) calloc(1, TYPE##_VER & 0xFFFF); \
	if (var) var->version = TYPE##_VER; \
}

//! Used in NvAPI_I2CReadEx()
typedef struct
{
	NvU32        version;
	NvU32        displayMask;        // Display Mask of the concerned display.
	NvU8         bIsDDCPort;         // indicates either the DDC port (TRUE) or the communication port (FALSE) of the concerned display.
	NvU8         i2cDevAddress;      // address of the I2C slave.  The address should be shifted left by one. 0x50 -> 0xA0.
	NvU8*        pbI2cRegAddress;    // I2C target register address.  May be NULL, which indicates no register address should be sent.
	NvU32        regAddrSize;        // size in bytes of target register address.  If pbI2cRegAddress is NULL, this field must be 0.
	NvU8*        pbData;             // buffer of data which is to be read or written (depending on the command).
	NvU32        cbRead;             // bytes to read ??? seems required on write too
	NvU32        cbSize;             // full size of the data buffer, pbData, to be read or written.
	NV_I2C_SPEED i2cSpeedKhz;        // target speed of the transaction in (kHz) (Chosen from the enum NV_I2C_SPEED).
	NvU8         portId;             // portid on which device is connected (remember to set bIsPortIdSet if this value is set)
	NvU32        bIsPortIdSet;       // set this flag on if and only if portid value is set

} NV_I2C_INFO_EX;
#define NV_I2C_INFO_EX_VER  MAKE_NVAPI_VERSION(NV_I2C_INFO_EX,3)
/*
sample evga x64 call (struct of 0x40 bytes)
ReadEx
$ ==> 40 00 03 00  00 00 00 00  00 40 00 00  00 00 00 00
$+10  58 F9 2B 00  00 00 00 00  01 00 00 00  00 00 00 00
$+20  C0 F9 2B 00  00 00 00 00  02 00 00 00  FF FF 00 00
$+30  00 00 00 00  02 00 00 00  01 00 00 00  00 00 00 00

$ ==> 40 00 03 00  00 00 00 00  00 10 00 00  00 00 00 00
$+10  68 F9 2B 00  00 00 00 00  01 00 00 00  00 00 00 00
$+20  C0 F9 2B 00  00 00 00 00  01 00 00 00  FF FF 00 00
$+30  00 00 00 00  01 00 00 00  01 00 00 00  00 00 00 00
00000000002BF968 > 75 83 CF 3F 01 00 00 00
00000000002BF9C0 > 0

WriteEx
$ ==> 40 00 03 00  00 00 00 00  00 8C 00 00  00 00 00 00
$+10  30 F9 2B 00  00 00 00 00  01 00 00 00  00 00 00 00
$+20  38 F9 2B 00  00 00 00 00  02 00 00 00  FF FF 00 00
$+30  00 00 00 00  01 00 00 00  01 00 00 00  00 00 00 00
00000000002BF930 > D1 00 00 00 00 00 00 00
00000000002BF938 > 38 00 00 00 00 00 00 00
*/

NvAPI_Status NvAPI_DLL_I2CReadEx(NvPhysicalGpuHandle, NV_I2C_INFO_EX*, NvU32*);
NvAPI_Status NvAPI_DLL_I2CWriteEx(NvPhysicalGpuHandle, NV_I2C_INFO_EX*, NvU32*);
