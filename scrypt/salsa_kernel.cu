//
// Contains the autotuning logic and some utility functions.
// Note that all CUDA kernels have been moved to other .cu files
//

#include <stdio.h>
#include <map>
#include <algorithm>
#include <unistd.h> // usleep
#include <ctype.h> // tolower
#include "cuda_helper.h"

#include "salsa_kernel.h"

#include "nv_kernel2.h"
#include "titan_kernel.h"
#include "nv_kernel.h"
#include "kepler_kernel.h"
#include "fermi_kernel.h"
#include "test_kernel.h"

#include "miner.h"

#if defined(_WIN64) || defined(__x86_64__) || defined(__64BIT__)
#define MAXMEM 0x300000000ULL  // 12 GB (the largest Kepler)
#else
#define MAXMEM  0xFFFFFFFFULL  // nearly 4 GB (32 bit limitations)
#endif

// require CUDA 5.5 driver API
#define DMAJ 5
#define DMIN 5

// define some error checking macros
#define DELIMITER '/'
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#undef checkCudaErrors
#define checkCudaErrors(x) \
{ \
	cudaGetLastError(); \
	x; \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess && !abort_flag) \
		applog(LOG_ERR, "GPU #%d: Err %d: %s (%s:%d)", device_map[thr_id], err, cudaGetErrorString(err), __FILENAME__, __LINE__); \
}

// some globals containing pointers to device memory (for chunked allocation)
// [MAX_GPUS] indexes up to MAX_GPUS threads (0...MAX_GPUS-1)
int       MAXWARPS[MAX_GPUS];
uint32_t* h_V[MAX_GPUS][TOTAL_WARP_LIMIT*64];          // NOTE: the *64 prevents buffer overflow for --keccak
uint32_t  h_V_extra[MAX_GPUS][TOTAL_WARP_LIMIT*64];    //       with really large kernel launch configurations

KernelInterface *Best_Kernel_Heuristics(cudaDeviceProp *props)
{
	KernelInterface *kernel = NULL;
	uint64_t N = 1UL << (opt_nfactor+1);

	if (IS_SCRYPT() || (IS_SCRYPT_JANE() && N <= 8192))
	{
		// high register count kernels (scrypt, low N-factor scrypt-jane)
		if (props->major > 3 || (props->major == 3 && props->minor >= 5))
			kernel = new NV2Kernel(); // we don't want this for Keccak though
		else if (props->major == 3 && props->minor == 0)
			kernel = new NVKernel();
		else
			kernel = new FermiKernel();
	}
	else
	{
	   // high N-factor scrypt-jane = low registers count kernels
	   if (props->major > 3 || (props->major == 3 && props->minor >= 5))
			kernel = new TitanKernel();
		else if (props->major == 3 && props->minor == 0)
			kernel = new KeplerKernel();
		else
			kernel = new TestKernel();
	}
	return kernel;
}


bool validate_config(char *config, int &b, int &w, KernelInterface **kernel = NULL, cudaDeviceProp *props = NULL)
{
	bool success = false;
	char kernelid = ' ';
	if (config != NULL)
	{
		if (config[0] == 'T' || config[0] == 'K' || config[0] == 'F' || config[0] == 'L' ||
			config[0] == 't' || config[0] == 'k' || config[0] == 'f' ||
			config[0] == 'Z' || config[0] == 'Y' || config[0] == 'X') {
			kernelid = config[0];
			config++;
		}

		if (config[0] >= '0' && config[0] <= '9')
			if (sscanf(config, "%dx%d", &b, &w) == 2)
				success = true;

		if (success && kernel != NULL)
		{
			switch (kernelid)
			{
				case 'T': case 'Z': *kernel = new NV2Kernel(); break;
				case 't':           *kernel = new TitanKernel(); break;
				case 'K': case 'Y': *kernel = new NVKernel(); break;
				case 'k':           *kernel = new KeplerKernel(); break;
				case 'F': case 'L': *kernel = new FermiKernel(); break;
				case 'f': case 'X': *kernel = new TestKernel(); break;
				case ' ': // choose based on device architecture
					*kernel = Best_Kernel_Heuristics(props);
				break;
			}
		}
	}
	return success;
}

std::map<int, int> context_blocks;
std::map<int, int> context_wpb;
std::map<int, bool> context_concurrent;
std::map<int, KernelInterface *> context_kernel;
std::map<int, uint32_t *> context_idata[2];
std::map<int, uint32_t *> context_odata[2];
std::map<int, cudaStream_t> context_streams[2];
std::map<int, uint32_t *> context_X[2];
std::map<int, uint32_t *> context_H[2];
std::map<int, cudaEvent_t> context_serialize[2];

// for SHA256 hashing on GPU
std::map<int, uint32_t *> context_tstate[2];
std::map<int, uint32_t *> context_ostate[2];
std::map<int, uint32_t *> context_hash[2];

int find_optimal_blockcount(int thr_id, KernelInterface* &kernel, bool &concurrent, int &wpb);

int cuda_throughput(int thr_id)
{
	int GRID_BLOCKS, WARPS_PER_BLOCK;
	if (context_blocks.find(thr_id) == context_blocks.end())
	{
#if 0
		CUcontext ctx;
		cuCtxCreate( &ctx, CU_CTX_SCHED_YIELD, device_map[thr_id] );
		cuCtxSetCurrent(ctx);
#else
		checkCudaErrors(cudaSetDevice(device_map[thr_id]));
		checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleYield));
#endif

		KernelInterface *kernel;
		bool concurrent;
		GRID_BLOCKS = find_optimal_blockcount(thr_id, kernel, concurrent, WARPS_PER_BLOCK);

		if(GRID_BLOCKS == 0)
			return 0;

		unsigned int THREADS_PER_WU = kernel->threads_per_wu();
		unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;
		unsigned int state_size = WU_PER_LAUNCH * sizeof(uint32_t) * 8;

		// allocate device memory for scrypt_core inputs and outputs
		uint32_t *tmp;
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_idata[0][thr_id] = tmp;
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_idata[1][thr_id] = tmp;
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_odata[0][thr_id] = tmp;
		checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_odata[1][thr_id] = tmp;

		// allocate pinned host memory for scrypt hashes
		checkCudaErrors(cudaHostAlloc((void **) &tmp, state_size, cudaHostAllocDefault)); context_H[0][thr_id] = tmp;
		checkCudaErrors(cudaHostAlloc((void **) &tmp, state_size, cudaHostAllocDefault)); context_H[1][thr_id] = tmp;

		if (IS_SCRYPT())
		{
			if (parallel < 2)
			{
				// allocate pinned host memory for scrypt_core input/output
				checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[0][thr_id] = tmp;
				checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[1][thr_id] = tmp;
			}
			else
			{
				// allocate tstate, ostate, scrypt hash device memory
				checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_tstate[0][thr_id] = tmp;
				checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_tstate[1][thr_id] = tmp;
				checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_ostate[0][thr_id] = tmp;
				checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_ostate[1][thr_id] = tmp;
				checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[0][thr_id] = tmp;
				checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[1][thr_id] = tmp;
			}
		}
		else /* if (IS_SCRYPT_JANE()) */
		{
			// allocate pinned host memory for scrypt_core input/output
			checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[0][thr_id] = tmp;
			checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[1][thr_id] = tmp;

			checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[0][thr_id] = tmp;
			checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[1][thr_id] = tmp;
		}

		// create two CUDA streams
		cudaStream_t tmp2;
		checkCudaErrors( cudaStreamCreate(&tmp2) ); context_streams[0][thr_id] = tmp2;
		checkCudaErrors( cudaStreamCreate(&tmp2) ); context_streams[1][thr_id] = tmp2;

		// events used to serialize the kernel launches (we don't want any overlapping of kernels)
		cudaEvent_t tmp4;
		checkCudaErrors(cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); context_serialize[0][thr_id] = tmp4;
		checkCudaErrors(cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); context_serialize[1][thr_id] = tmp4;
		checkCudaErrors(cudaEventRecord(context_serialize[1][thr_id]));

		context_kernel[thr_id] = kernel;
		context_concurrent[thr_id] = concurrent;
		context_blocks[thr_id] = GRID_BLOCKS;
		context_wpb[thr_id] = WARPS_PER_BLOCK;
	}

	GRID_BLOCKS = context_blocks[thr_id];
	WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	return WU_PER_LAUNCH;
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x10, 8   }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11, 8   }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12, 8   }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13, 8   }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32  }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48  }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class - GK104 = 1536 cores / 8 SMs
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GTX750/750Ti
		{ 0x52, 128 }, // Maxwell Second Generation (SM 5.2) GTX980 = 2048 cores / 16 SMs - GTX970 1664 cores / 13 SMs
		{ -1, -1 },
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	applog(LOG_WARNING, "MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM", major, minor, 128);
	return 128;
}

#ifdef WIN32
#include <windows.h>
static int console_width() {
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	return csbi.srWindow.Right - csbi.srWindow.Left + 1;
}
#else
static inline int console_width() {
	return 999;
}
#endif

int find_optimal_blockcount(int thr_id, KernelInterface* &kernel, bool &concurrent, int &WARPS_PER_BLOCK)
{
	int cw = console_width();
	int optimal_blocks = 0;

	cudaDeviceProp props;
	checkCudaErrors(cudaGetDeviceProperties(&props, device_map[thr_id]));
	concurrent = (props.concurrentKernels > 0);

	WARPS_PER_BLOCK = -1;

	// if not specified, use interactive mode for devices that have the watchdog timer enabled
	if (device_interactive[thr_id] == -1)
		device_interactive[thr_id] = props.kernelExecTimeoutEnabled;

	// turn off texture cache if not otherwise specified
	if (device_texturecache[thr_id] == -1)
		device_texturecache[thr_id] = 0;

	// if not otherwise specified or required, turn single memory allocations off as they reduce
	// the amount of memory that we can allocate on Windows Vista, 7 and 8 (WDDM driver model issue)
	if (device_singlememory[thr_id] == -1) device_singlememory[thr_id] = 0;

	// figure out which kernel implementation to use
	if (!validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK, &kernel, &props)) {
		kernel = NULL;
		if (device_config[thr_id] != NULL) {
				 if (device_config[thr_id][0] == 'T' || device_config[thr_id][0] == 'Z')
				kernel = new NV2Kernel();
			else if (device_config[thr_id][0] == 't')
				kernel = new TitanKernel();
			else if (device_config[thr_id][0] == 'K' || device_config[thr_id][0] == 'Y')
				kernel = new NVKernel();
			else if (device_config[thr_id][0] == 'k')
				kernel = new KeplerKernel();
			else if (device_config[thr_id][0] == 'F' || device_config[thr_id][0] == 'L')
				kernel = new FermiKernel();
			else if (device_config[thr_id][0] == 'f' || device_config[thr_id][0] == 'X')
				kernel = new TestKernel();
		}
		if (kernel == NULL) kernel = Best_Kernel_Heuristics(&props);
	}

	if (kernel->get_major_version() > props.major || kernel->get_major_version() == props.major && kernel->get_minor_version() > props.minor)
	{
		applog(LOG_ERR, "GPU #%d: FATAL: the '%c' kernel requires %d.%d capability!", device_map[thr_id], kernel->get_identifier(), kernel->get_major_version(), kernel->get_minor_version());
		return 0;
	}

	// set whatever cache configuration and shared memory bank mode the kernel prefers
	checkCudaErrors(cudaDeviceSetCacheConfig(kernel->cache_config()));
	checkCudaErrors(cudaDeviceSetSharedMemConfig(kernel->shared_mem_config()));

	// some kernels (e.g. Titan) do not support the texture cache
	if (kernel->no_textures() && device_texturecache[thr_id]) {
		applog(LOG_WARNING, "GPU #%d: the '%c' kernel ignores the texture cache argument", device_map[thr_id], kernel->get_identifier());
		device_texturecache[thr_id] = 0;
	}

	// Texture caching only works with single memory allocation
	if (device_texturecache[thr_id]) device_singlememory[thr_id] = 1;

	if (kernel->single_memory() && !device_singlememory[thr_id]) {
		applog(LOG_WARNING, "GPU #%d: the '%c' kernel requires single memory allocation", device_map[thr_id], kernel->get_identifier());
		device_singlememory[thr_id] = 1;
	}

	if (device_lookup_gap[thr_id] == 0) device_lookup_gap[thr_id] = 1;
	if (!kernel->support_lookup_gap() && device_lookup_gap[thr_id] > 1)
	{
		applog(LOG_WARNING, "GPU #%d: the '%c' kernel does not support a lookup gap", device_map[thr_id], kernel->get_identifier());
		device_lookup_gap[thr_id] = 1;
	}

	if (opt_debug) {
		applog(LOG_INFO, "GPU #%d: interactive: %d, tex-cache: %d%s, single-alloc: %d", device_map[thr_id],
		   (device_interactive[thr_id]  != 0) ? 1 : 0,
		   (device_texturecache[thr_id] != 0) ? device_texturecache[thr_id] : 0, (device_texturecache[thr_id] != 0) ? "D" : "",
		   (device_singlememory[thr_id] != 0) ? 1 : 0 );
	}

	// number of threads collaborating on one work unit (hash)
	unsigned int THREADS_PER_WU = kernel->threads_per_wu();
	unsigned int LOOKUP_GAP = device_lookup_gap[thr_id];
	unsigned int BACKOFF = device_backoff[thr_id];
	unsigned int N = (1 << (opt_nfactor+1));
	double szPerWarp = (double)(SCRATCH * WU_PER_WARP * sizeof(uint32_t));
	//applog(LOG_INFO, "WU_PER_WARP=%u, THREADS_PER_WU=%u, LOOKUP_GAP=%u, BACKOFF=%u, SCRATCH=%u", WU_PER_WARP, THREADS_PER_WU, LOOKUP_GAP, BACKOFF, SCRATCH);
	applog(LOG_INFO, "GPU #%d: %d hashes / %.1f MB per warp.", device_map[thr_id], WU_PER_WARP, szPerWarp / (1024.0 * 1024.0));

	// compute highest MAXWARPS numbers for kernels allowing cudaBindTexture to succeed
	int MW_1D_4 = 134217728 / (SCRATCH * WU_PER_WARP / 4); // for uint4_t textures
	int MW_1D_2 = 134217728 / (SCRATCH * WU_PER_WARP / 2); // for uint2_t textures
	int MW_1D = kernel->get_texel_width() == 2 ? MW_1D_2 : MW_1D_4;

	uint32_t *d_V = NULL;
	if (device_singlememory[thr_id])
	{
		// if no launch config was specified, we simply
		// allocate the single largest memory chunk on the device that we can get
		if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK)) {
			MAXWARPS[thr_id] = optimal_blocks * WARPS_PER_BLOCK;
		}
		else {
			// compute no. of warps to allocate the largest number producing a single memory block
			// PROBLEM: one some devices, ALL allocations will fail if the first one failed. This sucks.
			size_t MEM_LIMIT = (size_t)min((unsigned long long)MAXMEM, (unsigned long long)props.totalGlobalMem);
			int warpmax = (int)min((unsigned long long)TOTAL_WARP_LIMIT, (unsigned long long)(MEM_LIMIT / szPerWarp));

			// run a bisection algorithm for memory allocation (way more reliable than the previous approach)
			int best = 0;
			int warp = (warpmax+1)/2;
			int interval = (warpmax+1)/2;
			while (interval > 0)
			{
				cudaGetLastError(); // clear the error state
				cudaMalloc((void **)&d_V, (size_t)(szPerWarp * warp));
				if (cudaGetLastError() == cudaSuccess) {
					checkCudaErrors(cudaFree(d_V)); d_V = NULL;
					if (warp > best) best = warp;
					if (warp == warpmax) break;
					interval = (interval+1)/2;
					warp += interval;
					if (warp > warpmax) warp = warpmax;
				}
				else
				{
					interval = interval/2;
					warp -= interval;
					if (warp < 1) warp = 1;
				}
			}
			// back off a bit from the largest possible allocation size
			MAXWARPS[thr_id] = ((100-BACKOFF)*best+50)/100;
		}

		// now allocate a buffer for determined MAXWARPS setting
		cudaGetLastError(); // clear the error state
		cudaMalloc((void **)&d_V, (size_t)SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t));
		if (cudaGetLastError() == cudaSuccess) {
			for (int i=0; i < MAXWARPS[thr_id]; ++i)
				h_V[thr_id][i] = d_V + SCRATCH * WU_PER_WARP * i;

			if (device_texturecache[thr_id] == 1)
			{
				if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
				{
					if ( optimal_blocks * WARPS_PER_BLOCK > MW_1D ) {
						applog(LOG_ERR, "GPU #%d: '%s' exceeds limits for 1D cache. Using 2D cache instead.", device_map[thr_id], device_config[thr_id]);
						device_texturecache[thr_id] = 2;
					}
				}
				// bind linear memory to a 1D texture reference
				if (kernel->get_texel_width() == 2)
					kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * min(MAXWARPS[thr_id],MW_1D_2) * sizeof(uint32_t));
				else
					kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * min(MAXWARPS[thr_id],MW_1D_4) * sizeof(uint32_t));
			}
			else if (device_texturecache[thr_id] == 2)
			{
				// bind pitch linear memory to a 2D texture reference
				if (kernel->get_texel_width() == 2)
					kernel->bindtexture_2D(d_V, SCRATCH/2, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
				else
					kernel->bindtexture_2D(d_V, SCRATCH/4, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
			}
		}
		else
		{
			applog(LOG_ERR, "GPU #%d: FATAL: Launch config '%s' requires too much memory!", device_map[thr_id], device_config[thr_id]);
			return 0;
		}
	}
	else
	{
		if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
			MAXWARPS[thr_id] = optimal_blocks * WARPS_PER_BLOCK;
		else
			MAXWARPS[thr_id] = TOTAL_WARP_LIMIT;

		// chunked memory allocation up to device limits
		int warp;
		for (warp = 0; warp < MAXWARPS[thr_id]; ++warp) {
			// work around partition camping problems by adding a random start address offset to each allocation
			h_V_extra[thr_id][warp] = (props.major == 1) ? (16 * (rand()%(16384/16))) : 0;
			cudaGetLastError(); // clear the error state
			cudaMalloc((void **) &h_V[thr_id][warp], (SCRATCH * WU_PER_WARP + h_V_extra[thr_id][warp])*sizeof(uint32_t));
			if (cudaGetLastError() == cudaSuccess) h_V[thr_id][warp] += h_V_extra[thr_id][warp];
			else {
				h_V_extra[thr_id][warp] = 0;

				// back off by several warp allocations to have some breathing room
				int remove = (BACKOFF*warp+50)/100;
				for (int i=0; warp > 0 && i < remove; ++i) {
					warp--;
					checkCudaErrors(cudaFree(h_V[thr_id][warp]-h_V_extra[thr_id][warp]));
					h_V[thr_id][warp] = NULL; h_V_extra[thr_id][warp] = 0;
				}

				break;
			}
		}
		MAXWARPS[thr_id] = warp;
	}
	kernel->set_scratchbuf_constants(MAXWARPS[thr_id], h_V[thr_id]);

	if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
	{
		if (optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
		{
			applog(LOG_ERR, "GPU #%d: FATAL: Given launch config '%s' requires too much memory.", device_map[thr_id], device_config[thr_id]);
			return 0;
		}

		if (WARPS_PER_BLOCK > kernel->max_warps_per_block())
		{
			applog(LOG_ERR, "GPU #%d: FATAL: Given launch config '%s' exceeds warp limit for '%c' kernel.", device_map[thr_id], device_config[thr_id], kernel->get_identifier());
			return 0;
		}
	}
	else
	{
		if (device_config[thr_id] != NULL && strcasecmp("auto", device_config[thr_id]))
			applog(LOG_WARNING, "GPU #%d: Given launch config '%s' does not validate.", device_map[thr_id], device_config[thr_id]);

		if (opt_autotune)
		{
			applog(LOG_INFO, "GPU #%d: Performing auto-tuning, please wait 2 minutes...", device_map[thr_id]);

			// allocate device memory
			uint32_t *d_idata = NULL, *d_odata = NULL;
			unsigned int mem_size = MAXWARPS[thr_id] * WU_PER_WARP * sizeof(uint32_t) * 32;
			checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
			checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

			// pre-initialize some device memory
			uint32_t *h_idata = (uint32_t*)malloc(mem_size);
			for (unsigned int i=0; i < mem_size/sizeof(uint32_t); ++i) h_idata[i] = i*2654435761UL; // knuth's method
			checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
			free(h_idata);

			double best_hash_sec = 0.0;
			int best_wpb = 0;

			// auto-tuning loop
			{
				// we want to have enough total warps for half the multiprocessors at least
				// compute highest MAXWARPS number that we can support based on texture cache mode
				int MINTW = props.multiProcessorCount / 2;
				int MAXTW = (device_texturecache[thr_id] == 1) ? min(MAXWARPS[thr_id],MW_1D) : MAXWARPS[thr_id];

				// we want to have blocks for half the multiprocessors at least
				int MINB = props.multiProcessorCount / 2;
				int MAXB = MAXTW;

				double tmin = 0.05;

				applog(LOG_INFO, "GPU #%d: maximum total warps (BxW): %d", (int) device_map[thr_id], MAXTW);

				for (int GRID_BLOCKS = MINB; !abort_flag && GRID_BLOCKS <= MAXB; ++GRID_BLOCKS)
				{
					double Hash[32+1] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
					for (WARPS_PER_BLOCK = 1; !abort_flag && WARPS_PER_BLOCK <= kernel->max_warps_per_block(); ++WARPS_PER_BLOCK)
					{
						double hash_sec = 0;
						if (GRID_BLOCKS * WARPS_PER_BLOCK >= MINTW &&
							GRID_BLOCKS * WARPS_PER_BLOCK <= MAXTW)
						{
							// setup execution parameters
							dim3  grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
							dim3  threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

							struct timeval tv_start, tv_end;
							double tdelta = 0;

							checkCudaErrors(cudaDeviceSynchronize());
							gettimeofday(&tv_start, NULL);
							int repeat = 0;
							do  // average several measurements for better exactness
							{
								kernel->run_kernel(
									grid, threads, WARPS_PER_BLOCK, thr_id, NULL, d_idata, d_odata, N,
									LOOKUP_GAP, device_interactive[thr_id], true, device_texturecache[thr_id]
								);
								if(cudaDeviceSynchronize() != cudaSuccess)
									break;
								++repeat;
								gettimeofday(&tv_end, NULL);
								// for a better result averaging, measure for at least 50ms (10ms for Keccak)
							} while ((tdelta=(1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec))) < tmin);
							if (cudaGetLastError() != cudaSuccess) continue;

							tdelta /= repeat; // BUGFIX: this averaging over multiple measurements was missing

							// for scrypt: in interactive mode only find launch configs where kernel launch times are short enough
							// TODO: instead we could reduce the batchsize parameter to meet the launch time requirement.
							if (IS_SCRYPT() && device_interactive[thr_id]
								&& GRID_BLOCKS > 2*props.multiProcessorCount && tdelta > 1.0/30)
							{
								if (WARPS_PER_BLOCK == 1) goto skip; else goto skip2;
							}

							hash_sec = (double)WU_PER_LAUNCH / tdelta;
							Hash[WARPS_PER_BLOCK] = hash_sec;
							if (hash_sec > best_hash_sec) {
								optimal_blocks = GRID_BLOCKS;
								best_hash_sec = hash_sec;
								best_wpb = WARPS_PER_BLOCK;
							}
						}
					}
skip2:
					if (opt_debug) {

						if (GRID_BLOCKS == MINB) {
							char line[512] = "    ";
							for (int i=1; i<=kernel->max_warps_per_block(); ++i) {
								char tmp[16]; sprintf(tmp, i < 10 ? "   x%-2d" : "  x%-2d ", i);
								strcat(line, tmp);
								if (cw == 80 && (i % 8 == 0 && i != kernel->max_warps_per_block()))
									strcat(line, "\n                          ");
							}
							applog(LOG_DEBUG, line);
						}

						char kMGT = ' '; bool flag;
						for (int j=0; j < 4; ++j) {
							flag=false; for (int i=1; i<=kernel->max_warps_per_block(); flag|=Hash[i] >= 1000, i++);
							if (flag)   for (int i=1; i<=kernel->max_warps_per_block(); Hash[i] /= 1000, i++);
							else break;
								 if (kMGT == ' ') kMGT = 'k';
							else if (kMGT == 'k') kMGT = 'M';
							else if (kMGT == 'M') kMGT = 'G';
							else if (kMGT == 'G') kMGT = 'T';
						}
						const char *format = "%5.4f%c";
						flag = false; for (int i=1; i<=kernel->max_warps_per_block(); flag|=Hash[i] >= 1, i++); if (flag) format = "%5.3f%c";
						flag = false; for (int i=1; i<=kernel->max_warps_per_block(); flag|=Hash[i] >= 10, i++); if (flag) format = "%5.2f%c";
						flag = false; for (int i=1; i<=kernel->max_warps_per_block(); flag|=Hash[i] >= 100, i++); if (flag) format = "%5.1f%c";

						char line[512]; sprintf(line, "%3d:", GRID_BLOCKS);
						for (int i=1; i<=kernel->max_warps_per_block(); ++i) {
							char tmp[16];
							if (Hash[i]>0)
								sprintf(tmp, format, Hash[i], (i<kernel->max_warps_per_block())?'|':' ');
							else
								sprintf(tmp, "     %c", (i<kernel->max_warps_per_block())?'|':' ');
							strcat(line, tmp);
							if (cw == 80 && (i % 8 == 0 && i != kernel->max_warps_per_block()))
								strcat(line, "\n                          ");
						}
						int n = strlen(line)-1; line[n++] = '|'; line[n++] = ' '; line[n++] = kMGT; line[n++] = '\0';
						strcat(line, "H/s");
						applog(LOG_DEBUG, line);
					}
				}
skip:           ;
			}

			checkCudaErrors(cudaFree(d_odata));
			checkCudaErrors(cudaFree(d_idata));

			WARPS_PER_BLOCK = best_wpb;
			applog(LOG_INFO, "GPU #%d: %7.2f hash/s with configuration %c%dx%d", device_map[thr_id], best_hash_sec, kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);
		}
		else
		{
			// Heuristics to find a good kernel launch configuration

			// base the initial block estimate on the number of multiprocessors
			int device_cores = props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor);

			// defaults, in case nothing else is chosen below
			optimal_blocks = 4 * device_cores / WU_PER_WARP;
			WARPS_PER_BLOCK = 2;

			// Based on compute capability, pick a known good block x warp configuration.
			if (props.major >= 3)
			{
				if (props.major == 3 && props.minor == 5) // GK110 (Tesla K20X, K20, GeForce GTX TITAN)
				{
					// TODO: what to do with Titan and Tesla K20(X)?
					// for now, do the same as for GTX 660Ti (2GB)
					optimal_blocks = (int)(optimal_blocks * 0.8809524);
					WARPS_PER_BLOCK = 2;
				}
				else // GK104, GK106, GK107 ...
				{
					if (MAXWARPS[thr_id] > (int)(optimal_blocks * 1.7261905) * 2)
					{
						// this results in 290x2 configuration on GTX 660Ti (3GB)
						// but it requires 3GB memory on the card!
						optimal_blocks = (int)(optimal_blocks * 1.7261905);
						WARPS_PER_BLOCK = 2;
					}
					else
					{
						// this results in 148x2 configuration on GTX 660Ti (2GB)
						optimal_blocks = (int)(optimal_blocks * 0.8809524);
						WARPS_PER_BLOCK = 2;
					}
				}
			}
			// 1st generation Fermi (compute 2.0) GF100, GF110
			else if (props.major == 2 && props.minor == 0)
			{
				// this results in a 60x4 configuration on GTX 570
				optimal_blocks = 4 * device_cores / WU_PER_WARP;
				WARPS_PER_BLOCK = 4;
			}
			// 2nd generation Fermi (compute 2.1) GF104,106,108,114,116
			else if (props.major == 2 && props.minor == 1)
			{
				// this results in a 56x2 configuration on GTX 460
				optimal_blocks = props.multiProcessorCount * 8;
				WARPS_PER_BLOCK = 2;
			}

			// in case we run out of memory with the automatically chosen configuration,
			// first back off with WARPS_PER_BLOCK, then reduce optimal_blocks.
			if (WARPS_PER_BLOCK==3 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
				WARPS_PER_BLOCK = 2;
			while (optimal_blocks > 0 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
				optimal_blocks--;
		}
	}

	applog(LOG_INFO, "GPU #%d: using launch configuration %c%dx%d", device_map[thr_id], kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);

	if (device_singlememory[thr_id])
	{
		if (MAXWARPS[thr_id] != optimal_blocks * WARPS_PER_BLOCK)
		{
			MAXWARPS[thr_id] = optimal_blocks * WARPS_PER_BLOCK;
			if (device_texturecache[thr_id] == 1)
				kernel->unbindtexture_1D();
			else if (device_texturecache[thr_id] == 2)
				kernel->unbindtexture_2D();
			checkCudaErrors(cudaFree(d_V)); d_V = NULL;

			cudaGetLastError(); // clear the error state
			cudaMalloc((void **)&d_V, (size_t)SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t));
			if (cudaGetLastError() == cudaSuccess) {
				for (int i=0; i < MAXWARPS[thr_id]; ++i)
					h_V[thr_id][i] = d_V + SCRATCH * WU_PER_WARP * i;

				if (device_texturecache[thr_id] == 1)
				{
					// bind linear memory to a 1D texture reference
					if (kernel->get_texel_width() == 2)
						kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t));
					else
						kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t));
				}
				else if (device_texturecache[thr_id] == 2)
				{
					// bind pitch linear memory to a 2D texture reference
					if (kernel->get_texel_width() == 2)
						kernel->bindtexture_2D(d_V, SCRATCH/2, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
					else
						kernel->bindtexture_2D(d_V, SCRATCH/4, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
				}

				// update pointers to scratch buffer in constant memory after reallocation
				kernel->set_scratchbuf_constants(MAXWARPS[thr_id], h_V[thr_id]);
			}
			else
			{
				applog(LOG_ERR, "GPU #%d: Unable to allocate enough memory for launch config '%s'.", device_map[thr_id], device_config[thr_id]);
			}
		}
	}
	else
	{
		// back off unnecessary memory allocations to have some breathing room
		while (MAXWARPS[thr_id] > 0 && MAXWARPS[thr_id] > optimal_blocks * WARPS_PER_BLOCK) {
			(MAXWARPS[thr_id])--;
			checkCudaErrors(cudaFree(h_V[thr_id][MAXWARPS[thr_id]]-h_V_extra[thr_id][MAXWARPS[thr_id]]));
			h_V[thr_id][MAXWARPS[thr_id]] = NULL; h_V_extra[thr_id][MAXWARPS[thr_id]] = 0;
		}
	}

	return optimal_blocks;
}

void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream)
{
	unsigned int GRID_BLOCKS = context_blocks[thr_id];
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;

	// copy host memory to device
	cudaMemcpyAsync(context_idata[stream][thr_id], X, mem_size, cudaMemcpyHostToDevice, context_streams[stream][thr_id]);
}

void cuda_scrypt_serialize(int thr_id, int stream)
{
	// if the device can concurrently execute multiple kernels, then we must
	// wait for the serialization event recorded by the other stream
	if (context_concurrent[thr_id] || device_interactive[thr_id])
		cudaStreamWaitEvent(context_streams[stream][thr_id], context_serialize[(stream+1)&1][thr_id], 0);
}

void cuda_scrypt_done(int thr_id, int stream)
{
	// record the serialization event in the current stream
	cudaEventRecord(context_serialize[stream][thr_id], context_streams[stream][thr_id]);
}

void cuda_scrypt_flush(int thr_id, int stream)
{
	// flush the work queue (required for WDDM drivers)
	cudaStreamSynchronize(context_streams[stream][thr_id]);
}

void cuda_scrypt_core(int thr_id, int stream, unsigned int N)
{
	unsigned int GRID_BLOCKS = context_blocks[thr_id];
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	unsigned int LOOKUP_GAP = device_lookup_gap[thr_id];

	// setup execution parameters
	dim3 grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
	dim3 threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

	context_kernel[thr_id]->run_kernel(grid, threads, WARPS_PER_BLOCK, thr_id,
		context_streams[stream][thr_id], context_idata[stream][thr_id], context_odata[stream][thr_id],
		N, LOOKUP_GAP, device_interactive[thr_id], opt_benchmark, device_texturecache[thr_id]
	);
}

void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream, bool postSHA)
{
	unsigned int GRID_BLOCKS = context_blocks[thr_id];
	unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
	unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * (postSHA ? 8 : 32);
	// copy result from device to host (asynchronously)
	checkCudaErrors(cudaMemcpyAsync(X, postSHA ? context_hash[stream][thr_id] : context_odata[stream][thr_id], mem_size, cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
}

bool cuda_scrypt_sync(int thr_id, int stream)
{
	cudaError_t err;
	uint32_t wait_us = 0;

	if (device_interactive[thr_id] && !opt_benchmark)
	{
		// For devices that also do desktop rendering or compositing, we want to free up some time slots.
		// That requires making a pause in work submission when there is no active task on the GPU,
		// and Device Synchronize ensures that.

		// this call was replaced by the loop below to workaround the high CPU usage issue
		//err = cudaDeviceSynchronize();

		while((err = cudaStreamQuery(context_streams[0][thr_id])) == cudaErrorNotReady ||
			  (err == cudaSuccess && (err = cudaStreamQuery(context_streams[1][thr_id])) == cudaErrorNotReady)) {
			usleep(50); wait_us+=50;
		}

		usleep(50); wait_us+=50;
	} else {
		// this call was replaced by the loop below to workaround the high CPU usage issue
		//err = cudaStreamSynchronize(context_streams[stream][thr_id]);

		while((err = cudaStreamQuery(context_streams[stream][thr_id])) == cudaErrorNotReady) {
			usleep(50); wait_us+=50;
		}
	}

	if (err != cudaSuccess) {
		if (!abort_flag)
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", device_map[thr_id], cudaGetErrorString(err));
		return false;
	}

	//if (opt_debug) {
	//	applog(LOG_DEBUG, "GPU #%d: %s %u us", device_map[thr_id], __FUNCTION__, wait_us);
	//}

	return true;
}

uint32_t* cuda_transferbuffer(int thr_id, int stream)
{
	return context_X[stream][thr_id];
}

uint32_t* cuda_hashbuffer(int thr_id, int stream)
{
	return context_H[stream][thr_id];
}
