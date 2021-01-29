/**
 * Helper to trace gpu computed data with --cputest
 *
 * Sample usage in an algo scan cuda unit :
 *
 * #define _DEBUG
 * #define _DEBUG_PREFIX "x11-"
 * #include "cuda_debug.cuh"
 *
 * TRACE64("luffa", d_hash);
 * or
 * TRACE("luffa")
 *
 * Dont forget to link the scan function in util.cpp (do_gpu_tests)
 *
 */

#include <stdio.h>
//#include "cuda_helper.h"

#ifndef _DEBUG_PREFIX
#define _DEBUG_PREFIX ""
#endif

#ifdef _DEBUG
#define TRACE64(algo, d_buf) { \
	if (max_nonce == 1 && pdata[19] <= 1 && !opt_benchmark) { \
		uint32_t oft = 0; \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 16*sizeof(uint32_t)); \
		cudaMemcpy(debugbuf, d_buf[thr_id] + oft, 16*sizeof(uint32_t), cudaMemcpyDeviceToHost); \
		printf(_DEBUG_PREFIX "%s %08x %08x %08x %08x %08x %08x %08x %08x  %08x %08x %08x %08x %08x %08x %08x %08x\n", \
			algo, \
			swab32(debugbuf[0]), swab32(debugbuf[1]), swab32(debugbuf[2]), swab32(debugbuf[3]), \
			swab32(debugbuf[4]), swab32(debugbuf[5]), swab32(debugbuf[6]), swab32(debugbuf[7]), \
			swab32(debugbuf[8]), swab32(debugbuf[9]), swab32(debugbuf[10]),swab32(debugbuf[11]), \
			swab32(debugbuf[12]),swab32(debugbuf[13]),swab32(debugbuf[14]),swab32(debugbuf[15])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define TRACE64(algo, d_buf) {}
#endif

// simplified default
#define TRACE(algo) TRACE64(algo, d_hash)

