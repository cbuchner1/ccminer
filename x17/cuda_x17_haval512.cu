/*
 * Haval-512 for X17
 *
 * Built on cbuchner1's implementation, actual hashing code
 * heavily based on phm's sgminer
 *
 */

/*
 * Haval-512 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  djm34
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   phm <phm@inbox.com>
 */
#include <stdio.h>
#include <memory.h>

#define USE_SHARED 1

#include "cuda_helper.h"

#define SPH_ROTL32(x, n)   SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))

static __constant__ uint32_t initVector[8];

static const uint32_t c_initVector[8] = {
	SPH_C32(0x243F6A88),
	SPH_C32(0x85A308D3),
	SPH_C32(0x13198A2E),
	SPH_C32(0x03707344),
	SPH_C32(0xA4093822),
	SPH_C32(0x299F31D0),
	SPH_C32(0x082EFA98),
	SPH_C32(0xEC4E6C89)
};

#define PASS1(n, in) { \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], SPH_C32(0x00000000)); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[ 1], SPH_C32(0x00000000)); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[ 2], SPH_C32(0x00000000)); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], SPH_C32(0x00000000)); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], SPH_C32(0x00000000)); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[ 5], SPH_C32(0x00000000)); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[ 6], SPH_C32(0x00000000)); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[ 7], SPH_C32(0x00000000)); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[ 8], SPH_C32(0x00000000)); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], SPH_C32(0x00000000)); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[10], SPH_C32(0x00000000)); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[11], SPH_C32(0x00000000)); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[12], SPH_C32(0x00000000)); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[13], SPH_C32(0x00000000)); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[14], SPH_C32(0x00000000)); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[15], SPH_C32(0x00000000)); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[16], SPH_C32(0x00000000)); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[17], SPH_C32(0x00000000)); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[18], SPH_C32(0x00000000)); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[19], SPH_C32(0x00000000)); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[20], SPH_C32(0x00000000)); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[21], SPH_C32(0x00000000)); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[22], SPH_C32(0x00000000)); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[23], SPH_C32(0x00000000)); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[24], SPH_C32(0x00000000)); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[25], SPH_C32(0x00000000)); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[26], SPH_C32(0x00000000)); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[27], SPH_C32(0x00000000)); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[28], SPH_C32(0x00000000)); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[29], SPH_C32(0x00000000)); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[30], SPH_C32(0x00000000)); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[31], SPH_C32(0x00000000)); \
}

#define PASS2(n, in) { \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], SPH_C32(0x452821E6)); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[14], SPH_C32(0x38D01377)); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[26], SPH_C32(0xBE5466CF)); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[18], SPH_C32(0x34E90C6C)); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[11], SPH_C32(0xC0AC29B7)); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[28], SPH_C32(0xC97C50DD)); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 7], SPH_C32(0x3F84D5B5)); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[16], SPH_C32(0xB5470917)); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], SPH_C32(0x9216D5D9)); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[23], SPH_C32(0x8979FB1B)); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[20], SPH_C32(0xD1310BA6)); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[22], SPH_C32(0x98DFB5AC)); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], SPH_C32(0x2FFD72DB)); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[10], SPH_C32(0xD01ADFB7)); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 4], SPH_C32(0xB8E1AFED)); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 8], SPH_C32(0x6A267E96)); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[30], SPH_C32(0xBA7C9045)); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], SPH_C32(0xF12C7F99)); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[21], SPH_C32(0x24A19947)); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[ 9], SPH_C32(0xB3916CF7)); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[17], SPH_C32(0x0801F2E2)); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[24], SPH_C32(0x858EFC16)); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[29], SPH_C32(0x636920D8)); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 6], SPH_C32(0x71574E69)); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[19], SPH_C32(0xA458FEA3)); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[12], SPH_C32(0xF4933D7E)); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[15], SPH_C32(0x0D95748F)); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[13], SPH_C32(0x728EB658)); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], SPH_C32(0x718BCD58)); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[25], SPH_C32(0x82154AEE)); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[31], SPH_C32(0x7B54A41D)); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[27], SPH_C32(0xC25A59B5)); \
}

#define PASS3(n, in) { \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[19], SPH_C32(0x9C30D539)); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], SPH_C32(0x2AF26013)); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 4], SPH_C32(0xC5D1B023)); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[20], SPH_C32(0x286085F0)); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[28], SPH_C32(0xCA417918)); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[17], SPH_C32(0xB8DB38EF)); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 8], SPH_C32(0x8E79DCB0)); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[22], SPH_C32(0x603A180E)); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[29], SPH_C32(0x6C9E0E8B)); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[14], SPH_C32(0xB01E8A3E)); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[25], SPH_C32(0xD71577C1)); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[12], SPH_C32(0xBD314B27)); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[24], SPH_C32(0x78AF2FDA)); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[30], SPH_C32(0x55605C60)); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[16], SPH_C32(0xE65525F3)); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[26], SPH_C32(0xAA55AB94)); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[31], SPH_C32(0x57489862)); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[15], SPH_C32(0x63E81440)); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 7], SPH_C32(0x55CA396A)); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], SPH_C32(0x2AAB10B6)); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], SPH_C32(0xB4CC5C34)); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[ 0], SPH_C32(0x1141E8CE)); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[18], SPH_C32(0xA15486AF)); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[27], SPH_C32(0x7C72E993)); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[13], SPH_C32(0xB3EE1411)); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], SPH_C32(0x636FBC2A)); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[21], SPH_C32(0x2BA9C55D)); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[10], SPH_C32(0x741831F6)); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[23], SPH_C32(0xCE5C3E16)); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[11], SPH_C32(0x9B87931E)); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 5], SPH_C32(0xAFD6BA33)); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[ 2], SPH_C32(0x6C24CF5C)); \
}

#define PASS4(n, in) { \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[24], SPH_C32(0x7A325381)); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 4], SPH_C32(0x28958677)); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 0], SPH_C32(0x3B8F4898)); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[14], SPH_C32(0x6B4BB9AF)); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], SPH_C32(0xC4BFE81B)); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[ 7], SPH_C32(0x66282193)); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[28], SPH_C32(0x61D809CC)); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[23], SPH_C32(0xFB21A991)); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[26], SPH_C32(0x487CAC60)); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], SPH_C32(0x5DEC8032)); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[30], SPH_C32(0xEF845D5D)); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[20], SPH_C32(0xE98575B1)); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[18], SPH_C32(0xDC262302)); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[25], SPH_C32(0xEB651B88)); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[19], SPH_C32(0x23893E81)); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 3], SPH_C32(0xD396ACC5)); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[22], SPH_C32(0x0F6D6FF3)); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[11], SPH_C32(0x83F44239)); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[31], SPH_C32(0x2E0B4482)); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[21], SPH_C32(0xA4842004)); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 8], SPH_C32(0x69C8F04A)); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[27], SPH_C32(0x9E1F9B5E)); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[12], SPH_C32(0x21C66842)); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 9], SPH_C32(0xF6E96C9A)); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[ 1], SPH_C32(0x670C9C61)); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[29], SPH_C32(0xABD388F0)); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 5], SPH_C32(0x6A51A0D2)); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[15], SPH_C32(0xD8542F68)); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[17], SPH_C32(0x960FA728)); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[10], SPH_C32(0xAB5133A3)); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[16], SPH_C32(0x6EEF0B6C)); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[13], SPH_C32(0x137A3BE4)); \
}

#define PASS5(n, in) { \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[27], SPH_C32(0xBA3BF050)); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], SPH_C32(0x7EFB2A98)); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[21], SPH_C32(0xA1F1651D)); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[26], SPH_C32(0x39AF0176)); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[17], SPH_C32(0x66CA593E)); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[11], SPH_C32(0x82430E88)); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[20], SPH_C32(0x8CEE8619)); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[29], SPH_C32(0x456F9FB4)); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[19], SPH_C32(0x7D84A5C3)); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 0], SPH_C32(0x3B8B5EBE)); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[12], SPH_C32(0xE06F75D8)); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[ 7], SPH_C32(0x85C12073)); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[13], SPH_C32(0x401A449F)); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 8], SPH_C32(0x56C16AA6)); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[31], SPH_C32(0x4ED3AA62)); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[10], SPH_C32(0x363F7706)); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], SPH_C32(0x1BFEDF72)); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], SPH_C32(0x429B023D)); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[14], SPH_C32(0x37D0D724)); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[30], SPH_C32(0xD00A1248)); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[18], SPH_C32(0xDB0FEAD3)); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 6], SPH_C32(0x49F1C09B)); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[28], SPH_C32(0x075372C9)); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[24], SPH_C32(0x80991B7B)); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 2], SPH_C32(0x25D479D8)); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[23], SPH_C32(0xF6E8DEF7)); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[16], SPH_C32(0xE3FE501A)); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[22], SPH_C32(0xB6794C3B)); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], SPH_C32(0x976CE0BD)); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 1], SPH_C32(0x04C006BA)); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[25], SPH_C32(0xC1A94FB6)); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[15], SPH_C32(0x409F60C4)); \
}

#define F1(x6, x5, x4, x3, x2, x1, x0) \
	(((x1) & ((x0) ^ (x4))) ^ ((x2) & (x5)) ^ ((x3) & (x6)) ^ (x0))

#define F2(x6, x5, x4, x3, x2, x1, x0) \
	(((x2) & (((x1) & ~(x3)) ^ ((x4) & (x5)) ^ (x6) ^ (x0))) \
	^ ((x4) & ((x1) ^ (x5))) ^ ((x3 & (x5)) ^ (x0)))

#define F3(x6, x5, x4, x3, x2, x1, x0) \
	(((x3) & (((x1) & (x2)) ^ (x6) ^ (x0))) \
	^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ (x0))

#define F4(x6, x5, x4, x3, x2, x1, x0) \
	(((x3) & (((x1) & (x2)) ^ ((x4) | (x6)) ^ (x5))) \
	^ ((x4) & ((~(x2) & (x5)) ^ (x1) ^ (x6) ^ (x0))) \
	^ ((x2) & (x6)) ^ (x0))

#define F5(x6, x5, x4, x3, x2, x1, x0) \
	(((x0) & ~(((x1) & (x2) & (x3)) ^ (x5))) \
	^ ((x1) & (x4)) ^ ((x2) & (x5)) ^ ((x3) & (x6)))

#define FP5_1(x6, x5, x4, x3, x2, x1, x0) \
	F1(x3, x4, x1, x0, x5, x2, x6)
#define FP5_2(x6, x5, x4, x3, x2, x1, x0) \
	F2(x6, x2, x1, x0, x3, x4, x5)
#define FP5_3(x6, x5, x4, x3, x2, x1, x0) \
	F3(x2, x6, x0, x4, x3, x1, x5)
#define FP5_4(x6, x5, x4, x3, x2, x1, x0) \
	F4(x1, x5, x3, x2, x0, x4, x6)
#define FP5_5(x6, x5, x4, x3, x2, x1, x0) \
	F5(x2, x5, x0, x6, x4, x3, x1)


#define STEP(n, p, x7, x6, x5, x4, x3, x2, x1, x0, w, c) { \
		uint32_t t = FP ## n ## _ ## p(x6, x5, x4, x3, x2, x1, x0); \
		(x7) = SPH_T32(SPH_ROTR32(t, 7) + SPH_ROTR32((x7), 11) \
			+ (w) + (c)); \
	}


__global__
void x17_haval256_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
		int hashPosition = nounce - startNounce;
		uint32_t *inpHash = (uint32_t*)&g_hash[8 * hashPosition];
		union {
			uint8_t h1[64];
			uint32_t h4[16];
			uint64_t h8[8];
		} hash;

		uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
		uint32_t s0,s1,s2,s3,s4,s5,s6,s7;
		uint32_t buf[32];

		s0 = initVector[0];
		s1 = initVector[1];
		s2 = initVector[2];
		s3 = initVector[3];
		s4 = initVector[4];
		s5 = initVector[5];
		s6 = initVector[6];
		s7 = initVector[7];

		u0 = s0;
		u1 = s1;
		u2 = s2;
		u3 = s3;
		u4 = s4;
		u5 = s5;
		u6 = s6;
		u7 = s7;

		#pragma unroll
		for (int i=0; i<16; i++) {
			hash.h4[i]= inpHash[i];
		}

///////// input big /////////////////////

		#pragma unroll
		for (int i=0; i<32; i++) {
			if (i<16) {
				buf[i]=hash.h4[i];
			} else {
				buf[i]=0;
			}
		}

		buf[16]=0x00000001;
		buf[29]=0x40290000;
		buf[30]=0x00000200;

		PASS1(5, buf);
		PASS2(5, buf);
		PASS3(5, buf);
		PASS4(5, buf);
		PASS5(5, buf);

		s0 = SPH_T32(s0 + u0);
		s1 = SPH_T32(s1 + u1);
		s2 = SPH_T32(s2 + u2);
		s3 = SPH_T32(s3 + u3);
		s4 = SPH_T32(s4 + u4);
		s5 = SPH_T32(s5 + u5);
		s6 = SPH_T32(s6 + u6);
		s7 = SPH_T32(s7 + u7);

		hash.h4[0]=s0;
		hash.h4[1]=s1;
			  hash.h4[2]=s2;
		hash.h4[3]=s3;
			  hash.h4[4]=s4;
		hash.h4[5]=s5;
		hash.h4[6]=s6;
		hash.h4[7]=s7;

		#pragma unroll 16
		for (int u = 0; u < 16; u ++)
			inpHash[u] = hash.h4[u];
	} // threads
}

__host__
void x17_haval256_cpu_init(int thr_id, int threads)
{
	cudaMemcpyToSymbol(initVector,c_initVector,sizeof(c_initVector),0, cudaMemcpyHostToDevice);
}

__host__
void x17_haval256_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const int threadsperblock = 256; // Alignment mit mixtab Grösse. NICHT ÄNDERN

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	x17_haval256_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	MyStreamSynchronize(NULL, order, thr_id);
}
