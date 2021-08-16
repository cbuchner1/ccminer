/*
 * haval-256 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014 djm34
 *               2016 tpruvot
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
 */
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

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
	(x7) = (uint32_t)(ROTR32(t, 7) + ROTR32((x7), 11) + (w) + (c)); \
}

#define PASS1(n, in) { \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[ 1], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[ 2], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[ 5], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[ 6], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[ 7], 0U); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[ 8], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[10], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[11], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[12], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[13], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[14], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[15], 0U); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[16], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[17], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[18], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[19], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[20], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[21], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[22], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[23], 0U); \
 \
	STEP(n, 1, s7, s6, s5, s4, s3, s2, s1, s0, in[24], 0U); \
	STEP(n, 1, s6, s5, s4, s3, s2, s1, s0, s7, in[25], 0U); \
	STEP(n, 1, s5, s4, s3, s2, s1, s0, s7, s6, in[26], 0U); \
	STEP(n, 1, s4, s3, s2, s1, s0, s7, s6, s5, in[27], 0U); \
	STEP(n, 1, s3, s2, s1, s0, s7, s6, s5, s4, in[28], 0U); \
	STEP(n, 1, s2, s1, s0, s7, s6, s5, s4, s3, in[29], 0U); \
	STEP(n, 1, s1, s0, s7, s6, s5, s4, s3, s2, in[30], 0U); \
	STEP(n, 1, s0, s7, s6, s5, s4, s3, s2, s1, in[31], 0U); \
}

#define PASS2(n, in) { \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x452821E6); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[14], 0x38D01377); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[26], 0xBE5466CF); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[18], 0x34E90C6C); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[11], 0xC0AC29B7); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[28], 0xC97C50DD); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 7], 0x3F84D5B5); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[16], 0xB5470917); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[ 0], 0x9216D5D9); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0x8979FB1B); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[20], 0xD1310BA6); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[22], 0x98DFB5AC); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], 0x2FFD72DB); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[10], 0xD01ADFB7); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[ 4], 0xB8E1AFED); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 8], 0x6A267E96); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[30], 0xBA7C9045); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0xF12C7F99); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0x24A19947); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[ 9], 0xB3916CF7); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x0801F2E2); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[24], 0x858EFC16); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[29], 0x636920D8); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[ 6], 0x71574E69); \
 \
	STEP(n, 2, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0xA458FEA3); \
	STEP(n, 2, s6, s5, s4, s3, s2, s1, s0, s7, in[12], 0xF4933D7E); \
	STEP(n, 2, s5, s4, s3, s2, s1, s0, s7, s6, in[15], 0x0D95748F); \
	STEP(n, 2, s4, s3, s2, s1, s0, s7, s6, s5, in[13], 0x728EB658); \
	STEP(n, 2, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], 0x718BCD58); \
	STEP(n, 2, s2, s1, s0, s7, s6, s5, s4, s3, in[25], 0x82154AEE); \
	STEP(n, 2, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x7B54A41D); \
	STEP(n, 2, s0, s7, s6, s5, s4, s3, s2, s1, in[27], 0xC25A59B5); \
}

#define PASS3(n, in) { \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x9C30D539); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x2AF26013); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 4], 0xC5D1B023); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[20], 0x286085F0); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[28], 0xCA417918); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[17], 0xB8DB38EF); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 8], 0x8E79DCB0); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[22], 0x603A180E); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[29], 0x6C9E0E8B); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[14], 0xB01E8A3E); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[25], 0xD71577C1); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[12], 0xBD314B27); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[24], 0x78AF2FDA); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[30], 0x55605C60); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[16], 0xE65525F3); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[26], 0xAA55AB94); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[31], 0x57489862); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[15], 0x63E81440); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[ 7], 0x55CA396A); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[ 3], 0x2AAB10B6); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[ 1], 0xB4CC5C34); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[ 0], 0x1141E8CE); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[18], 0xA15486AF); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[27], 0x7C72E993); \
 \
	STEP(n, 3, s7, s6, s5, s4, s3, s2, s1, s0, in[13], 0xB3EE1411); \
	STEP(n, 3, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], 0x636FBC2A); \
	STEP(n, 3, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0x2BA9C55D); \
	STEP(n, 3, s4, s3, s2, s1, s0, s7, s6, s5, in[10], 0x741831F6); \
	STEP(n, 3, s3, s2, s1, s0, s7, s6, s5, s4, in[23], 0xCE5C3E16); \
	STEP(n, 3, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x9B87931E); \
	STEP(n, 3, s1, s0, s7, s6, s5, s4, s3, s2, in[ 5], 0xAFD6BA33); \
	STEP(n, 3, s0, s7, s6, s5, s4, s3, s2, s1, in[ 2], 0x6C24CF5C); \
}

#define PASS4(n, in) { \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[24], 0x7A325381); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 4], 0x28958677); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 0], 0x3B8F4898); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[14], 0x6B4BB9AF); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 2], 0xC4BFE81B); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[ 7], 0x66282193); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x61D809CC); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[23], 0xFB21A991); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[26], 0x487CAC60); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[ 6], 0x5DEC8032); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[30], 0xEF845D5D); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[20], 0xE98575B1); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDC262302); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[25], 0xEB651B88); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[19], 0x23893E81); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 3], 0xD396ACC5); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[22], 0x0F6D6FF3); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[11], 0x83F44239); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[31], 0x2E0B4482); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[21], 0xA4842004); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[ 8], 0x69C8F04A); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[27], 0x9E1F9B5E); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[12], 0x21C66842); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[ 9], 0xF6E96C9A); \
 \
	STEP(n, 4, s7, s6, s5, s4, s3, s2, s1, s0, in[ 1], 0x670C9C61); \
	STEP(n, 4, s6, s5, s4, s3, s2, s1, s0, s7, in[29], 0xABD388F0); \
	STEP(n, 4, s5, s4, s3, s2, s1, s0, s7, s6, in[ 5], 0x6A51A0D2); \
	STEP(n, 4, s4, s3, s2, s1, s0, s7, s6, s5, in[15], 0xD8542F68); \
	STEP(n, 4, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x960FA728); \
	STEP(n, 4, s2, s1, s0, s7, s6, s5, s4, s3, in[10], 0xAB5133A3); \
	STEP(n, 4, s1, s0, s7, s6, s5, s4, s3, s2, in[16], 0x6EEF0B6C); \
	STEP(n, 4, s0, s7, s6, s5, s4, s3, s2, s1, in[13], 0x137A3BE4); \
}

#define PASS5(n, in) { \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[27], 0xBA3BF050); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 3], 0x7EFB2A98); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[21], 0xA1F1651D); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[26], 0x39AF0176); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[17], 0x66CA593E); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[11], 0x82430E88); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[20], 0x8CEE8619); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[29], 0x456F9FB4); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[19], 0x7D84A5C3); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 0], 0x3B8B5EBE); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[12], 0xE06F75D8); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[ 7], 0x85C12073); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[13], 0x401A449F); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 8], 0x56C16AA6); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[31], 0x4ED3AA62); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[10], 0x363F7706); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 5], 0x1BFEDF72); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[ 9], 0x429B023D); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[14], 0x37D0D724); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[30], 0xD00A1248); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[18], 0xDB0FEAD3); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 6], 0x49F1C09B); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[28], 0x075372C9); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[24], 0x80991B7B); \
 \
	STEP(n, 5, s7, s6, s5, s4, s3, s2, s1, s0, in[ 2], 0x25D479D8); \
	STEP(n, 5, s6, s5, s4, s3, s2, s1, s0, s7, in[23], 0xF6E8DEF7); \
	STEP(n, 5, s5, s4, s3, s2, s1, s0, s7, s6, in[16], 0xE3FE501A); \
	STEP(n, 5, s4, s3, s2, s1, s0, s7, s6, s5, in[22], 0xB6794C3B); \
	STEP(n, 5, s3, s2, s1, s0, s7, s6, s5, s4, in[ 4], 0x976CE0BD); \
	STEP(n, 5, s2, s1, s0, s7, s6, s5, s4, s3, in[ 1], 0x04C006BA); \
	STEP(n, 5, s1, s0, s7, s6, s5, s4, s3, s2, in[25], 0xC1A94FB6); \
	STEP(n, 5, s0, s7, s6, s5, s4, s3, s2, s1, in[15], 0x409F60C4); \
}

__global__ /* __launch_bounds__(256, 6) */
void x17_haval256_gpu_hash_64(const uint32_t threads, uint64_t *g_hash, const int outlen)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint64_t hashPosition = thread*8U;
		uint64_t *pHash = &g_hash[hashPosition];

		uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
		const uint32_t u0 = s0 = 0x243F6A88;
		const uint32_t u1 = s1 = 0x85A308D3;
		const uint32_t u2 = s2 = 0x13198A2E;
		const uint32_t u3 = s3 = 0x03707344;
		const uint32_t u4 = s4 = 0xA4093822;
		const uint32_t u5 = s5 = 0x299F31D0;
		const uint32_t u6 = s6 = 0x082EFA98;
		const uint32_t u7 = s7 = 0xEC4E6C89;

		union {
			uint32_t h4[16];
			uint64_t h8[8];
		} hash;

		#pragma unroll
		for (int i=0; i<8; i++) {
			hash.h8[i] = pHash[i];
		}

		///////// input big /////////////////////

		uint32_t buf[32];

		#pragma unroll
		for (int i=0; i<16; i++)
			buf[i] = hash.h4[i];

		buf[16] = 0x00000001;

		#pragma unroll
		for (int i=17; i<29; i++)
			buf[i] = 0;

		buf[29] = 0x40290000;
		buf[30] = 0x00000200;
		buf[31] = 0;

		PASS1(5, buf);
		PASS2(5, buf);
		PASS3(5, buf);
		PASS4(5, buf);
		PASS5(5, buf);

		hash.h4[0] = s0 + u0;
		hash.h4[1] = s1 + u1;
		hash.h4[2] = s2 + u2;
		hash.h4[3] = s3 + u3;
		hash.h4[4] = s4 + u4;
		hash.h4[5] = s5 + u5;
		hash.h4[6] = s6 + u6;
		hash.h4[7] = s7 + u7;

		pHash[0] = hash.h8[0];
		pHash[1] = hash.h8[1];
		pHash[2] = hash.h8[2];
		pHash[3] = hash.h8[3];

		if (outlen == 512) {
			pHash[4] = 0; //hash.h8[4];
			pHash[5] = 0; //hash.h8[5];
			pHash[6] = 0; //hash.h8[6];
			pHash[7] = 0; //hash.h8[7];
		}
	}
}

__host__
void x17_haval256_cpu_init(int thr_id, uint32_t threads)
{
}

__host__
void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const int outlen)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x17_haval256_gpu_hash_64 <<<grid, block>>> (threads, (uint64_t*)d_hash, outlen);
}
