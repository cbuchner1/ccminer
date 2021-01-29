/* File included in quark/groestl (quark/jha,nist5/X11+) and groest/myriad coins for SM 3+ */

#define merge8(z,x,y)\
	z=__byte_perm(x, y, 0x5140); \

#define SWAP8(x,y)\
	x=__byte_perm(x, y, 0x5410); \
	y=__byte_perm(x, y, 0x7632);

#define SWAP4(x,y)\
	t = (y<<4); \
	t = (x ^ t); \
	t = 0xf0f0f0f0UL & t; \
	x = (x ^ t); \
	t=  t>>4;\
	y=  y ^ t;

#define SWAP2(x,y)\
	t = (y<<2); \
	t = (x ^ t); \
	t = 0xccccccccUL & t; \
	x = (x ^ t); \
	t=  t>>2;\
	y=  y ^ t;

#define SWAP1(x,y)\
	t = (y+y); \
	t = (x ^ t); \
	t = 0xaaaaaaaaUL & t; \
	x = (x ^ t); \
	t=  t>>1;\
	y=  y ^ t;


__device__ __forceinline__
void to_bitslice_quad(uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t other[8];
	uint32_t d[8];
	uint32_t t;
	const unsigned int n = threadIdx.x & 3;

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		input[i] = __shfl((int)input[i], n ^ (3*(n >=1 && n <=2)), 4);
		other[i] = __shfl((int)input[i], (threadIdx.x + 1) & 3, 4);
		input[i] = __shfl((int)input[i], threadIdx.x & 2, 4);
		other[i] = __shfl((int)other[i], threadIdx.x & 2, 4);
		if (threadIdx.x & 1) {
			input[i] = __byte_perm(input[i], 0, 0x1032);
			other[i] = __byte_perm(other[i], 0, 0x1032);
		}
	}

	merge8(d[0], input[0], input[4]);
	merge8(d[1], other[0], other[4]);
	merge8(d[2], input[1], input[5]);
	merge8(d[3], other[1], other[5]);
	merge8(d[4], input[2], input[6]);
	merge8(d[5], other[2], other[6]);
	merge8(d[6], input[3], input[7]);
	merge8(d[7], other[3], other[7]);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);
	SWAP1(d[4], d[5]);
	SWAP1(d[6], d[7]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);
	SWAP2(d[4], d[6]);
	SWAP2(d[5], d[7]);

	SWAP4(d[0], d[4]);
	SWAP4(d[1], d[5]);
	SWAP4(d[2], d[6]);
	SWAP4(d[3], d[7]);

	output[0] = d[0];
	output[1] = d[1];
	output[2] = d[2];
	output[3] = d[3];
	output[4] = d[4];
	output[5] = d[5];
	output[6] = d[6];
	output[7] = d[7];
}

__device__ __forceinline__
void from_bitslice_quad(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t d[8];
	uint32_t t;

	d[0] = __byte_perm(input[0], input[4], 0x7531);
	d[1] = __byte_perm(input[1], input[5], 0x7531);
	d[2] = __byte_perm(input[2], input[6], 0x7531);
	d[3] = __byte_perm(input[3], input[7], 0x7531);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);

	t = __byte_perm(d[0], d[2], 0x5410);
	d[2] = __byte_perm(d[0], d[2], 0x7632);
	d[0] = t;

	t = __byte_perm(d[1], d[3], 0x5410);
	d[3] = __byte_perm(d[1], d[3], 0x7632);
	d[1] = t;

	SWAP4(d[0], d[2]);
	SWAP4(d[1], d[3]);

	output[0] = d[0];
	output[2] = d[1];
	output[4] = d[0] >> 16;
	output[6] = d[1] >> 16;
	output[8] = d[2];
	output[10] = d[3];
	output[12] = d[2] >> 16;
	output[14] = d[3] >> 16;

	#pragma unroll 8
	for (int i = 0; i < 16; i+=2) {
		if (threadIdx.x & 1) output[i] = __byte_perm(output[i], 0, 0x1032);
		output[i] = __byte_perm(output[i], __shfl((int)output[i], (threadIdx.x+1)&3, 4), 0x7610);
		output[i+1] = __shfl((int)output[i], (threadIdx.x+2)&3, 4);
		if (threadIdx.x & 3) output[i] = output[i+1] = 0;
	}
}
