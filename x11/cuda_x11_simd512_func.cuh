#define SIMD_FUNCTIONS_CUH

__device__ __forceinline__ void STEP8_IF_0(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for(int j=0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[1];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[0];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[3];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[2];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[5];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[4];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[7];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[6];
#pragma unroll 8
	for(int j=0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_1(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[6];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[7];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[4];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[5];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[2];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[3];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[0];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[1];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_2(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[2];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[3];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[0];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[1];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[6];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[7];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[4];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[5];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_3(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int  j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[3];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[2];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[1];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[0];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[7];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[6];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[5];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[4];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_4(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int  j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[5];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[4];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[7];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[6];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[1];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[0];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[3];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[2];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_5(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[7];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[6];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[5];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[4];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[3];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[2];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[1];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[0];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_6(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[4];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[5];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[6];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[7];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[0];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[1];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[2];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[3];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_7(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[1];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[0];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[3];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[2];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[5];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[4];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[7];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[6];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_8(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[6];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[7];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[4];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[5];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[2];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[3];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[0];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[1];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_9(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[2];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[3];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[0];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[1];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[6];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[7];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[4];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[5];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_10(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{

	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[3];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[2];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[1];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[0];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[7];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[6];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[5];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[4];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_11(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[5];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[4];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[7];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[6];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[1];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[0];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[3];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[2];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_12(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[7];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[6];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[5];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[4];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[3];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[2];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[1];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[0];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_13(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[4];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[5];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[6];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[7];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[0];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[1];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[2];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[3];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_14(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[1];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[0];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[3];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[2];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[5];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[4];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[7];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[6];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_15(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[6];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[7];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[4];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[5];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[2];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[3];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[0];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[1];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_16(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[2];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[3];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[0];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[1];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[6];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[7];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[4];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[5];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_17(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[3];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[2];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[1];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[0];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[7];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[6];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[5];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[4];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_18(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[5];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[4];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[7];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[6];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[1];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[0];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[3];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[2];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_19(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[7];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[6];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[5];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[4];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[3];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[2];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[1];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[0];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_20(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[4];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[5];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[6];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[7];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[0];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[1];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[2];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[3];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_21(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[1];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[0];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[3];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[2];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[5];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[4];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[7];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[6];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_22(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[6];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[7];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[4];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[5];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[2];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[3];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[0];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[1];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_23(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[2];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[3];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[0];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[1];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[6];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[7];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[4];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[5];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_24(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[3];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[2];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[1];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[0];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[7];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[6];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[5];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[4];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_25(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[5];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[4];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[7];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[6];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[1];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[0];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[3];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[2];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_26(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[7];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[6];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[5];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[4];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[3];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[2];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[1];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[0];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_27(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[4];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[5];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[6];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[7];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[0];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[1];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[2];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[3];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_28(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[1];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[0];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[3];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[2];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[5];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[4];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[7];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[6];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_29(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[6];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[7];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[4];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[5];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[2];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[3];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[0];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[1];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_30(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[2];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[3];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[0];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[1];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[6];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[7];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[4];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[5];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_MAJ_31(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + MAJ(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[3];
	temp = D[1] + w[1] + MAJ(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[2];
	temp = D[2] + w[2] + MAJ(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[1];
	temp = D[3] + w[3] + MAJ(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[0];
	temp = D[4] + w[4] + MAJ(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[7];
	temp = D[5] + w[5] + MAJ(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[6];
	temp = D[6] + w[6] + MAJ(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[5];
	temp = D[7] + w[7] + MAJ(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[4];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_32(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[5];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[4];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[7];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[6];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[1];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[0];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[3];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[2];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_33(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[7];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[6];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[5];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[4];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[3];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[2];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[1];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[0];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_34(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[4];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[5];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[6];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[7];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[0];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[1];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[2];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[3];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}
__device__ __forceinline__ void STEP8_IF_35(const uint32_t *w, const int r, const int s, uint32_t * A, const uint32_t * B, const uint32_t * C, uint32_t * D)
{
	uint32_t temp;
	uint32_t R[8];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		R[j] = ROTL32(A[j], r);
	}
	temp = D[0] + w[0] + IF(A[0], B[0], C[0]);
	D[0] = ROTL32(temp, s) + R[1];
	temp = D[1] + w[1] + IF(A[1], B[1], C[1]);
	D[1] = ROTL32(temp, s) + R[0];
	temp = D[2] + w[2] + IF(A[2], B[2], C[2]);
	D[2] = ROTL32(temp, s) + R[3];
	temp = D[3] + w[3] + IF(A[3], B[3], C[3]);
	D[3] = ROTL32(temp, s) + R[2];
	temp = D[4] + w[4] + IF(A[4], B[4], C[4]);
	D[4] = ROTL32(temp, s) + R[5];
	temp = D[5] + w[5] + IF(A[5], B[5], C[5]);
	D[5] = ROTL32(temp, s) + R[4];
	temp = D[6] + w[6] + IF(A[6], B[6], C[6]);
	D[6] = ROTL32(temp, s) + R[7];
	temp = D[7] + w[7] + IF(A[7], B[7], C[7]);
	D[7] = ROTL32(temp, s) + R[6];
#pragma unroll 8
	for (int j = 0; j<8; j++) {
		A[j] = R[j];
	}
}

#ifdef DEVICE_DIRECT_CONSTANTS
static __constant__ uint32_t d_cw0[8][8] = {
#else
static __constant__ uint32_t d_cw0[8][8];
static const uint32_t h_cw0[8][8] = {
#endif
	0x531B1720, 0xAC2CDE09, 0x0B902D87, 0x2369B1F4, 0x2931AA01, 0x02E4B082, 0xC914C914, 0xC1DAE1A6,
	0xF18C2B5C, 0x08AC306B, 0x27BFC914, 0xCEDC548D, 0xC630C4BE, 0xF18C4335, 0xF0D3427C, 0xBE3DA380,
	0x143C02E4, 0xA948C630, 0xA4F2DE09, 0xA71D2085, 0xA439BD84, 0x109FCD6A, 0xEEA8EF61, 0xA5AB1CE8,
	0x0B90D4A4, 0x3D6D039D, 0x25944D53, 0xBAA0E034, 0x5BC71E5A, 0xB1F4F2FE, 0x12CADE09, 0x548D41C3,
	0x3CB4F80D, 0x36ECEBC4, 0xA66443EE, 0x43351ABD, 0xC7A20C49, 0xEB0BB366, 0xF5293F98, 0x49B6DE09,
	0x531B29EA, 0x02E402E4, 0xDB25C405, 0x53D4E543, 0x0AD71720, 0xE1A61A04, 0xB87534C1, 0x3EDF43EE,
	0x213E50F0, 0x39173EDF, 0xA9485B0E, 0xEEA82EF9, 0x14F55771, 0xFAF15546, 0x3D6DD9B3, 0xAB73B92E,
	0x582A48FD, 0xEEA81892, 0x4F7EAA01, 0xAF10A88F, 0x11581720, 0x34C124DB, 0xD1C0AB73, 0x1E5AF0D3
};

__device__ __forceinline__ void Round8_0_final(uint32_t *A, int r, int s, int t, int u)
{
	STEP8_IF_0(d_cw0[0], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_IF_1(d_cw0[1], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_IF_2(d_cw0[2], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_IF_3(d_cw0[3], u, r, &A[8], &A[16], &A[24], A);
	STEP8_MAJ_4(d_cw0[4], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_MAJ_5(d_cw0[5], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_MAJ_6(d_cw0[6], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_MAJ_7(d_cw0[7], u, r, &A[8], &A[16], &A[24], A);
}

#ifdef DEVICE_DIRECT_CONSTANTS
static __constant__ uint32_t d_cw1[8][8] = {
#else
static __constant__ uint32_t d_cw1[8][8];
static const uint32_t h_cw1[8][8] = {
#endif
	0xC34C07F3, 0xC914143C, 0x599CBC12, 0xBCCBE543, 0x385EF3B7, 0x14F54C9A, 0x0AD7C068, 0xB64A21F7,
	0xDEC2AF10, 0xC6E9C121, 0x56B8A4F2, 0x1158D107, 0xEB0BA88F, 0x050FAABA, 0xC293264D, 0x548D46D2,
	0xACE5E8E0, 0x53D421F7, 0xF470D279, 0xDC974E0C, 0xD6CF55FF, 0xFD1C4F7E, 0x36EC36EC, 0x3E261E5A,
	0xEBC4FD1C, 0x56B839D0, 0x5B0E21F7, 0x58E3DF7B, 0x5BC7427C, 0xEF613296, 0x1158109F, 0x5A55E318,
	0xA7D6B703, 0x1158E76E, 0xB08255FF, 0x50F05771, 0xEEA8E8E0, 0xCB3FDB25, 0x2E40548D, 0xE1A60F2D,
	0xACE5D616, 0xFD1CFD1C, 0x24DB3BFB, 0xAC2C1ABD, 0xF529E8E0, 0x1E5AE5FC, 0x478BCB3F, 0xC121BC12,
	0xF4702B5C, 0xC293FC63, 0xDA6CB2AD, 0x45601FCC, 0xA439E1A6, 0x4E0C0D02, 0xED3621F7, 0xAB73BE3D,
	0x0E74D4A4, 0xF754CF95, 0xD84136EC, 0x3124AB73, 0x39D03B42, 0x0E74BCCB, 0x0F2DBD84, 0x41C35C80
};

__device__ __forceinline__ void Round8_1_final(uint32_t *A, int r, int s, int t, int u)
{
	STEP8_IF_8(d_cw1[0], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_IF_9(d_cw1[1], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_IF_10(d_cw1[2], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_IF_11(d_cw1[3], u, r, &A[8], &A[16], &A[24], A);
	STEP8_MAJ_12(d_cw1[4], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_MAJ_13(d_cw1[5], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_MAJ_14(d_cw1[6], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_MAJ_15(d_cw1[7], u, r, &A[8], &A[16], &A[24], A);
}

#ifdef DEVICE_DIRECT_CONSTANTS
static __constant__ uint32_t d_cw2[8][8] = {
#else
static __constant__ uint32_t d_cw2[8][8];
static const uint32_t h_cw2[8][8] = {
#endif
	0xA4135BED, 0xE10E1EF2, 0x6C4F93B1, 0x6E2191DF, 0xE2E01D20, 0xD1952E6B, 0x6A7D9583, 0x131DECE3,
	0x369CC964, 0xFB73048D, 0x9E9D6163, 0x280CD7F4, 0xD9C6263A, 0x1062EF9E, 0x2AC7D539, 0xAD2D52D3,
	0x0A03F5FD, 0x197CE684, 0xAA72558E, 0xDE5321AD, 0xF0870F79, 0x607A9F86, 0xAFE85018, 0x2AC7D539,
	0xE2E01D20, 0x2AC7D539, 0xC6A93957, 0x624C9DB4, 0x6C4F93B1, 0x641E9BE2, 0x452CBAD4, 0x263AD9C6,
	0xC964369C, 0xC3053CFB, 0x452CBAD4, 0x95836A7D, 0x4AA2B55E, 0xAB5B54A5, 0xAC4453BC, 0x74808B80,
	0xCB3634CA, 0xFC5C03A4, 0x4B8BB475, 0x21ADDE53, 0xE2E01D20, 0xDF3C20C4, 0xBD8F4271, 0xAA72558E,
	0xFC5C03A4, 0x48D0B730, 0x2AC7D539, 0xD70B28F5, 0x53BCAC44, 0x3FB6C04A, 0x14EFEB11, 0xDB982468,
	0x9A1065F0, 0xB0D14F2F, 0x8D5272AE, 0xC4D73B29, 0x91DF6E21, 0x949A6B66, 0x303DCFC3, 0x5932A6CE
};

__device__ __forceinline__ void Round8_2_final(uint32_t *A, int r, int s, int t, int u)
{
	STEP8_IF_16(d_cw2[0], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_IF_17(d_cw2[1], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_IF_18(d_cw2[2], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_IF_19(d_cw2[3], u, r, &A[8], &A[16], &A[24], A);
	STEP8_MAJ_20(d_cw2[4], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_MAJ_21(d_cw2[5], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_MAJ_22(d_cw2[6], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_MAJ_23(d_cw2[7], u, r, &A[8], &A[16], &A[24], A);
}

#ifdef DEVICE_DIRECT_CONSTANTS
static __constant__ uint32_t d_cw3[8][8] = {
#else
static __constant__ uint32_t d_cw3[8][8];
static const uint32_t h_cw3[8][8] = {
#endif
	0x1234EDCC, 0xF5140AEC, 0xCDF1320F, 0x3DE4C21C, 0x48D0B730, 0x1234EDCC, 0x131DECE3, 0x52D3AD2D,
	0xE684197C, 0x6D3892C8, 0x72AE8D52, 0x6FF3900D, 0x73978C69, 0xEB1114EF, 0x15D8EA28, 0x71C58E3B,
	0x90F66F0A, 0x15D8EA28, 0x9BE2641E, 0x65F09A10, 0xEA2815D8, 0xBD8F4271, 0x3A40C5C0, 0xD9C6263A,
	0xB38C4C74, 0xBAD4452C, 0x70DC8F24, 0xAB5B54A5, 0x46FEB902, 0x1A65E59B, 0x0DA7F259, 0xA32A5CD6,
	0xD62229DE, 0xB81947E7, 0x6D3892C8, 0x15D8EA28, 0xE59B1A65, 0x065FF9A1, 0xB2A34D5D, 0x6A7D9583,
	0x975568AB, 0xFC5C03A4, 0x2E6BD195, 0x966C6994, 0xF2590DA7, 0x263AD9C6, 0x5A1BA5E5, 0xB0D14F2F,
	0x975568AB, 0x6994966C, 0xF1700E90, 0xD3672C99, 0xCC1F33E1, 0xFC5C03A4, 0x452CBAD4, 0x4E46B1BA,
	0xF1700E90, 0xB2A34D5D, 0xD0AC2F54, 0x5760A8A0, 0x8C697397, 0x624C9DB4, 0xE85617AA, 0x95836A7D
};

__device__ __forceinline__ void Round8_3_final(uint32_t *A, int r, int s, int t, int u)
{
	STEP8_IF_24(d_cw3[0], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_IF_25(d_cw3[1], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_IF_26(d_cw3[2], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_IF_27(d_cw3[3], u, r, &A[8], &A[16], &A[24], A);
	STEP8_MAJ_28(d_cw3[4], r, s, A, &A[8], &A[16], &A[24]);
	STEP8_MAJ_29(d_cw3[5], s, t, &A[24], A, &A[8], &A[16]);
	STEP8_MAJ_30(d_cw3[6], t, u, &A[16], &A[24], A, &A[8]);
	STEP8_MAJ_31(d_cw3[7], u, r, &A[8], &A[16], &A[24], A);
}

#if __CUDA_ARCH__ < 350
#define expanded_vector(x) tex1Dfetch(texRef1D_128, (x))
#else
//#define expanded_vector(x) tex1Dfetch(texRef1D_128, (x))
#define expanded_vector(x) __ldg(&g_fft4[x])
#endif

__device__ __forceinline__ void Round8_0(uint32_t *A, const int thr_offset,
		int r, int s, int t, int u, uint4 *g_fft4) {
	uint32_t w[8];
    uint4 hv1, hv2;

	int tmp = 0 + thr_offset;
	hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
	hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_0(w, r, s, A, &A[8], &A[16], &A[24]);
	hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
	hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_1(w, s, t, &A[24], A, &A[8], &A[16]);
	hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
	hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_2(w, t, u, &A[16], &A[24], A, &A[8]);
	hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
	hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_3(w, u, r, &A[8], &A[16], &A[24], A);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_4(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_5(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_6(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_7(w, u, r, &A[8], &A[16], &A[24], A);


}
__device__ __forceinline__ void Round8_1(uint32_t *A, const int thr_offset,
		int r, int s, int t, int u, uint4 *g_fft4) {
	uint32_t w[8];
    uint4 hv1, hv2;

	int tmp = 16 + thr_offset;
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_8(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_9(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_10(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_11(w, u, r, &A[8], &A[16], &A[24], A);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_12(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_13(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_14(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_15(w, u, r, &A[8], &A[16], &A[24], A);


}
__device__ __forceinline__ void Round8_2(uint32_t *A, const int thr_offset,
		int r, int s, int t, int u, uint4 *g_fft4) {
	uint32_t w[8];
    uint4 hv1, hv2;

	int tmp = 32 + thr_offset;
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_16(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_17(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_18(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_19(w, u, r, &A[8], &A[16], &A[24], A);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_20(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_21(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_22(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_23(w, u, r, &A[8], &A[16], &A[24], A);


}
__device__ __forceinline__ void Round8_3(uint32_t *A, const int thr_offset,
		int r, int s, int t, int u, uint4 *g_fft4) {
	uint32_t w[8];
    uint4 hv1, hv2;

	int tmp = 48 + thr_offset;
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_24(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_25(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_26(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_IF_27(w, u, r, &A[8], &A[16], &A[24], A);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_28(w, r, s, A, &A[8], &A[16], &A[24]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_29(w, s, t, &A[24], A, &A[8], &A[16]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_30(w, t, u, &A[16], &A[24], A, &A[8]);
    hv1 = expanded_vector(tmp++); w[0] = hv1.x; w[1] = hv1.y; w[2] = hv1.z; w[3] = hv1.w;
    hv2 = expanded_vector(tmp++); w[4] = hv2.x; w[5] = hv2.y; w[6] = hv2.z; w[7] = hv2.w;
	STEP8_MAJ_31(w, u, r, &A[8], &A[16], &A[24], A);


}

__device__ __forceinline__ void SIMD_Compress1(uint32_t *A, const int thr_id, const uint32_t *M, uint4 *g_fft4) {
	int i;
	const int thr_offset = thr_id << 6; // thr_id * 128 (je zwei elemente)
#pragma unroll 8
	for(i=0; i<8; i++) {
		A[i] ^= M[i];
		(&A[8])[i] ^= M[8+i];
	}
	Round8_0(A, thr_offset, 3, 23, 17, 27, g_fft4);
	Round8_1(A, thr_offset, 28, 19, 22, 7, g_fft4);
}

__device__ __forceinline__ void Compression1(const uint32_t *hashval, const int texture_id, uint4 *g_fft4, uint32_t *g_state) {
	uint32_t A[32];
	int i;
#pragma unroll 32
	for (i=0; i < 32; i++) A[i] = c_IV_512[i];
	uint32_t buffer[16];
#pragma unroll 16
	for (i=0; i < 16; i++) buffer[i] = hashval[i];
	SIMD_Compress1(A, texture_id, buffer, g_fft4);
	uint32_t *state = (uint32_t*)&g_state[blockIdx.x * (blockDim.x*32)];
#pragma unroll 32
	for (i=0; i < 32; i++) state[threadIdx.x+blockDim.x*i] = A[i];
}

__device__ __forceinline__ void SIMD_Compress2(uint32_t *A, const int thr_id, uint4 *g_fft4) {
	uint32_t IV[4][8];
	int i;
	const int thr_offset = thr_id << 6; // thr_id * 128 (je zwei elemente)
#pragma unroll 8
	for(i=0; i<8; i++) {
		IV[0][i] = c_IV_512[i];
		IV[1][i] = c_IV_512[8+i];
		IV[2][i] = c_IV_512[16+i];
		IV[3][i] = c_IV_512[24+i];
	}
	Round8_2(A, thr_offset, 29, 9, 15, 5, g_fft4);
	Round8_3(A, thr_offset, 4, 13, 10, 25, g_fft4);
	STEP8_IF_32(IV[0],  4, 13, A, &A[8], &A[16], &A[24]);
	STEP8_IF_33(IV[1], 13, 10, &A[24], A, &A[8], &A[16]);
	STEP8_IF_34(IV[2], 10, 25, &A[16], &A[24], A, &A[8]);
	STEP8_IF_35(IV[3], 25,  4, &A[8], &A[16], &A[24], A);
}

__device__ __forceinline__ void Compression2(const int texture_id, uint4 *g_fft4, uint32_t *g_state) {
	uint32_t A[32];
	int i;
	uint32_t *state = &g_state[blockIdx.x * (blockDim.x*32)];
#pragma unroll 32
	for (i=0; i < 32; i++) A[i] = state[threadIdx.x+blockDim.x*i];
	SIMD_Compress2(A, texture_id, g_fft4);
#pragma unroll 32
	for (i=0; i < 32; i++) state[threadIdx.x+blockDim.x*i] = A[i];
}

__device__ __forceinline__ void SIMD_Compress_Final(uint32_t *A, const uint32_t *M) {
	uint32_t IV[4][8];
	int i;
#pragma unroll 8
	for(i=0; i<8; i++) {
		IV[0][i] = A[i];
		IV[1][i] = (&A[8])[i];
		IV[2][i] = (&A[16])[i];
		IV[3][i] = (&A[24])[i];
	}
#pragma unroll 8
	for(i=0; i<8; i++) {
		A[i] ^= M[i];
		(&A[8])[i] ^= M[8+i];
	}
	Round8_0_final(A, 3, 23, 17, 27);
	Round8_1_final(A, 28, 19, 22, 7);
	Round8_2_final(A, 29, 9, 15, 5);
	Round8_3_final(A, 4, 13, 10, 25);
	STEP8_IF_32(IV[0],  4, 13, A, &A[8], &A[16], &A[24]);
	STEP8_IF_33(IV[1], 13, 10, &A[24], A, &A[8], &A[16]);
	STEP8_IF_34(IV[2], 10, 25, &A[16], &A[24], A, &A[8]);
	STEP8_IF_35(IV[3], 25,  4, &A[8], &A[16], &A[24], A);
}

__device__ __forceinline__ void Final(uint32_t *hashval, const int texture_id, uint4 *g_fft4, uint32_t *g_state) {
	uint32_t A[32];
	int i;
	uint32_t *state = &g_state[blockIdx.x * (blockDim.x*32)];
#pragma unroll 32
	for (i=0; i < 32; i++) A[i] = state[threadIdx.x+blockDim.x*i];
	uint32_t buffer[16];
	buffer[0] = 512;
#pragma unroll 15
	for (i=1; i < 16; i++) buffer[i] = 0;
	SIMD_Compress_Final(A, buffer);
#pragma unroll 16
	for (i=0; i < 16; i++)
		hashval[i] = A[i];
}
