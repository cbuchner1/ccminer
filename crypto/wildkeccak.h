
#define WILD_KECCAK_SCRATCHPAD_BUFFSIZE   1ULL << 29
#define WILD_KECCAK_ADDENDUMS_ARRAY_SIZE  10

extern uint64_t scratchpad_size;

extern uint32_t WK_CUDABlocks, WK_CUDAThreads;

void wildkeccak_scratchpad_need_update(uint64_t* pscratchpad_buff);

