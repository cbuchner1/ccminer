
#include <jansson.h>

#include "wildkeccak.h"

#ifdef WIN32
#define _PACKED _ALIGN(4)
#else
#define _PACKED __attribute__((__packed__))
#endif

struct _PACKED scratchpad_hi {
    unsigned char prevhash[32];
    uint64_t height;
};

struct _PACKED addendums_array_entry {
    struct scratchpad_hi prev_hi;
    uint64_t add_size;
};


struct _PACKED scratchpad_file_header {
    struct scratchpad_hi current_hi;
    struct addendums_array_entry add_arr[WILD_KECCAK_ADDENDUMS_ARRAY_SIZE];
    uint64_t scratchpad_size;
};


bool rpc2_job_decode(const json_t *job, struct work *work);
bool rpc2_stratum_job(struct stratum_ctx *sctx, json_t *id, json_t *params);
bool rpc2_stratum_gen_work(struct stratum_ctx *sctx, struct work *work);
bool rpc2_stratum_submit(struct pool_infos *pool, struct work *work);

int  rpc2_stratum_thread_stuff(struct pool_infos* pool);

bool rpc2_login_decode(const json_t *val);

void rpc2_init();

void GetScratchpad();
