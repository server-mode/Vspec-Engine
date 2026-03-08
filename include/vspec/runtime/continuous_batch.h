#ifndef VSPEC_RUNTIME_CONTINUOUS_BATCH_H
#define VSPEC_RUNTIME_CONTINUOUS_BATCH_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/scheduler/request_scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSPEC_CONT_BATCH_MAX_REQUESTS VSPEC_SCHED_MAX_REQUESTS
#define VSPEC_CONT_BATCH_PHASE_PREFILL 1U
#define VSPEC_CONT_BATCH_PHASE_DECODE 2U

typedef struct VspecContinuousBatchItem {
    uint64_t request_id;
    uint32_t phase;
    size_t token_quota;
    size_t prompt_cursor;
} VspecContinuousBatchItem;

typedef struct VspecContinuousBatchStats {
    uint64_t admitted_requests;
    uint64_t rejected_requests;
    uint64_t completed_requests;
    uint64_t cancelled_requests;
    uint64_t prefill_steps;
    uint64_t decode_steps;
    uint64_t prefill_tokens;
    uint64_t decode_tokens;
    size_t active_prefill_requests;
    size_t active_decode_requests;
    size_t reserved_vram_bytes;
} VspecContinuousBatchStats;

typedef struct VspecContinuousBatchRequest {
    uint64_t request_id;
    size_t prompt_tokens_total;
    size_t prompt_tokens_done;
    size_t max_new_tokens;
    size_t generated_tokens;
    uint16_t priority;
    uint8_t active;
    uint8_t finished;
} VspecContinuousBatchRequest;

typedef struct VspecContinuousBatcher {
    VspecRequestScheduler scheduler;
    size_t max_batch_items;
    size_t max_prefill_tokens;
    size_t max_decode_tokens;
    size_t prefill_quantum;
    size_t decode_quantum;
    size_t rr_prefill_cursor;
    size_t rr_decode_cursor;
    VspecContinuousBatchRequest requests[VSPEC_CONT_BATCH_MAX_REQUESTS];
    VspecContinuousBatchStats stats;
} VspecContinuousBatcher;

void vspec_continuous_batch_init(
    VspecContinuousBatcher* batcher,
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_items,
    size_t max_batch_tokens,
    size_t prefill_quantum,
    size_t decode_quantum
);

int vspec_continuous_batch_submit(
    VspecContinuousBatcher* batcher,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority,
    uint64_t* out_request_id
);

size_t vspec_continuous_batch_next_batch(
    VspecContinuousBatcher* batcher,
    VspecContinuousBatchItem* out_items,
    size_t max_items
);

int vspec_continuous_batch_commit_prefill(
    VspecContinuousBatcher* batcher,
    uint64_t request_id,
    size_t consumed_tokens
);

int vspec_continuous_batch_commit_decode(
    VspecContinuousBatcher* batcher,
    uint64_t request_id,
    size_t generated_tokens,
    int reached_eos
);

int vspec_continuous_batch_cancel(
    VspecContinuousBatcher* batcher,
    uint64_t request_id
);

void vspec_continuous_batch_stats(
    const VspecContinuousBatcher* batcher,
    VspecContinuousBatchStats* out_stats
);

#ifdef __cplusplus
}
#endif

#endif
