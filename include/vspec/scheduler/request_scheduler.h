#ifndef VSPEC_SCHEDULER_REQUEST_SCHEDULER_H
#define VSPEC_SCHEDULER_REQUEST_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/memory/vram_scheduler.h"

#define VSPEC_SCHED_MAX_REQUESTS 256U

typedef struct VspecSchedRequest {
    uint64_t request_id;
    size_t reserved_bytes;
    size_t prompt_tokens;
    size_t max_new_tokens;
    size_t generated_tokens;
    uint16_t priority;
    uint8_t active;
    uint8_t finished;
} VspecSchedRequest;

typedef struct VspecSchedBatchItem {
    uint64_t request_id;
    size_t token_quota;
} VspecSchedBatchItem;

typedef struct VspecSchedStats {
    uint64_t admitted_requests;
    uint64_t rejected_requests;
    uint64_t completed_requests;
    uint64_t cancelled_requests;
    uint64_t total_scheduled_steps;
    size_t active_requests;
    size_t queued_requests;
    size_t reserved_vram_bytes;
} VspecSchedStats;

typedef struct VspecRequestScheduler {
    VspecVramBudget budget;
    size_t max_active;
    size_t max_batch_tokens;
    size_t token_quantum;

    uint64_t next_request_id;
    size_t rr_cursor;

    VspecSchedRequest slots[VSPEC_SCHED_MAX_REQUESTS];
    VspecSchedStats stats;
} VspecRequestScheduler;

typedef struct VspecSchedEnqueueArgs {
    size_t reserve_bytes;
    size_t prompt_tokens;
    size_t max_new_tokens;
    uint16_t priority;
} VspecSchedEnqueueArgs;

void vspec_request_scheduler_init(
    VspecRequestScheduler* scheduler,
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_tokens,
    size_t token_quantum
);

int vspec_request_scheduler_enqueue(
    VspecRequestScheduler* scheduler,
    const VspecSchedEnqueueArgs* args,
    uint64_t* out_request_id
);

size_t vspec_request_scheduler_build_batch(
    VspecRequestScheduler* scheduler,
    VspecSchedBatchItem* out_items,
    size_t max_items
);

int vspec_request_scheduler_commit(
    VspecRequestScheduler* scheduler,
    uint64_t request_id,
    size_t generated_tokens,
    int reached_eos
);

int vspec_request_scheduler_cancel(
    VspecRequestScheduler* scheduler,
    uint64_t request_id
);

void vspec_request_scheduler_stats(
    const VspecRequestScheduler* scheduler,
    VspecSchedStats* out_stats
);

#endif