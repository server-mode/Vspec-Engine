#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vspec/scheduler/request_scheduler.h"

static size_t lcg_next(size_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state;
}

int main(void) {
    VspecRequestScheduler scheduler;
    vspec_request_scheduler_init(
        &scheduler,
        3ULL * 1024ULL * 1024ULL * 1024ULL,
        32U,
        64U,
        2U
    );

    enum { REQUEST_COUNT = 96 };
    uint64_t request_ids[REQUEST_COUNT];
    size_t admitted = 0U;
    size_t rng = 7U;

    for (size_t i = 0U; i < REQUEST_COUNT; ++i) {
        VspecSchedEnqueueArgs req;
        req.reserve_bytes = (4U + (lcg_next(&rng) % 24U)) * 1024U * 1024U;
        req.prompt_tokens = 32U + (lcg_next(&rng) % 256U);
        req.max_new_tokens = 24U + (lcg_next(&rng) % 128U);
        req.priority = (uint16_t)(lcg_next(&rng) % 4U);
        if (vspec_request_scheduler_enqueue(&scheduler, &req, &request_ids[admitted])) {
            admitted += 1U;
        }
    }

    size_t steps = 0U;
    size_t completed = 0U;
    size_t total_generated = 0U;

    while (steps < 4096U) {
        VspecSchedBatchItem batch[64];
        size_t count = vspec_request_scheduler_build_batch(&scheduler, batch, 64U);
        if (count == 0U) {
            break;
        }

        for (size_t i = 0U; i < count; ++i) {
            int eos = ((lcg_next(&rng) % 100U) < 3U) ? 1 : 0;
            total_generated += batch[i].token_quota;
            if (vspec_request_scheduler_commit(&scheduler, batch[i].request_id, batch[i].token_quota, eos)) {
                (void)eos;
            }
        }

        steps += 1U;

        VspecSchedStats stats;
        vspec_request_scheduler_stats(&scheduler, &stats);
        completed = (size_t)stats.completed_requests;
        if (stats.active_requests == 0U) {
            break;
        }
    }

    VspecSchedStats stats;
    vspec_request_scheduler_stats(&scheduler, &stats);

    printf("[req-sched-bench] admitted=%zu rejected=%llu completed=%llu cancelled=%llu\n",
        admitted,
        (unsigned long long)stats.rejected_requests,
        (unsigned long long)stats.completed_requests,
        (unsigned long long)stats.cancelled_requests);
    printf("[req-sched-bench] steps=%zu total_generated=%zu active=%zu queued=%zu reserved_vram=%zu\n",
        steps,
        total_generated,
        stats.active_requests,
        stats.queued_requests,
        stats.reserved_vram_bytes);
    printf("[req-sched-bench] efficiency_tokens_per_step=%.3f\n",
        (steps > 0U) ? ((double)total_generated / (double)steps) : 0.0);
    printf("[req-sched-bench] completed_snapshot=%zu\n", completed);
    return 0;
}