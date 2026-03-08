#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vspec/runtime/continuous_batch.h"

static size_t lcg_next(size_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state;
}

int main(void) {
    VspecContinuousBatcher batcher;
    size_t rng = 17U;
    size_t steps = 0U;
    size_t admitted = 0U;
    size_t total_prefill = 0U;
    size_t total_decode = 0U;

    vspec_continuous_batch_init(
        &batcher,
        4ULL * 1024ULL * 1024ULL * 1024ULL,
        32U,
        64U,
        128U,
        64U,
        2U
    );

    for (size_t i = 0U; i < 48U; ++i) {
        uint64_t request_id = 0U;
        const size_t reserve = (8U + (lcg_next(&rng) % 24U)) * 1024U * 1024U;
        const size_t prompt_tokens = 32U + (lcg_next(&rng) % 192U);
        const size_t max_new_tokens = 24U + (lcg_next(&rng) % 96U);
        const uint16_t priority = (uint16_t)(lcg_next(&rng) % 4U);
        if (vspec_continuous_batch_submit(&batcher, reserve, prompt_tokens, max_new_tokens, priority, &request_id)) {
            admitted += 1U;
        }
    }

    while (steps < 4096U) {
        VspecContinuousBatchItem items[64];
        size_t count = vspec_continuous_batch_next_batch(&batcher, items, 64U);
        if (count == 0U) {
            break;
        }
        for (size_t i = 0U; i < count; ++i) {
            if (items[i].phase == VSPEC_CONT_BATCH_PHASE_PREFILL) {
                total_prefill += items[i].token_quota;
                (void)vspec_continuous_batch_commit_prefill(&batcher, items[i].request_id, items[i].token_quota);
            } else {
                const int eos = ((lcg_next(&rng) % 100U) < 4U) ? 1 : 0;
                total_decode += items[i].token_quota;
                (void)vspec_continuous_batch_commit_decode(&batcher, items[i].request_id, items[i].token_quota, eos);
            }
        }
        steps += 1U;

        VspecContinuousBatchStats stats;
        vspec_continuous_batch_stats(&batcher, &stats);
        if (stats.active_prefill_requests == 0U && stats.active_decode_requests == 0U) {
            break;
        }
    }

    {
        VspecContinuousBatchStats stats;
        vspec_continuous_batch_stats(&batcher, &stats);
        printf("[cont-batch-bench] admitted=%zu rejected=%llu completed=%llu cancelled=%llu\n",
            admitted,
            (unsigned long long)stats.rejected_requests,
            (unsigned long long)stats.completed_requests,
            (unsigned long long)stats.cancelled_requests);
        printf("[cont-batch-bench] steps=%zu prefill_tokens=%zu decode_tokens=%zu active_prefill=%zu active_decode=%zu reserved_vram=%zu\n",
            steps,
            total_prefill,
            total_decode,
            stats.active_prefill_requests,
            stats.active_decode_requests,
            stats.reserved_vram_bytes);
        printf("[cont-batch-bench] avg_tokens_per_step=%.3f\n",
            (steps > 0U) ? ((double)(total_prefill + total_decode) / (double)steps) : 0.0);
    }

    return 0;
}
