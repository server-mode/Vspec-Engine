#include "vspec/runtime/continuous_batch.h"

#include <string.h>

static void vspec_cont_batch_reset_request(VspecContinuousBatchRequest* request) {
    if (!request) {
        return;
    }
    (void)memset(request, 0, sizeof(*request));
}

static VspecContinuousBatchRequest* vspec_cont_batch_find_request(
    VspecContinuousBatcher* batcher,
    uint64_t request_id
) {
    if (!batcher || request_id == 0U) {
        return NULL;
    }
    for (size_t i = 0U; i < VSPEC_CONT_BATCH_MAX_REQUESTS; ++i) {
        if (batcher->requests[i].active && batcher->requests[i].request_id == request_id) {
            return &batcher->requests[i];
        }
    }
    return NULL;
}

static const VspecContinuousBatchRequest* vspec_cont_batch_find_request_const(
    const VspecContinuousBatcher* batcher,
    uint64_t request_id
) {
    if (!batcher || request_id == 0U) {
        return NULL;
    }
    for (size_t i = 0U; i < VSPEC_CONT_BATCH_MAX_REQUESTS; ++i) {
        if (batcher->requests[i].active && batcher->requests[i].request_id == request_id) {
            return &batcher->requests[i];
        }
    }
    return NULL;
}

static VspecContinuousBatchRequest* vspec_cont_batch_find_free_request(VspecContinuousBatcher* batcher) {
    if (!batcher) {
        return NULL;
    }
    for (size_t i = 0U; i < VSPEC_CONT_BATCH_MAX_REQUESTS; ++i) {
        if (!batcher->requests[i].active) {
            return &batcher->requests[i];
        }
    }
    return NULL;
}

static int vspec_cont_batch_request_finished(const VspecContinuousBatchRequest* request) {
    if (!request || !request->active || request->finished) {
        return 1;
    }
    if (request->prompt_tokens_done < request->prompt_tokens_total) {
        return 0;
    }
    if (request->max_new_tokens > 0U && request->generated_tokens < request->max_new_tokens) {
        return 0;
    }
    return 1;
}

static void vspec_cont_batch_sync_finished(VspecContinuousBatcher* batcher, uint64_t request_id) {
    VspecContinuousBatchRequest* request = vspec_cont_batch_find_request(batcher, request_id);
    if (!request) {
        return;
    }
    if (vspec_cont_batch_request_finished(request)) {
        request->finished = 1U;
        request->active = 0U;
        request->request_id = 0U;
    }
}

void vspec_continuous_batch_init(
    VspecContinuousBatcher* batcher,
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_items,
    size_t max_batch_tokens,
    size_t prefill_quantum,
    size_t decode_quantum
) {
    if (!batcher) {
        return;
    }

    (void)memset(batcher, 0, sizeof(*batcher));
    vspec_request_scheduler_init(
        &batcher->scheduler,
        total_vram_bytes,
        max_active,
        max_batch_tokens,
        (decode_quantum == 0U) ? 1U : decode_quantum
    );
    batcher->max_batch_items = (max_batch_items == 0U) ? 16U : max_batch_items;
    batcher->max_prefill_tokens = (max_batch_tokens == 0U) ? 8U : max_batch_tokens;
    batcher->max_decode_tokens = (max_batch_tokens == 0U) ? 8U : max_batch_tokens;
    batcher->prefill_quantum = (prefill_quantum == 0U) ? 32U : prefill_quantum;
    batcher->decode_quantum = (decode_quantum == 0U) ? 1U : decode_quantum;
}

int vspec_continuous_batch_submit(
    VspecContinuousBatcher* batcher,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority,
    uint64_t* out_request_id
) {
    VspecSchedEnqueueArgs args;
    VspecContinuousBatchRequest* request = NULL;
    uint64_t request_id = 0U;

    if (!batcher) {
        return 0;
    }

    request = vspec_cont_batch_find_free_request(batcher);
    if (!request) {
        batcher->stats.rejected_requests += 1U;
        return 0;
    }

    (void)memset(&args, 0, sizeof(args));
    args.reserve_bytes = (reserve_bytes == 0U) ? 1U : reserve_bytes;
    args.prompt_tokens = prompt_tokens;
    args.max_new_tokens = max_new_tokens;
    args.priority = priority;

    if (!vspec_request_scheduler_enqueue(&batcher->scheduler, &args, &request_id)) {
        batcher->stats.rejected_requests += 1U;
        return 0;
    }

    vspec_cont_batch_reset_request(request);
    request->request_id = request_id;
    request->prompt_tokens_total = prompt_tokens;
    request->max_new_tokens = max_new_tokens;
    request->priority = priority;
    request->active = 1U;
    request->finished = 0U;

    batcher->stats.admitted_requests += 1U;
    if (out_request_id) {
        *out_request_id = request_id;
    }
    return 1;
}

size_t vspec_continuous_batch_next_batch(
    VspecContinuousBatcher* batcher,
    VspecContinuousBatchItem* out_items,
    size_t max_items
) {
    size_t emitted = 0U;
    size_t token_budget_prefill = 0U;
    size_t token_budget_decode = 0U;
    size_t start_cursor = 0U;
    size_t scanned = 0U;

    if (!batcher || !out_items || max_items == 0U) {
        return 0U;
    }

    token_budget_prefill = batcher->max_prefill_tokens;
    token_budget_decode = batcher->max_decode_tokens;

    start_cursor = batcher->rr_prefill_cursor;
    scanned = 0U;
    while (
        scanned < VSPEC_CONT_BATCH_MAX_REQUESTS &&
        emitted < max_items &&
        emitted < batcher->max_batch_items &&
        token_budget_prefill > 0U
    ) {
        const size_t idx = (start_cursor + scanned) % VSPEC_CONT_BATCH_MAX_REQUESTS;
        VspecContinuousBatchRequest* request = &batcher->requests[idx];
        scanned += 1U;

        if (!request->active || request->finished) {
            continue;
        }
        if (request->prompt_tokens_done >= request->prompt_tokens_total) {
            continue;
        }

        size_t quota = batcher->prefill_quantum;
        const size_t remaining = request->prompt_tokens_total - request->prompt_tokens_done;
        if (quota > remaining) {
            quota = remaining;
        }
        if (quota > token_budget_prefill) {
            quota = token_budget_prefill;
        }
        if (quota == 0U) {
            continue;
        }

        out_items[emitted].request_id = request->request_id;
        out_items[emitted].phase = VSPEC_CONT_BATCH_PHASE_PREFILL;
        out_items[emitted].token_quota = quota;
        out_items[emitted].prompt_cursor = request->prompt_tokens_done;
        emitted += 1U;
        token_budget_prefill -= quota;
        batcher->rr_prefill_cursor = (idx + 1U) % VSPEC_CONT_BATCH_MAX_REQUESTS;
    }

    if (emitted > 0U) {
        batcher->stats.prefill_steps += 1U;
    }

    start_cursor = batcher->rr_decode_cursor;
    scanned = 0U;
    while (
        scanned < VSPEC_CONT_BATCH_MAX_REQUESTS &&
        emitted < max_items &&
        emitted < batcher->max_batch_items &&
        token_budget_decode > 0U
    ) {
        const size_t idx = (start_cursor + scanned) % VSPEC_CONT_BATCH_MAX_REQUESTS;
        VspecContinuousBatchRequest* request = &batcher->requests[idx];
        scanned += 1U;

        if (!request->active || request->finished) {
            continue;
        }
        if (request->prompt_tokens_done < request->prompt_tokens_total) {
            continue;
        }
        if (request->max_new_tokens > 0U && request->generated_tokens >= request->max_new_tokens) {
            request->finished = 1U;
            request->active = 0U;
            continue;
        }

        size_t quota = batcher->decode_quantum;
        if (request->max_new_tokens > 0U) {
            const size_t remaining = request->max_new_tokens - request->generated_tokens;
            if (quota > remaining) {
                quota = remaining;
            }
        }
        if (quota > token_budget_decode) {
            quota = token_budget_decode;
        }
        if (quota == 0U) {
            continue;
        }

        out_items[emitted].request_id = request->request_id;
        out_items[emitted].phase = VSPEC_CONT_BATCH_PHASE_DECODE;
        out_items[emitted].token_quota = quota;
        out_items[emitted].prompt_cursor = request->prompt_tokens_done;
        emitted += 1U;
        token_budget_decode -= quota;
        batcher->rr_decode_cursor = (idx + 1U) % VSPEC_CONT_BATCH_MAX_REQUESTS;
    }

    if (emitted > 0U && token_budget_decode < batcher->max_decode_tokens) {
        batcher->stats.decode_steps += 1U;
    }

    return emitted;
}

int vspec_continuous_batch_commit_prefill(
    VspecContinuousBatcher* batcher,
    uint64_t request_id,
    size_t consumed_tokens
) {
    VspecContinuousBatchRequest* request = vspec_cont_batch_find_request(batcher, request_id);
    if (!request || !request->active || request->finished) {
        return 0;
    }
    if (request->prompt_tokens_done >= request->prompt_tokens_total) {
        return 0;
    }

    request->prompt_tokens_done += consumed_tokens;
    if (request->prompt_tokens_done > request->prompt_tokens_total) {
        request->prompt_tokens_done = request->prompt_tokens_total;
    }
    batcher->stats.prefill_tokens += consumed_tokens;
    return 1;
}

int vspec_continuous_batch_commit_decode(
    VspecContinuousBatcher* batcher,
    uint64_t request_id,
    size_t generated_tokens,
    int reached_eos
) {
    VspecContinuousBatchRequest* request = vspec_cont_batch_find_request(batcher, request_id);
    if (!request || !request->active || request->finished) {
        return 0;
    }
    if (request->prompt_tokens_done < request->prompt_tokens_total) {
        return 0;
    }

    if (!vspec_request_scheduler_commit(&batcher->scheduler, request_id, generated_tokens, reached_eos)) {
        return 0;
    }

    request->generated_tokens += generated_tokens;
    batcher->stats.decode_tokens += generated_tokens;
    if (reached_eos || (request->max_new_tokens > 0U && request->generated_tokens >= request->max_new_tokens)) {
        request->finished = 1U;
        request->active = 0U;
        batcher->stats.completed_requests += 1U;
        vspec_cont_batch_reset_request(request);
    }
    return 1;
}

int vspec_continuous_batch_cancel(
    VspecContinuousBatcher* batcher,
    uint64_t request_id
) {
    VspecContinuousBatchRequest* request = vspec_cont_batch_find_request(batcher, request_id);
    if (!request || !request->active) {
        return 0;
    }
    if (!vspec_request_scheduler_cancel(&batcher->scheduler, request_id)) {
        return 0;
    }
    batcher->stats.cancelled_requests += 1U;
    vspec_cont_batch_reset_request(request);
    return 1;
}

void vspec_continuous_batch_stats(
    const VspecContinuousBatcher* batcher,
    VspecContinuousBatchStats* out_stats
) {
    VspecSchedStats sched_stats;
    if (!batcher || !out_stats) {
        return;
    }

    (void)memset(out_stats, 0, sizeof(*out_stats));
    *out_stats = batcher->stats;
    vspec_request_scheduler_stats(&batcher->scheduler, &sched_stats);
    out_stats->reserved_vram_bytes = sched_stats.reserved_vram_bytes;

    for (size_t i = 0U; i < VSPEC_CONT_BATCH_MAX_REQUESTS; ++i) {
        const VspecContinuousBatchRequest* request = &batcher->requests[i];
        if (!request->active || request->finished) {
            continue;
        }
        if (request->prompt_tokens_done < request->prompt_tokens_total) {
            out_stats->active_prefill_requests += 1U;
        } else {
            out_stats->active_decode_requests += 1U;
        }
    }
}
