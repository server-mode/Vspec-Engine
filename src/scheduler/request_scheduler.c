#include "vspec/scheduler/request_scheduler.h"

#include <string.h>

static void vspec_sched_reset_slot(VspecSchedRequest* slot) {
    if (!slot) {
        return;
    }
    (void)memset(slot, 0, sizeof(*slot));
}

static VspecSchedRequest* vspec_sched_find_by_id(VspecRequestScheduler* scheduler, uint64_t request_id) {
    if (!scheduler || request_id == 0U) {
        return NULL;
    }
    for (size_t i = 0U; i < VSPEC_SCHED_MAX_REQUESTS; ++i) {
        if (scheduler->slots[i].active && scheduler->slots[i].request_id == request_id) {
            return &scheduler->slots[i];
        }
    }
    return NULL;
}

static size_t vspec_sched_count_active(const VspecRequestScheduler* scheduler) {
    size_t active = 0U;
    if (!scheduler) {
        return 0U;
    }
    for (size_t i = 0U; i < VSPEC_SCHED_MAX_REQUESTS; ++i) {
        if (scheduler->slots[i].active && !scheduler->slots[i].finished) {
            active += 1U;
        }
    }
    return active;
}

static size_t vspec_sched_count_queued(const VspecRequestScheduler* scheduler) {
    return vspec_sched_count_active(scheduler);
}

void vspec_request_scheduler_init(
    VspecRequestScheduler* scheduler,
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_tokens,
    size_t token_quantum
) {
    if (!scheduler) {
        return;
    }

    (void)memset(scheduler, 0, sizeof(*scheduler));
    vspec_vram_budget_init(&scheduler->budget, total_vram_bytes);
    scheduler->max_active = (max_active == 0U) ? 1U : max_active;
    scheduler->max_batch_tokens = (max_batch_tokens == 0U) ? 8U : max_batch_tokens;
    scheduler->token_quantum = (token_quantum == 0U) ? 1U : token_quantum;
    scheduler->next_request_id = 1U;
    scheduler->rr_cursor = 0U;
    for (size_t i = 0U; i < VSPEC_SCHED_MAX_REQUESTS; ++i) {
        vspec_sched_reset_slot(&scheduler->slots[i]);
    }
}

int vspec_request_scheduler_enqueue(
    VspecRequestScheduler* scheduler,
    const VspecSchedEnqueueArgs* args,
    uint64_t* out_request_id
) {
    if (!scheduler || !args || args->reserve_bytes == 0U) {
        return 0;
    }

    if (vspec_sched_count_active(scheduler) >= scheduler->max_active) {
        scheduler->stats.rejected_requests += 1U;
        return 0;
    }

    if (!vspec_vram_try_reserve(&scheduler->budget, args->reserve_bytes)) {
        scheduler->stats.rejected_requests += 1U;
        return 0;
    }

    size_t free_idx = VSPEC_SCHED_MAX_REQUESTS;
    for (size_t i = 0U; i < VSPEC_SCHED_MAX_REQUESTS; ++i) {
        if (!scheduler->slots[i].active) {
            free_idx = i;
            break;
        }
    }

    if (free_idx == VSPEC_SCHED_MAX_REQUESTS) {
        (void)vspec_vram_release(&scheduler->budget, args->reserve_bytes);
        scheduler->stats.rejected_requests += 1U;
        return 0;
    }

    VspecSchedRequest* slot = &scheduler->slots[free_idx];
    vspec_sched_reset_slot(slot);
    slot->request_id = scheduler->next_request_id++;
    slot->reserved_bytes = args->reserve_bytes;
    slot->prompt_tokens = args->prompt_tokens;
    slot->max_new_tokens = args->max_new_tokens;
    slot->generated_tokens = 0U;
    slot->priority = args->priority;
    slot->active = 1U;
    slot->finished = 0U;

    scheduler->stats.admitted_requests += 1U;
    if (out_request_id) {
        *out_request_id = slot->request_id;
    }
    return 1;
}

size_t vspec_request_scheduler_build_batch(
    VspecRequestScheduler* scheduler,
    VspecSchedBatchItem* out_items,
    size_t max_items
) {
    if (!scheduler || !out_items || max_items == 0U) {
        return 0U;
    }

    size_t emitted = 0U;
    size_t token_budget = scheduler->max_batch_tokens;
    if (token_budget == 0U) {
        token_budget = 1U;
    }

    const size_t start_cursor = scheduler->rr_cursor;
    size_t scanned = 0U;

    while (scanned < VSPEC_SCHED_MAX_REQUESTS && emitted < max_items && token_budget > 0U) {
        size_t idx = (start_cursor + scanned) % VSPEC_SCHED_MAX_REQUESTS;
        VspecSchedRequest* slot = &scheduler->slots[idx];
        scanned += 1U;

        if (!slot->active || slot->finished) {
            continue;
        }

        if (slot->max_new_tokens > 0U && slot->generated_tokens >= slot->max_new_tokens) {
            slot->finished = 1U;
            continue;
        }

        size_t quota = scheduler->token_quantum;
        if (quota > token_budget) {
            quota = token_budget;
        }
        if (slot->max_new_tokens > 0U) {
            const size_t remaining = slot->max_new_tokens - slot->generated_tokens;
            if (quota > remaining) {
                quota = remaining;
            }
        }
        if (quota == 0U) {
            continue;
        }

        out_items[emitted].request_id = slot->request_id;
        out_items[emitted].token_quota = quota;
        emitted += 1U;
        token_budget -= quota;
        scheduler->rr_cursor = (idx + 1U) % VSPEC_SCHED_MAX_REQUESTS;
    }

    if (emitted > 0U) {
        scheduler->stats.total_scheduled_steps += 1U;
    }

    return emitted;
}

int vspec_request_scheduler_commit(
    VspecRequestScheduler* scheduler,
    uint64_t request_id,
    size_t generated_tokens,
    int reached_eos
) {
    VspecSchedRequest* slot = vspec_sched_find_by_id(scheduler, request_id);
    if (!slot || !slot->active) {
        return 0;
    }

    slot->generated_tokens += generated_tokens;
    if (reached_eos) {
        slot->finished = 1U;
    }
    if (slot->max_new_tokens > 0U && slot->generated_tokens >= slot->max_new_tokens) {
        slot->finished = 1U;
    }

    if (slot->finished) {
        (void)vspec_vram_release(&scheduler->budget, slot->reserved_bytes);
        scheduler->stats.completed_requests += 1U;
        vspec_sched_reset_slot(slot);
    }

    return 1;
}

int vspec_request_scheduler_cancel(
    VspecRequestScheduler* scheduler,
    uint64_t request_id
) {
    VspecSchedRequest* slot = vspec_sched_find_by_id(scheduler, request_id);
    if (!slot || !slot->active) {
        return 0;
    }

    (void)vspec_vram_release(&scheduler->budget, slot->reserved_bytes);
    scheduler->stats.cancelled_requests += 1U;
    vspec_sched_reset_slot(slot);
    return 1;
}

void vspec_request_scheduler_stats(
    const VspecRequestScheduler* scheduler,
    VspecSchedStats* out_stats
) {
    if (!scheduler || !out_stats) {
        return;
    }
    *out_stats = scheduler->stats;
    out_stats->active_requests = vspec_sched_count_active(scheduler);
    out_stats->queued_requests = vspec_sched_count_queued(scheduler);
    out_stats->reserved_vram_bytes = scheduler->budget.used_bytes;
}