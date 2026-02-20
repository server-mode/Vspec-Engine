#include "vspec/scheduler/dynamic.h"

void vspec_dynamic_scheduler_init(VspecDynamicScheduler* sched, size_t total_vram, size_t max_active) {
    if (!sched) {
        return;
    }
    vspec_vram_budget_init(&sched->budget, total_vram);
    sched->max_active = max_active;
    sched->active_count = 0U;
}

int vspec_dynamic_scheduler_try_schedule(VspecDynamicScheduler* sched, const VspecScheduleRequest* req) {
    if (!sched || !req) {
        return 0;
    }
    if (sched->active_count >= sched->max_active) {
        return 0;
    }
    if (!vspec_vram_try_reserve(&sched->budget, req->bytes)) {
        return 0;
    }

    sched->active_count += 1U;
    return 1;
}

void vspec_dynamic_scheduler_finish(VspecDynamicScheduler* sched, const VspecScheduleRequest* req) {
    if (!sched || !req) {
        return;
    }
    if (sched->active_count > 0U) {
        sched->active_count -= 1U;
    }
    vspec_vram_release(&sched->budget, req->bytes);
}
