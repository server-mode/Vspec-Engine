#ifndef VSPEC_SCHEDULER_DYNAMIC_H
#define VSPEC_SCHEDULER_DYNAMIC_H

#include <stddef.h>
#include "vspec/memory/vram_scheduler.h"

typedef struct VspecScheduleRequest {
    size_t bytes;
    size_t tokens;
} VspecScheduleRequest;

typedef struct VspecDynamicScheduler {
    VspecVramBudget budget;
    size_t max_active;
    size_t active_count;
} VspecDynamicScheduler;

void vspec_dynamic_scheduler_init(VspecDynamicScheduler* sched, size_t total_vram, size_t max_active);
int vspec_dynamic_scheduler_try_schedule(VspecDynamicScheduler* sched, const VspecScheduleRequest* req);
void vspec_dynamic_scheduler_finish(VspecDynamicScheduler* sched, const VspecScheduleRequest* req);

#endif
