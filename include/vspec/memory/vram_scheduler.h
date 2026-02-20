#ifndef VSPEC_MEMORY_VRAM_SCHEDULER_H
#define VSPEC_MEMORY_VRAM_SCHEDULER_H

#include <stddef.h>

typedef struct VspecVramBudget {
    size_t total_bytes;
    size_t used_bytes;
} VspecVramBudget;

void vspec_vram_budget_init(VspecVramBudget* budget, size_t total_bytes);
int vspec_vram_try_reserve(VspecVramBudget* budget, size_t bytes);
void vspec_vram_release(VspecVramBudget* budget, size_t bytes);

#endif
