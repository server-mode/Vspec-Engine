#include "vspec/memory/vram_scheduler.h"

void vspec_vram_budget_init(VspecVramBudget* budget, size_t total_bytes) {
    if (!budget) {
        return;
    }
    budget->total_bytes = total_bytes;
    budget->used_bytes = 0U;
}

int vspec_vram_try_reserve(VspecVramBudget* budget, size_t bytes) {
    if (!budget || bytes == 0U) {
        return 0;
    }
    if (budget->used_bytes + bytes > budget->total_bytes) {
        return 0;
    }
    budget->used_bytes += bytes;
    return 1;
}

void vspec_vram_release(VspecVramBudget* budget, size_t bytes) {
    if (!budget) {
        return;
    }
    if (bytes > budget->used_bytes) {
        budget->used_bytes = 0U;
        return;
    }
    budget->used_bytes -= bytes;
}
