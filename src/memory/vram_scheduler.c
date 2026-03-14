#include "vspec/memory/vram_scheduler.h"

void vspec_vram_budget_init(VspecVramBudget* budget, size_t total_bytes) {
    if (!budget) {
        return;
    }
    budget->total_bytes = total_bytes;
    budget->used_bytes = 0U;
    budget->peak_used_bytes = 0U;
    budget->reserve_fail_count = 0U;
}

int vspec_vram_try_reserve(VspecVramBudget* budget, size_t bytes) {
    if (!budget || bytes == 0U) {
        return 0;
    }

    if (bytes > budget->total_bytes || budget->used_bytes > budget->total_bytes - bytes) {
        budget->reserve_fail_count += 1U;
        return 0;
    }

    budget->used_bytes += bytes;
    if (budget->used_bytes > budget->peak_used_bytes) {
        budget->peak_used_bytes = budget->used_bytes;
    }
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

size_t vspec_vram_available(const VspecVramBudget* budget) {
    if (!budget || budget->used_bytes >= budget->total_bytes) {
        return 0U;
    }
    return budget->total_bytes - budget->used_bytes;
}

float vspec_vram_utilization(const VspecVramBudget* budget) {
    if (!budget || budget->total_bytes == 0U) {
        return 0.0f;
    }
    return (float)budget->used_bytes / (float)budget->total_bytes;
}
