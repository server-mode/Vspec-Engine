#ifndef VSPEC_MEMORY_METRICS_H
#define VSPEC_MEMORY_METRICS_H

#include <stddef.h>

typedef struct VspecMemoryMetrics {
    size_t weight_bytes;
    size_t activation_bytes;
    size_t kv_bytes;
    size_t scratch_bytes;
} VspecMemoryMetrics;

void vspec_memory_metrics_reset(VspecMemoryMetrics* metrics);
void vspec_memory_metrics_add(VspecMemoryMetrics* metrics, size_t weight, size_t act, size_t kv, size_t scratch);

#endif
