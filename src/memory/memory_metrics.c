#include "vspec/memory/memory_metrics.h"

void vspec_memory_metrics_reset(VspecMemoryMetrics* metrics) {
    if (!metrics) {
        return;
    }
    metrics->weight_bytes = 0U;
    metrics->activation_bytes = 0U;
    metrics->kv_bytes = 0U;
    metrics->scratch_bytes = 0U;
}

void vspec_memory_metrics_add(VspecMemoryMetrics* metrics, size_t weight, size_t act, size_t kv, size_t scratch) {
    if (!metrics) {
        return;
    }
    metrics->weight_bytes += weight;
    metrics->activation_bytes += act;
    metrics->kv_bytes += kv;
    metrics->scratch_bytes += scratch;
}

size_t vspec_memory_metrics_total(const VspecMemoryMetrics* metrics) {
    if (!metrics) {
        return 0U;
    }
    return metrics->weight_bytes + metrics->activation_bytes + metrics->kv_bytes + metrics->scratch_bytes;
}

float vspec_memory_metrics_pressure(const VspecMemoryMetrics* metrics, size_t budget_bytes) {
    if (!metrics || budget_bytes == 0U) {
        return 0.0f;
    }
    return (float)vspec_memory_metrics_total(metrics) / (float)budget_bytes;
}
