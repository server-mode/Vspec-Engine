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
