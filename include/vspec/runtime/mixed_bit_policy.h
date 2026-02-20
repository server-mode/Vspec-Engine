#ifndef VSPEC_RUNTIME_MIXED_BIT_POLICY_H
#define VSPEC_RUNTIME_MIXED_BIT_POLICY_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/memory/memory_metrics.h"
#include "vspec/memory/vram_scheduler.h"
#include "vspec/quant/dynamic_map.h"
#include "vspec/runtime/mixed_bit_runtime.h"

typedef struct VspecMixedBitPolicy {
    uint8_t attention_bits;
    uint8_t mlp_bits;
    uint8_t embed_bits;
    uint8_t min_bits;
    uint8_t max_bits;
    uint8_t downshift_step;
    float pressure_high;
    float pressure_critical;
    size_t memory_target_bytes;
    VspecDynamicQuantConfig dyn_cfg;
} VspecMixedBitPolicy;

void vspec_mixed_bit_policy_default(VspecMixedBitPolicy* policy);
uint8_t vspec_mixed_bit_select_bits(
    const VspecMixedBitRuntime* runtime,
    const VspecMixedBitPolicy* policy,
    uint32_t layer_id,
    VspecLayerType type,
    const float* data,
    size_t count,
    const VspecMemoryMetrics* metrics,
    const VspecVramBudget* budget
);

#endif
