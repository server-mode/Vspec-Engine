#ifndef VSPEC_QUANT_DYNAMIC_MAP_H
#define VSPEC_QUANT_DYNAMIC_MAP_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecDynamicQuantConfig {
    uint8_t min_bits;
    uint8_t max_bits;
    size_t group_size;
} VspecDynamicQuantConfig;

typedef struct VspecDynamicQuantDecision {
    uint8_t bits;
    float scale;
} VspecDynamicQuantDecision;

void vspec_dynamic_quant_default(VspecDynamicQuantConfig* cfg);
VspecDynamicQuantDecision vspec_dynamic_quant_decide(
    const float* data,
    size_t count,
    const VspecDynamicQuantConfig* cfg
);

#endif
