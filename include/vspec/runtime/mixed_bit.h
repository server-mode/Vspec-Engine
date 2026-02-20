#ifndef VSPEC_RUNTIME_MIXED_BIT_H
#define VSPEC_RUNTIME_MIXED_BIT_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecMixedBitPlan {
    uint8_t bits;
    size_t group_size;
} VspecMixedBitPlan;

void vspec_mixed_bit_plan_init(VspecMixedBitPlan* plan);
VspecMixedBitPlan vspec_mixed_bit_plan_from_data(
    const float* data,
    size_t count
);

#endif
