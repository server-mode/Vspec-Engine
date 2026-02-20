#include "vspec/runtime/mixed_bit.h"
#include "vspec/quant/dynamic_map.h"

void vspec_mixed_bit_plan_init(VspecMixedBitPlan* plan) {
    if (!plan) {
        return;
    }
    plan->bits = 4;
    plan->group_size = 64;
}

VspecMixedBitPlan vspec_mixed_bit_plan_from_data(
    const float* data,
    size_t count
) {
    VspecMixedBitPlan plan;
    vspec_mixed_bit_plan_init(&plan);

    VspecDynamicQuantConfig cfg;
    vspec_dynamic_quant_default(&cfg);
    VspecDynamicQuantDecision d = vspec_dynamic_quant_decide(data, count, &cfg);

    plan.bits = d.bits;
    plan.group_size = cfg.group_size;
    return plan;
}
