#include <stdio.h>

#include "vspec/memory/memory_metrics.h"
#include "vspec/memory/vram_scheduler.h"
#include "vspec/parallel/multi_gpu.h"
#include "vspec/runtime/mixed_bit_policy.h"

int main(void) {
    VspecMixedBitRuntime rt;
    vspec_mixed_bit_runtime_init(&rt);
    vspec_mixed_bit_runtime_add(&rt, 0, VSPEC_LAYER_ATTENTION, 4);
    vspec_mixed_bit_runtime_add(&rt, 2, VSPEC_LAYER_EMBED, 8);

    VspecMixedBitPolicy policy;
    vspec_mixed_bit_policy_default(&policy);
    policy.memory_target_bytes = 1024 * 1024 * 8;

    float sample[8] = {0.05f, 0.08f, 0.12f, 0.2f, 0.15f, 0.18f, 0.22f, 0.1f};

    VspecMemoryMetrics metrics;
    vspec_memory_metrics_reset(&metrics);
    vspec_memory_metrics_add(&metrics, 4 * 1024 * 1024, 2 * 1024 * 1024, 1 * 1024 * 1024, 512 * 1024);

    VspecVramBudget budget;
    vspec_vram_budget_init(&budget, 16 * 1024 * 1024);
    vspec_vram_try_reserve(&budget, 14 * 1024 * 1024);

    printf("layer0 bits=%u\n", (unsigned)vspec_mixed_bit_select_bits(
        &rt, &policy, 0, VSPEC_LAYER_ATTENTION, sample, 8, &metrics, &budget));
    printf("layer1 bits=%u\n", (unsigned)vspec_mixed_bit_select_bits(
        &rt, &policy, 1, VSPEC_LAYER_MLP, sample, 8, &metrics, &budget));
    printf("layer2 bits=%u\n", (unsigned)vspec_mixed_bit_select_bits(
        &rt, &policy, 2, VSPEC_LAYER_EMBED, sample, 8, &metrics, &budget));

    VspecMultiGpuPlan plan;
    vspec_multi_gpu_plan_init(&plan, 4);
    printf("multi-gpu devices=%u tensor=%u pipeline=%u\n",
        plan.device_count, plan.tensor_parallel, plan.pipeline_parallel);

    return 0;
}
