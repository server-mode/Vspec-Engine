#include "vspec/parallel/multi_gpu.h"

static uint32_t clamp_u32(uint32_t value, uint32_t min_v, uint32_t max_v) {
    if (value < min_v) {
        return min_v;
    }
    if (value > max_v) {
        return max_v;
    }
    return value;
}

static uint32_t choose_tensor_parallel(uint32_t device_count, uint32_t max_tensor_parallel) {
    if (device_count == 0U) {
        return 1U;
    }
    uint32_t max_tp = clamp_u32(max_tensor_parallel, 1U, 16U);
    if (max_tp > device_count) {
        max_tp = device_count;
    }

    uint32_t best = 1U;
    for (uint32_t tp = 1U; tp <= max_tp; ++tp) {
        if ((device_count % tp) == 0U) {
            best = tp;
        }
    }
    return best;
}

static void clear_plan_ranges(VspecMultiGpuPlan* plan) {
    if (!plan) {
        return;
    }
    for (size_t i = 0U; i < 16U; ++i) {
        plan->stage_layer_start[i] = 0U;
        plan->stage_layer_end[i] = 0U;
    }
}

void vspec_multi_gpu_plan_init(VspecMultiGpuPlan* plan, uint32_t device_count) {
    if (!plan) {
        return;
    }
    (void)vspec_multi_gpu_plan_build(plan, device_count, 0U, 2U);
}

int vspec_multi_gpu_plan_build(
    VspecMultiGpuPlan* plan,
    uint32_t device_count,
    uint32_t total_layers,
    uint32_t max_tensor_parallel
) {
    if (!plan) {
        return 0;
    }

    if (device_count == 0U) {
        device_count = 1U;
    }
    if (device_count > 16U) {
        device_count = 16U;
    }

    clear_plan_ranges(plan);
    plan->device_count = device_count;
    plan->total_layers = total_layers;
    plan->tensor_parallel = choose_tensor_parallel(device_count, max_tensor_parallel);
    if (plan->tensor_parallel == 0U) {
        plan->tensor_parallel = 1U;
    }
    plan->pipeline_parallel = device_count / plan->tensor_parallel;
    if (plan->pipeline_parallel == 0U) {
        plan->pipeline_parallel = 1U;
    }

    if (total_layers == 0U) {
        return 1;
    }

    const uint32_t stages = plan->pipeline_parallel;
    const uint32_t base = total_layers / stages;
    const uint32_t rem = total_layers % stages;

    uint32_t cursor = 0U;
    for (uint32_t s = 0U; s < stages && s < 16U; ++s) {
        uint32_t span = base + ((s < rem) ? 1U : 0U);
        plan->stage_layer_start[s] = cursor;
        plan->stage_layer_end[s] = (span == 0U) ? cursor : (cursor + span - 1U);
        cursor += span;
    }
    return 1;
}

uint32_t vspec_multi_gpu_plan_stage_for_layer(const VspecMultiGpuPlan* plan, uint32_t layer_id) {
    if (!plan || plan->pipeline_parallel == 0U) {
        return 0U;
    }

    const uint32_t max_stage = (plan->pipeline_parallel > 16U) ? 16U : plan->pipeline_parallel;
    for (uint32_t s = 0U; s < max_stage; ++s) {
        if (layer_id >= plan->stage_layer_start[s] && layer_id <= plan->stage_layer_end[s]) {
            return s;
        }
    }

    if (plan->total_layers == 0U) {
        return 0U;
    }
    return (layer_id >= plan->total_layers) ? (max_stage - 1U) : 0U;
}

uint32_t vspec_multi_gpu_plan_device_for_layer(const VspecMultiGpuPlan* plan, uint32_t layer_id) {
    if (!plan || plan->device_count == 0U) {
        return 0U;
    }
    const uint32_t stage = vspec_multi_gpu_plan_stage_for_layer(plan, layer_id);
    const uint32_t tp = (plan->tensor_parallel == 0U) ? 1U : plan->tensor_parallel;
    const uint32_t device = stage * tp;
    if (device >= plan->device_count) {
        return plan->device_count - 1U;
    }
    return device;
}

uint32_t vspec_multi_gpu_plan_shard_width(const VspecMultiGpuPlan* plan, uint32_t hidden_size) {
    if (!plan || hidden_size == 0U) {
        return 0U;
    }
    const uint32_t tp = (plan->tensor_parallel == 0U) ? 1U : plan->tensor_parallel;
    return (hidden_size + tp - 1U) / tp;
}
