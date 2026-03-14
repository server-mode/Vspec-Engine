#ifndef VSPEC_PARALLEL_MULTI_GPU_H
#define VSPEC_PARALLEL_MULTI_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecMultiGpuPlan {
    uint32_t device_count;
    uint32_t tensor_parallel;
    uint32_t pipeline_parallel;
    uint32_t total_layers;
    uint32_t stage_layer_start[16];
    uint32_t stage_layer_end[16];
} VspecMultiGpuPlan;

void vspec_multi_gpu_plan_init(VspecMultiGpuPlan* plan, uint32_t device_count);
int vspec_multi_gpu_plan_build(
    VspecMultiGpuPlan* plan,
    uint32_t device_count,
    uint32_t total_layers,
    uint32_t max_tensor_parallel
);
uint32_t vspec_multi_gpu_plan_stage_for_layer(const VspecMultiGpuPlan* plan, uint32_t layer_id);
uint32_t vspec_multi_gpu_plan_device_for_layer(const VspecMultiGpuPlan* plan, uint32_t layer_id);
uint32_t vspec_multi_gpu_plan_shard_width(const VspecMultiGpuPlan* plan, uint32_t hidden_size);

#endif
