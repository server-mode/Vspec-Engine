#ifndef VSPEC_PARALLEL_MULTI_GPU_H
#define VSPEC_PARALLEL_MULTI_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecMultiGpuPlan {
    uint32_t device_count;
    uint32_t tensor_parallel;
    uint32_t pipeline_parallel;
} VspecMultiGpuPlan;

void vspec_multi_gpu_plan_init(VspecMultiGpuPlan* plan, uint32_t device_count);

#endif
