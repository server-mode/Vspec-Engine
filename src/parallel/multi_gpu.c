#include "vspec/parallel/multi_gpu.h"

void vspec_multi_gpu_plan_init(VspecMultiGpuPlan* plan, uint32_t device_count) {
    if (!plan) {
        return;
    }
    plan->device_count = device_count;
    plan->tensor_parallel = device_count > 1 ? 2 : 1;
    plan->pipeline_parallel = 1;
}
