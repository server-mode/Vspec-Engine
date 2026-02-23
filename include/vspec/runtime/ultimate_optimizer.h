#ifndef VSPEC_RUNTIME_ULTIMATE_OPTIMIZER_H
#define VSPEC_RUNTIME_ULTIMATE_OPTIMIZER_H

#include <stddef.h>

#include "vspec/quant/quant.h"
#include "vspec/runtime/hw_performance_manager.h"

typedef struct VspecRuntimeUltimateState {
    int enabled;
    int outlier_aware;
    int qlora_enabled;
    int tensor_core_preferred;
    float outlier_threshold;
    float quality_bias;
    unsigned int qlora_rank;

    float latest_outlier_ratio;
    VspecQuantType latest_recommended_quant;
    unsigned int medium_outlier_streak;
    unsigned int high_outlier_streak;
    unsigned int low_outlier_streak;
} VspecRuntimeUltimateState;

typedef struct VspecRuntimeUltimateReport {
    int enabled;
    int outlier_aware;
    int qlora_enabled;
    int tensor_core_preferred;
    float outlier_threshold;
    float quality_bias;
    unsigned int qlora_rank;
    float latest_outlier_ratio;
    VspecQuantType latest_recommended_quant;
    unsigned int medium_outlier_streak;
    unsigned int high_outlier_streak;
    unsigned int low_outlier_streak;
} VspecRuntimeUltimateReport;

void vspec_runtime_ultimate_init(
    VspecRuntimeUltimateState* state,
    const VspecRuntimeHwConfig* hw_config
);

VspecQuantType vspec_runtime_ultimate_recommend_quant(
    VspecRuntimeUltimateState* state,
    const float* input,
    size_t count
);

void vspec_runtime_ultimate_report(
    const VspecRuntimeUltimateState* state,
    VspecRuntimeUltimateReport* report
);

#endif