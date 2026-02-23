#include "vspec/runtime/ultimate_optimizer.h"

#include <math.h>
#include <string.h>

static float vspec_clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

void vspec_runtime_ultimate_init(
    VspecRuntimeUltimateState* state,
    const VspecRuntimeHwConfig* hw_config
) {
    if (!state) {
        return;
    }

    (void)memset(state, 0, sizeof(*state));

    if (!hw_config) {
        state->enabled = 0;
        state->outlier_threshold = 6.0f;
        state->quality_bias = 0.75f;
        state->qlora_rank = 16U;
        state->latest_recommended_quant = VSPEC_QUANT_INT3;
        return;
    }

    state->enabled = hw_config->enable_ultimate_mode ? 1 : 0;
    state->outlier_aware = hw_config->enable_outlier_aware ? 1 : 0;
    state->qlora_enabled = hw_config->enable_qlora_adapter ? 1 : 0;
    state->tensor_core_preferred = hw_config->prefer_tensor_core ? 1 : 0;
    state->outlier_threshold = (hw_config->outlier_threshold > 0.0f) ? hw_config->outlier_threshold : 6.0f;
    state->quality_bias = vspec_clamp01(hw_config->quality_bias);
    state->qlora_rank = hw_config->qlora_rank;
    state->latest_recommended_quant = VSPEC_QUANT_INT4;
}

VspecQuantType vspec_runtime_ultimate_recommend_quant(
    VspecRuntimeUltimateState* state,
    const float* input,
    size_t count
) {
    if (!state || !state->enabled) {
        return VSPEC_QUANT_INT3;
    }

    if (!input || count == 0U) {
        state->latest_outlier_ratio = 0.0f;
        state->latest_recommended_quant = (state->quality_bias >= 0.70f) ? VSPEC_QUANT_INT4 : VSPEC_QUANT_INT3;
        return state->latest_recommended_quant;
    }

    size_t outlier_count = 0U;
    if (state->outlier_aware) {
        for (size_t i = 0; i < count; ++i) {
            if (fabsf(input[i]) > state->outlier_threshold) {
                outlier_count += 1U;
            }
        }
    }

    state->latest_outlier_ratio = (float)outlier_count / (float)count;

    if (state->latest_outlier_ratio >= 0.03f) {
        state->high_outlier_streak += 1U;
        state->medium_outlier_streak = 0U;
        state->low_outlier_streak = 0U;
    } else if (state->latest_outlier_ratio >= 0.01f) {
        state->medium_outlier_streak += 1U;
        state->high_outlier_streak = 0U;
        state->low_outlier_streak = 0U;
    } else {
        state->low_outlier_streak += 1U;
        state->high_outlier_streak = 0U;
        state->medium_outlier_streak = 0U;
    }

    if (state->high_outlier_streak >= 3U) {
        state->latest_recommended_quant = VSPEC_QUANT_NONE;
        return state->latest_recommended_quant;
    }

    if (state->medium_outlier_streak >= 2U || state->latest_outlier_ratio >= 0.01f || state->quality_bias >= 0.85f) {
        state->latest_recommended_quant = VSPEC_QUANT_INT4;
    } else {
        state->latest_recommended_quant = VSPEC_QUANT_INT3;
    }

    if (state->low_outlier_streak >= 5U && state->quality_bias < 0.80f) {
        state->latest_recommended_quant = VSPEC_QUANT_INT3;
    }

    return state->latest_recommended_quant;
}

void vspec_runtime_ultimate_report(
    const VspecRuntimeUltimateState* state,
    VspecRuntimeUltimateReport* report
) {
    if (!state || !report) {
        return;
    }

    report->enabled = state->enabled;
    report->outlier_aware = state->outlier_aware;
    report->qlora_enabled = state->qlora_enabled;
    report->tensor_core_preferred = state->tensor_core_preferred;
    report->outlier_threshold = state->outlier_threshold;
    report->quality_bias = state->quality_bias;
    report->qlora_rank = state->qlora_rank;
    report->latest_outlier_ratio = state->latest_outlier_ratio;
    report->latest_recommended_quant = state->latest_recommended_quant;
    report->medium_outlier_streak = state->medium_outlier_streak;
    report->high_outlier_streak = state->high_outlier_streak;
    report->low_outlier_streak = state->low_outlier_streak;
}