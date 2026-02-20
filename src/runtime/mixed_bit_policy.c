#include "vspec/runtime/mixed_bit_policy.h"

static float vspec_pressure_from_metrics(
    const VspecMemoryMetrics* metrics,
    const VspecVramBudget* budget,
    size_t target_bytes
) {
    if (budget && budget->total_bytes > 0U) {
        return (float)budget->used_bytes / (float)budget->total_bytes;
    }
    if (metrics && target_bytes > 0U) {
        size_t used = metrics->weight_bytes + metrics->activation_bytes + metrics->kv_bytes + metrics->scratch_bytes;
        return (float)used / (float)target_bytes;
    }
    return 0.0f;
}

static uint8_t vspec_base_bits_from_policy(const VspecMixedBitPolicy* policy, VspecLayerType type) {
    if (!policy) {
        switch (type) {
            case VSPEC_LAYER_ATTENTION:
                return 4;
            case VSPEC_LAYER_MLP:
                return 3;
            case VSPEC_LAYER_EMBED:
                return 8;
            default:
                return 4;
        }
    }

    switch (type) {
        case VSPEC_LAYER_ATTENTION:
            return policy->attention_bits;
        case VSPEC_LAYER_MLP:
            return policy->mlp_bits;
        case VSPEC_LAYER_EMBED:
            return policy->embed_bits;
        default:
            return policy->attention_bits;
    }
}

static uint8_t vspec_clamp_bits(uint8_t bits, uint8_t min_bits, uint8_t max_bits) {
    if (bits < min_bits) {
        return min_bits;
    }
    if (bits > max_bits) {
        return max_bits;
    }
    return bits;
}

void vspec_mixed_bit_policy_default(VspecMixedBitPolicy* policy) {
    if (!policy) {
        return;
    }
    policy->attention_bits = 4;
    policy->mlp_bits = 3;
    policy->embed_bits = 8;
    policy->min_bits = 2;
    policy->max_bits = 8;
    policy->downshift_step = 1;
    policy->pressure_high = 0.80f;
    policy->pressure_critical = 0.92f;
    policy->memory_target_bytes = 0U;
    vspec_dynamic_quant_default(&policy->dyn_cfg);
}

uint8_t vspec_mixed_bit_select_bits(
    const VspecMixedBitRuntime* runtime,
    const VspecMixedBitPolicy* policy,
    uint32_t layer_id,
    VspecLayerType type,
    const float* data,
    size_t count,
    const VspecMemoryMetrics* metrics,
    const VspecVramBudget* budget
) {
    uint8_t override_bits = 0U;
    if (vspec_mixed_bit_runtime_get(runtime, layer_id, type, &override_bits)) {
        return override_bits;
    }

    uint8_t bits = vspec_base_bits_from_policy(policy, type);
    uint8_t min_bits = policy ? policy->min_bits : 2;
    uint8_t max_bits = policy ? policy->max_bits : 8;

    if (data && count > 0U) {
        VspecDynamicQuantConfig cfg = policy ? policy->dyn_cfg : (VspecDynamicQuantConfig){2, 4, 64};
        if (cfg.min_bits < min_bits) {
            cfg.min_bits = min_bits;
        }
        if (cfg.max_bits > max_bits) {
            cfg.max_bits = max_bits;
        }
        VspecDynamicQuantDecision decision = vspec_dynamic_quant_decide(data, count, &cfg);
        bits = decision.bits;
    }

    bits = vspec_clamp_bits(bits, min_bits, max_bits);

    if (policy) {
        float pressure = vspec_pressure_from_metrics(metrics, budget, policy->memory_target_bytes);
        if (pressure >= policy->pressure_critical) {
            if (bits > policy->downshift_step * 2U) {
                bits -= (uint8_t)(policy->downshift_step * 2U);
            } else {
                bits = min_bits;
            }
        } else if (pressure >= policy->pressure_high) {
            if (bits > policy->downshift_step) {
                bits -= policy->downshift_step;
            } else {
                bits = min_bits;
            }
        }
    }

    return vspec_clamp_bits(bits, min_bits, max_bits);
}
