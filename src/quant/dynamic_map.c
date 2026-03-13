#include <stddef.h>
#include <math.h>
#include <stdlib.h>

#include "vspec/quant/dynamic_map.h"

static int vspec_cmp_float(const void* lhs, const void* rhs) {
    const float a = *(const float*)lhs;
    const float b = *(const float*)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static float vspec_percentile_abs(
    const float* data,
    size_t count,
    float percentile
) {
    if (!data || count == 0U) {
        return 0.0f;
    }

    if (percentile < 50.0f) {
        percentile = 50.0f;
    }
    if (percentile > 100.0f) {
        percentile = 100.0f;
    }

    float* abs_vals = (float*)malloc(count * sizeof(float));
    if (!abs_vals) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            float v = fabsf(data[i]);
            if (v > max_abs) {
                max_abs = v;
            }
        }
        return max_abs;
    }

    for (size_t i = 0; i < count; ++i) {
        abs_vals[i] = fabsf(data[i]);
    }
    qsort(abs_vals, count, sizeof(float), vspec_cmp_float);

    const float p = percentile / 100.0f;
    size_t idx = (size_t)((double)(count - 1U) * (double)p);
    if (idx >= count) {
        idx = count - 1U;
    }
    float out = abs_vals[idx];
    free(abs_vals);
    return out;
}

void vspec_dynamic_quant_default(VspecDynamicQuantConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->min_bits = 2;
    cfg->max_bits = 4;
    cfg->group_size = 32;
    cfg->percentile = 99.5f;
}

VspecDynamicQuantDecision vspec_dynamic_quant_decide(
    const float* data,
    size_t count,
    const VspecDynamicQuantConfig* cfg
) {
    VspecDynamicQuantDecision d = {4, 1.0f};
    if (!data || count == 0 || !cfg) {
        return d;
    }

    size_t group_size = cfg->group_size;
    if (group_size == 0U) {
        group_size = 32U;
    }

    float representative_abs = 0.0f;
    for (size_t base = 0U; base < count; base += group_size) {
        size_t len = group_size;
        if (base + len > count) {
            len = count - base;
        }
        float p_abs = vspec_percentile_abs(data + base, len, cfg->percentile);
        if (p_abs > representative_abs) {
            representative_abs = p_abs;
        }
    }

    if (representative_abs <= 0.0f) {
        representative_abs = vspec_percentile_abs(data, count, cfg->percentile);
    }

    const float global_abs = vspec_percentile_abs(data, count, cfg->percentile);
    const float denom = (global_abs > 1e-6f) ? global_abs : 1e-6f;
    const float relative_strength = representative_abs / denom;

    d.bits = cfg->max_bits;
    if (relative_strength < 0.18f && cfg->min_bits <= 2) {
        d.bits = 2;
    } else if (relative_strength < 0.55f && cfg->min_bits <= 3) {
        d.bits = 3;
    }

    const float qmax = (float)((1 << (d.bits - 1)) - 1);
    d.scale = (representative_abs > 0.0f) ? (representative_abs / qmax) : 1.0f;
    return d;
}
