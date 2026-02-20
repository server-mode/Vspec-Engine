#include <stddef.h>
#include <math.h>

#include "vspec/quant/dynamic_map.h"

void vspec_dynamic_quant_default(VspecDynamicQuantConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->min_bits = 2;
    cfg->max_bits = 4;
    cfg->group_size = 64;
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

    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float v = fabsf(data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }

    d.bits = cfg->max_bits;
    if (max_abs < 0.25f && cfg->min_bits <= 2) {
        d.bits = 2;
    } else if (max_abs < 1.0f && cfg->min_bits <= 3) {
        d.bits = 3;
    }

    const float qmax = (float)((1 << (d.bits - 1)) - 1);
    d.scale = (max_abs > 0.0f) ? (max_abs / qmax) : 1.0f;
    return d;
}
