#include "vspec/runtime/mixed_bit_runtime.h"

uint8_t vspec_mixed_bit_enforce_sub4(uint8_t bits) {
    if (bits > 4U) {
        return 4U;
    }
    if (bits < 2U) {
        return 2U;
    }
    return bits;
}

void vspec_mixed_bit_runtime_init(VspecMixedBitRuntime* rt) {
    if (!rt) {
        return;
    }
    rt->layer_count = 0U;
    rt->last_lookup_layer_id = 0U;
    rt->last_lookup_type = VSPEC_LAYER_UNKNOWN;
    rt->last_lookup_bits = 0U;
    rt->has_last_lookup = 0;
}

void vspec_mixed_bit_runtime_add(VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t bits) {
    vspec_mixed_bit_runtime_set(rt, layer_id, type, bits);
}

void vspec_mixed_bit_runtime_set(VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t bits) {
    if (!rt || rt->layer_count >= VSPEC_MIXED_BIT_MAX_LAYERS) {
        if (!rt) {
            return;
        }
    }

    bits = vspec_mixed_bit_enforce_sub4(bits);

    for (size_t i = 0; i < rt->layer_count; ++i) {
        VspecLayerBitConfig* cfg = &rt->layers[i];
        if (cfg->layer_id == layer_id && cfg->type == type) {
            cfg->bits = bits;
            rt->last_lookup_layer_id = layer_id;
            rt->last_lookup_type = type;
            rt->last_lookup_bits = bits;
            rt->has_last_lookup = 1;
            return;
        }
    }

    if (rt->layer_count < VSPEC_MIXED_BIT_MAX_LAYERS) {
        VspecLayerBitConfig* cfg = &rt->layers[rt->layer_count++];
        cfg->layer_id = layer_id;
        cfg->type = type;
        cfg->bits = bits;
        rt->last_lookup_layer_id = layer_id;
        rt->last_lookup_type = type;
        rt->last_lookup_bits = bits;
        rt->has_last_lookup = 1;
    }
}

uint8_t vspec_mixed_bit_runtime_bits_for_layer(const VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type) {
    if (!rt) {
        return 4;
    }

    if (rt->has_last_lookup && rt->last_lookup_layer_id == layer_id && rt->last_lookup_type == type) {
        return rt->last_lookup_bits;
    }

    for (size_t i = 0; i < rt->layer_count; ++i) {
        const VspecLayerBitConfig* cfg = &rt->layers[i];
        if (cfg->layer_id == layer_id && cfg->type == type) {
            return cfg->bits;
        }
    }

    switch (type) {
        case VSPEC_LAYER_ATTENTION:
            return 4;
        case VSPEC_LAYER_ATTENTION_QK:
            return 4;
        case VSPEC_LAYER_ATTENTION_PROJ:
            return 4;
        case VSPEC_LAYER_MLP:
            return 4;
        case VSPEC_LAYER_EMBED:
            return 4;
        case VSPEC_LAYER_LM_HEAD:
            return 4;
        default:
            return 4;
    }
}

int vspec_mixed_bit_runtime_get(const VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t* out_bits) {
    if (!rt || !out_bits) {
        return 0;
    }

    if (rt->has_last_lookup && rt->last_lookup_layer_id == layer_id && rt->last_lookup_type == type) {
        *out_bits = vspec_mixed_bit_enforce_sub4(rt->last_lookup_bits);
        return 1;
    }

    for (size_t i = 0; i < rt->layer_count; ++i) {
        const VspecLayerBitConfig* cfg = &rt->layers[i];
        if (cfg->layer_id == layer_id && cfg->type == type) {
            *out_bits = vspec_mixed_bit_enforce_sub4(cfg->bits);
            return 1;
        }
    }

    return 0;
}
