#include "vspec/runtime/mixed_bit_runtime.h"

void vspec_mixed_bit_runtime_init(VspecMixedBitRuntime* rt) {
    if (!rt) {
        return;
    }
    rt->layer_count = 0U;
}

void vspec_mixed_bit_runtime_add(VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t bits) {
    if (!rt || rt->layer_count >= VSPEC_MIXED_BIT_MAX_LAYERS) {
        return;
    }
    VspecLayerBitConfig* cfg = &rt->layers[rt->layer_count++];
    cfg->layer_id = layer_id;
    cfg->type = type;
    cfg->bits = bits;
}

uint8_t vspec_mixed_bit_runtime_bits_for_layer(const VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type) {
    if (!rt) {
        return 4;
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
        case VSPEC_LAYER_MLP:
            return 3;
        case VSPEC_LAYER_EMBED:
            return 8;
        default:
            return 4;
    }
}

int vspec_mixed_bit_runtime_get(const VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t* out_bits) {
    if (!rt || !out_bits) {
        return 0;
    }

    for (size_t i = 0; i < rt->layer_count; ++i) {
        const VspecLayerBitConfig* cfg = &rt->layers[i];
        if (cfg->layer_id == layer_id && cfg->type == type) {
            *out_bits = cfg->bits;
            return 1;
        }
    }

    return 0;
}
