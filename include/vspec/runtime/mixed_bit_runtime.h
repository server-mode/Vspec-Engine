#ifndef VSPEC_RUNTIME_MIXED_BIT_RUNTIME_H
#define VSPEC_RUNTIME_MIXED_BIT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define VSPEC_MIXED_BIT_MAX_LAYERS 128

typedef enum VspecLayerType {
    VSPEC_LAYER_UNKNOWN = 0,
    VSPEC_LAYER_ATTENTION = 1,
    VSPEC_LAYER_MLP = 2,
    VSPEC_LAYER_EMBED = 3
} VspecLayerType;

typedef struct VspecLayerBitConfig {
    uint32_t layer_id;
    VspecLayerType type;
    uint8_t bits;
} VspecLayerBitConfig;

typedef struct VspecMixedBitRuntime {
    VspecLayerBitConfig layers[VSPEC_MIXED_BIT_MAX_LAYERS];
    size_t layer_count;
    uint32_t last_lookup_layer_id;
    VspecLayerType last_lookup_type;
    uint8_t last_lookup_bits;
    int has_last_lookup;
} VspecMixedBitRuntime;

void vspec_mixed_bit_runtime_init(VspecMixedBitRuntime* rt);
void vspec_mixed_bit_runtime_add(VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t bits);
void vspec_mixed_bit_runtime_set(VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t bits);
uint8_t vspec_mixed_bit_runtime_bits_for_layer(const VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type);
int vspec_mixed_bit_runtime_get(const VspecMixedBitRuntime* rt, uint32_t layer_id, VspecLayerType type, uint8_t* out_bits);
uint8_t vspec_mixed_bit_enforce_sub4(uint8_t bits);

#endif
