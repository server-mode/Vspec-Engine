#ifndef VSPEC_RUNTIME_QLORA_ADAPTER_H
#define VSPEC_RUNTIME_QLORA_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecQloraLayerConfig {
    uint32_t layer_id;
    size_t in_dim;
    size_t rank;
    size_t out_dim;
    float alpha;
} VspecQloraLayerConfig;

int vspec_qlora_adapter_add_layer(
    uint32_t layer_id,
    size_t in_dim,
    size_t rank,
    size_t out_dim,
    float alpha,
    const float* matrix_a,
    const float* matrix_b
);

int vspec_qlora_adapter_load_file(const char* path);
int vspec_qlora_adapter_load_manifest_json(const char* manifest_path);
void vspec_qlora_adapter_clear(void);
int vspec_qlora_adapter_has_layer(uint32_t layer_id);

void vspec_qlora_adapter_apply_layer_f32(
    uint32_t layer_id,
    const float* input,
    size_t m,
    size_t k,
    size_t n,
    float* output
);

#endif