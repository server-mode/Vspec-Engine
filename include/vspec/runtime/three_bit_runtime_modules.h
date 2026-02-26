#ifndef VSPEC_RUNTIME_THREE_BIT_RUNTIME_MODULES_H
#define VSPEC_RUNTIME_THREE_BIT_RUNTIME_MODULES_H

#include <stddef.h>
#include <stdint.h>

typedef struct Vspec3BitSoftmaxManager {
    int enabled;
    float logit_clip;
    float temperature_floor;
    float min_denom;
} Vspec3BitSoftmaxManager;

typedef struct Vspec3BitAccumManager {
    int enabled;
    float compensation_limit;
    size_t pairwise_block;
} Vspec3BitAccumManager;

typedef struct Vspec3BitNoiseReducer {
    int enabled;
    float input_clip;
    float smooth_alpha;
    float outlier_threshold;
    float activation_clamp_alpha;
} Vspec3BitNoiseReducer;

typedef struct Vspec3BitAccumState {
    double sum;
    double compensation;
} Vspec3BitAccumState;

typedef struct Vspec3BitAttentionManager {
    int enabled;
    uint8_t qk_compute_bits;
    uint8_t output_projection_bits;
    uint8_t mlp_compute_bits;
    float qk_scale_min;
    float qk_scale_max;
    float output_clip;
    Vspec3BitSoftmaxManager softmax;
    Vspec3BitAccumManager accum;
    Vspec3BitNoiseReducer noise;
} Vspec3BitAttentionManager;

int vspec_runtime_3bit_enabled(void);
uint8_t vspec_runtime_3bit_bits_for_component(const char* component_name);
size_t vspec_3bit_resolve_block_size(size_t requested);

void vspec_3bit_softmax_manager_default(Vspec3BitSoftmaxManager* manager);
void vspec_3bit_accum_manager_default(Vspec3BitAccumManager* manager);
void vspec_3bit_noise_reducer_default(Vspec3BitNoiseReducer* reducer);
void vspec_3bit_attention_manager_default(Vspec3BitAttentionManager* manager);

void vspec_3bit_noise_reduce_vector(
    const Vspec3BitNoiseReducer* reducer,
    const float* input,
    size_t n,
    float* output
);

void vspec_3bit_softmax_apply(
    const Vspec3BitSoftmaxManager* manager,
    const float* logits,
    size_t n,
    float* probs
);

void vspec_3bit_accum_reset(Vspec3BitAccumState* state);
void vspec_3bit_accum_add(
    const Vspec3BitAccumManager* manager,
    Vspec3BitAccumState* state,
    float value
);
float vspec_3bit_accum_value(const Vspec3BitAccumState* state);
float vspec_3bit_accum_dot_f32(
    const Vspec3BitAccumManager* manager,
    const float* a,
    const float* b,
    size_t n
);

void vspec_3bit_dynamic_clamp_std(
    const float* input,
    size_t n,
    float alpha,
    float* output
);

float vspec_3bit_attention_qk_score(
    const Vspec3BitAttentionManager* manager,
    const float* query,
    const float* key,
    size_t head_dim,
    float inv_sqrt_d
);

void vspec_3bit_attention_output_projection(
    const Vspec3BitAttentionManager* manager,
    const float* input,
    const float* weight,
    const float* bias,
    size_t in_dim,
    size_t out_dim,
    float* output
);

#endif
