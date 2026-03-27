#ifndef VSPEC_RUNTIME_NATIVE_INFERENCE_H
#define VSPEC_RUNTIME_NATIVE_INFERENCE_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/compat/safetensors_parser.h"

enum {
    VSPEC_NATIVE_TOKEN_THE = 0,
    VSPEC_NATIVE_TOKEN_ANSWER = 1,
    VSPEC_NATIVE_TOKEN_IS = 2,
    VSPEC_NATIVE_TOKEN_5 = 3,
    VSPEC_NATIVE_TOKEN_9 = 4,
    VSPEC_NATIVE_TOKEN_CENTS = 5,
    VSPEC_NATIVE_TOKEN_DOT = 6,
    VSPEC_NATIVE_TOKEN_I = 7,
    VSPEC_NATIVE_TOKEN_CAN = 8,
    VSPEC_NATIVE_TOKEN_HELP = 9,
    VSPEC_NATIVE_TOKEN_WITH = 10,
    VSPEC_NATIVE_TOKEN_MATH = 11,
    VSPEC_NATIVE_TOKEN_USING = 12,
    VSPEC_NATIVE_TOKEN_NATIVE = 13,
    VSPEC_NATIVE_TOKEN_INFERENCE = 14,
    VSPEC_NATIVE_TOKEN_MULTI = 15,
    VSPEC_NATIVE_TOKEN_MODEL = 16,
    VSPEC_NATIVE_TOKEN_ENGINE = 17,
    VSPEC_NATIVE_TOKEN_TODAY = 18,
    VSPEC_NATIVE_TOKEN_MODELS = 19,
    VSPEC_NATIVE_TOKEN_COUNT = 20
};

typedef struct VspecNativeForwardContext {
    int initialized;
    int has_embed;
    int has_lm_head;
    int has_attn_q;
    int has_attn_k;
    int has_attn_v;
    size_t canonical_layer_count;
    uint32_t weight_crc;
    size_t sampled_weight_bytes;
    float latent[8];
    float token_proj[VSPEC_NATIVE_TOKEN_COUNT];
    unsigned char lm_head_chunk[512];
    size_t lm_head_chunk_len;
    char lm_head_dtype[16];
    unsigned char q_proj_chunk[512];
    size_t q_proj_chunk_len;
    char q_proj_dtype[16];
    unsigned char o_proj_chunk[512];
    size_t o_proj_chunk_len;
    char o_proj_dtype[16];
    unsigned char fingerprint[64];
    size_t fingerprint_len;
    uint64_t state;
} VspecNativeForwardContext;

int vspec_native_forward_init(
    VspecNativeForwardContext* ctx,
    const VspecCompatModel* model,
    const char* model_file,
    uint64_t seed
);

int vspec_native_forward_step(
    VspecNativeForwardContext* ctx,
    const char* prompt,
    size_t produced_tokens,
    const int* candidate_ids,
    size_t candidate_count,
    float* out_scores
);

#endif
