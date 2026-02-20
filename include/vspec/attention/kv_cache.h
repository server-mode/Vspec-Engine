#ifndef VSPEC_ATTENTION_KV_CACHE_H
#define VSPEC_ATTENTION_KV_CACHE_H

#include <stddef.h>

typedef struct VspecKVCache {
    float* key;
    float* value;
    size_t max_tokens;
    size_t num_heads;
    size_t head_dim;
    size_t current_tokens;
} VspecKVCache;

int vspec_kv_cache_init(
    VspecKVCache* cache,
    float* key_buffer,
    float* value_buffer,
    size_t max_tokens,
    size_t num_heads,
    size_t head_dim
);

int vspec_kv_cache_append(
    VspecKVCache* cache,
    const float* key_token,
    const float* value_token
);

const float* vspec_kv_cache_key_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx);
const float* vspec_kv_cache_value_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx);

void vspec_kv_cache_reset(VspecKVCache* cache);

#endif
