#include <stddef.h>

#include "vspec/attention/kv_cache.h"

static size_t token_stride(const VspecKVCache* cache) {
    return cache->num_heads * cache->head_dim;
}

int vspec_kv_cache_init(
    VspecKVCache* cache,
    float* key_buffer,
    float* value_buffer,
    size_t max_tokens,
    size_t num_heads,
    size_t head_dim
) {
    if (!cache || !key_buffer || !value_buffer || max_tokens == 0U || num_heads == 0U || head_dim == 0U) {
        return 0;
    }

    cache->key = key_buffer;
    cache->value = value_buffer;
    cache->max_tokens = max_tokens;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->current_tokens = 0U;
    return 1;
}

int vspec_kv_cache_append(
    VspecKVCache* cache,
    const float* key_token,
    const float* value_token
) {
    if (!cache || !key_token || !value_token || cache->current_tokens >= cache->max_tokens) {
        return 0;
    }

    const size_t stride = token_stride(cache);
    const size_t base = cache->current_tokens * stride;

    for (size_t i = 0; i < stride; ++i) {
        cache->key[base + i] = key_token[i];
        cache->value[base + i] = value_token[i];
    }

    cache->current_tokens += 1U;
    return 1;
}

const float* vspec_kv_cache_key_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx) {
    if (!cache || token_idx >= cache->current_tokens || head_idx >= cache->num_heads) {
        return NULL;
    }

    const size_t stride = token_stride(cache);
    const size_t offset = token_idx * stride + head_idx * cache->head_dim;
    return cache->key + offset;
}

const float* vspec_kv_cache_value_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx) {
    if (!cache || token_idx >= cache->current_tokens || head_idx >= cache->num_heads) {
        return NULL;
    }

    const size_t stride = token_stride(cache);
    const size_t offset = token_idx * stride + head_idx * cache->head_dim;
    return cache->value + offset;
}

void vspec_kv_cache_reset(VspecKVCache* cache) {
    if (!cache) {
        return;
    }
    cache->current_tokens = 0U;
}
