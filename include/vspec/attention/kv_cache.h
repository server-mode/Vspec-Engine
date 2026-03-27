#ifndef VSPEC_ATTENTION_KV_CACHE_H
#define VSPEC_ATTENTION_KV_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VspecKVCache {
    float* key;
    float* value;
    size_t max_tokens;
    size_t num_heads;
    size_t head_dim;
    size_t current_tokens;

    int int3_compressed;
    uint8_t* key_int3;
    uint8_t* value_int3;
    float* key_scales;
    float* value_scales;
    size_t block_size;
    size_t blocks_per_head;
    size_t packed_head_bytes;
    float* scratch_key_head;
    float* scratch_value_head;
    int owns_int3_buffers;
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

int vspec_kv_cache_enable_int3_compression(VspecKVCache* cache, size_t block_size);
void vspec_kv_cache_disable_int3_compression(VspecKVCache* cache);

const float* vspec_kv_cache_key_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx);
const float* vspec_kv_cache_value_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx);

void vspec_kv_cache_reset(VspecKVCache* cache);

#ifdef __cplusplus
}
#endif

#endif
