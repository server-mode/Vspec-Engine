#ifndef VSPEC_ATTENTION_KV_RING_H
#define VSPEC_ATTENTION_KV_RING_H

#include <stddef.h>

typedef struct VspecKVCacheRing {
    float* key;
    float* value;
    size_t max_tokens;
    size_t num_heads;
    size_t head_dim;
    size_t start;
    size_t count;
} VspecKVCacheRing;

int vspec_kv_ring_init(
    VspecKVCacheRing* ring,
    float* key_buffer,
    float* value_buffer,
    size_t max_tokens,
    size_t num_heads,
    size_t head_dim
);

int vspec_kv_ring_push(
    VspecKVCacheRing* ring,
    const float* key_token,
    const float* value_token
);

const float* vspec_kv_ring_key_at(const VspecKVCacheRing* ring, size_t idx, size_t head_idx);
const float* vspec_kv_ring_value_at(const VspecKVCacheRing* ring, size_t idx, size_t head_idx);
void vspec_kv_ring_evict(VspecKVCacheRing* ring, size_t keep_last);

#endif
