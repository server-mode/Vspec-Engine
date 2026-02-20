#include "vspec/attention/kv_ring.h"

static size_t stride(const VspecKVCacheRing* ring) {
    return ring->num_heads * ring->head_dim;
}

int vspec_kv_ring_init(
    VspecKVCacheRing* ring,
    float* key_buffer,
    float* value_buffer,
    size_t max_tokens,
    size_t num_heads,
    size_t head_dim
) {
    if (!ring || !key_buffer || !value_buffer || max_tokens == 0U || num_heads == 0U || head_dim == 0U) {
        return 0;
    }
    ring->key = key_buffer;
    ring->value = value_buffer;
    ring->max_tokens = max_tokens;
    ring->num_heads = num_heads;
    ring->head_dim = head_dim;
    ring->start = 0U;
    ring->count = 0U;
    return 1;
}

int vspec_kv_ring_push(
    VspecKVCacheRing* ring,
    const float* key_token,
    const float* value_token
) {
    if (!ring || !key_token || !value_token) {
        return 0;
    }

    const size_t idx = (ring->start + ring->count) % ring->max_tokens;
    const size_t base = idx * stride(ring);

    for (size_t i = 0; i < stride(ring); ++i) {
        ring->key[base + i] = key_token[i];
        ring->value[base + i] = value_token[i];
    }

    if (ring->count < ring->max_tokens) {
        ring->count += 1U;
    } else {
        ring->start = (ring->start + 1U) % ring->max_tokens;
    }

    return 1;
}

static size_t ring_index(const VspecKVCacheRing* ring, size_t idx) {
    return (ring->start + idx) % ring->max_tokens;
}

const float* vspec_kv_ring_key_at(const VspecKVCacheRing* ring, size_t idx, size_t head_idx) {
    if (!ring || idx >= ring->count || head_idx >= ring->num_heads) {
        return 0;
    }
    const size_t base = ring_index(ring, idx) * stride(ring) + head_idx * ring->head_dim;
    return ring->key + base;
}

const float* vspec_kv_ring_value_at(const VspecKVCacheRing* ring, size_t idx, size_t head_idx) {
    if (!ring || idx >= ring->count || head_idx >= ring->num_heads) {
        return 0;
    }
    const size_t base = ring_index(ring, idx) * stride(ring) + head_idx * ring->head_dim;
    return ring->value + base;
}

void vspec_kv_ring_evict(VspecKVCacheRing* ring, size_t keep_last) {
    if (!ring) {
        return;
    }
    if (keep_last >= ring->count) {
        return;
    }

    const size_t drop = ring->count - keep_last;
    ring->start = (ring->start + drop) % ring->max_tokens;
    ring->count = keep_last;
}
