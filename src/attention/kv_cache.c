#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/attention/kv_cache.h"
#include "vspec/quant/pack.h"

static size_t token_stride(const VspecKVCache* cache) {
    return cache->num_heads * cache->head_dim;
}

static size_t clamp_block_size(size_t block_size) {
    if (block_size == 16U || block_size == 32U) {
        return block_size;
    }
    return 32U;
}

static float percentile_abs_block(const float* data, size_t n, float percentile) {
    if (!data || n == 0U) {
        return 0.0f;
    }

    if (percentile < 50.0f) {
        percentile = 50.0f;
    }
    if (percentile > 100.0f) {
        percentile = 100.0f;
    }

    float temp[32];
    if (n > 32U) {
        n = 32U;
    }
    for (size_t i = 0; i < n; ++i) {
        temp[i] = fabsf(data[i]);
    }

    for (size_t i = 0; i < n; ++i) {
        size_t min_idx = i;
        for (size_t j = i + 1U; j < n; ++j) {
            if (temp[j] < temp[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            float swap = temp[i];
            temp[i] = temp[min_idx];
            temp[min_idx] = swap;
        }
    }

    size_t idx = (size_t)((double)(n - 1U) * ((double)percentile / 100.0));
    if (idx >= n) {
        idx = n - 1U;
    }
    return temp[idx];
}

static void quantize_head_int3_percentile(
    const float* src,
    size_t head_dim,
    size_t block_size,
    float percentile,
    uint8_t* packed_out,
    float* scales_out
) {
    if (!src || !packed_out || !scales_out || head_dim == 0U) {
        return;
    }

    const size_t blocks = (head_dim + block_size - 1U) / block_size;
    int8_t* qtmp = (int8_t*)malloc(head_dim * sizeof(int8_t));
    if (!qtmp) {
        memset(packed_out, 0, vspec_quant_packed_bytes(head_dim, 3));
        for (size_t b = 0; b < blocks; ++b) {
            scales_out[b] = 1.0f;
        }
        return;
    }

    for (size_t b = 0; b < blocks; ++b) {
        const size_t base = b * block_size;
        size_t len = block_size;
        if (base + len > head_dim) {
            len = head_dim - base;
        }

        const float rep_abs = percentile_abs_block(src + base, len, percentile);
        const float qmax = 3.0f;
        float scale = (rep_abs > 0.0f) ? (rep_abs / qmax) : 1.0f;
        if (scale < 1e-6f) {
            scale = 1e-6f;
        }
        scales_out[b] = scale;

        for (size_t i = 0; i < len; ++i) {
            const float v = src[base + i] / scale;
            const int q = (int)lrintf(v);
            qtmp[base + i] = vspec_quant_clip_signed((int8_t)q, 3);
        }
    }

    vspec_quant_pack_signed(qtmp, head_dim, 3, packed_out);
    free(qtmp);
}

static void dequantize_head_int3(
    const uint8_t* packed,
    const float* scales,
    size_t head_dim,
    size_t block_size,
    float* out
) {
    if (!packed || !scales || !out || head_dim == 0U) {
        return;
    }

    for (size_t i = 0; i < head_dim; ++i) {
        const size_t b = i / block_size;
        const float s = scales[b];
        const int8_t q = vspec_quant_get_signed(packed, i, 3);
        out[i] = (float)q * s;
    }
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
    cache->int3_compressed = 0;
    cache->key_int3 = NULL;
    cache->value_int3 = NULL;
    cache->key_scales = NULL;
    cache->value_scales = NULL;
    cache->block_size = 0U;
    cache->blocks_per_head = 0U;
    cache->packed_head_bytes = 0U;
    cache->scratch_key_head = NULL;
    cache->scratch_value_head = NULL;
    cache->owns_int3_buffers = 0;
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

    if (cache->int3_compressed && cache->key_int3 && cache->value_int3 && cache->key_scales && cache->value_scales) {
        const size_t token_idx = cache->current_tokens;
        for (size_t h = 0; h < cache->num_heads; ++h) {
            const float* k_head = key_token + h * cache->head_dim;
            const float* v_head = value_token + h * cache->head_dim;

            const size_t qoff = (token_idx * cache->num_heads + h) * cache->packed_head_bytes;
            const size_t soff = (token_idx * cache->num_heads + h) * cache->blocks_per_head;

            quantize_head_int3_percentile(
                k_head,
                cache->head_dim,
                cache->block_size,
                99.5f,
                cache->key_int3 + qoff,
                cache->key_scales + soff
            );
            quantize_head_int3_percentile(
                v_head,
                cache->head_dim,
                cache->block_size,
                99.5f,
                cache->value_int3 + qoff,
                cache->value_scales + soff
            );
        }
    } else {
        for (size_t i = 0; i < stride; ++i) {
            cache->key[base + i] = key_token[i];
            cache->value[base + i] = value_token[i];
        }
    }

    if (cache->key && cache->value) {
        for (size_t i = 0; i < stride; ++i) {
            cache->key[base + i] = key_token[i];
            cache->value[base + i] = value_token[i];
        }
    }

    cache->current_tokens += 1U;
    return 1;
}

int vspec_kv_cache_enable_int3_compression(VspecKVCache* cache, size_t block_size) {
    if (!cache || cache->max_tokens == 0U || cache->num_heads == 0U || cache->head_dim == 0U) {
        return 0;
    }

    block_size = clamp_block_size(block_size);
    const size_t packed_head_bytes = vspec_quant_packed_bytes(cache->head_dim, 3);
    const size_t blocks_per_head = (cache->head_dim + block_size - 1U) / block_size;
    const size_t total_heads = cache->max_tokens * cache->num_heads;

    uint8_t* key_int3 = (uint8_t*)malloc(total_heads * packed_head_bytes);
    uint8_t* value_int3 = (uint8_t*)malloc(total_heads * packed_head_bytes);
    float* key_scales = (float*)malloc(total_heads * blocks_per_head * sizeof(float));
    float* value_scales = (float*)malloc(total_heads * blocks_per_head * sizeof(float));
    float* scratch_k = (float*)malloc(cache->head_dim * sizeof(float));
    float* scratch_v = (float*)malloc(cache->head_dim * sizeof(float));

    if (!key_int3 || !value_int3 || !key_scales || !value_scales || !scratch_k || !scratch_v) {
        free(key_int3);
        free(value_int3);
        free(key_scales);
        free(value_scales);
        free(scratch_k);
        free(scratch_v);
        return 0;
    }

    cache->key_int3 = key_int3;
    cache->value_int3 = value_int3;
    cache->key_scales = key_scales;
    cache->value_scales = value_scales;
    cache->scratch_key_head = scratch_k;
    cache->scratch_value_head = scratch_v;
    cache->block_size = block_size;
    cache->blocks_per_head = blocks_per_head;
    cache->packed_head_bytes = packed_head_bytes;
    cache->int3_compressed = 1;
    cache->owns_int3_buffers = 1;
    return 1;
}

void vspec_kv_cache_disable_int3_compression(VspecKVCache* cache) {
    if (!cache || !cache->int3_compressed) {
        return;
    }

    if (cache->owns_int3_buffers) {
        free(cache->key_int3);
        free(cache->value_int3);
        free(cache->key_scales);
        free(cache->value_scales);
        free(cache->scratch_key_head);
        free(cache->scratch_value_head);
    }

    cache->key_int3 = NULL;
    cache->value_int3 = NULL;
    cache->key_scales = NULL;
    cache->value_scales = NULL;
    cache->scratch_key_head = NULL;
    cache->scratch_value_head = NULL;
    cache->block_size = 0U;
    cache->blocks_per_head = 0U;
    cache->packed_head_bytes = 0U;
    cache->int3_compressed = 0;
    cache->owns_int3_buffers = 0;
}

const float* vspec_kv_cache_key_at(const VspecKVCache* cache, size_t token_idx, size_t head_idx) {
    if (!cache || token_idx >= cache->current_tokens || head_idx >= cache->num_heads) {
        return NULL;
    }

    if (cache->int3_compressed && cache->key_int3 && cache->key_scales && cache->scratch_key_head) {
        const size_t qoff = (token_idx * cache->num_heads + head_idx) * cache->packed_head_bytes;
        const size_t soff = (token_idx * cache->num_heads + head_idx) * cache->blocks_per_head;
        dequantize_head_int3(
            cache->key_int3 + qoff,
            cache->key_scales + soff,
            cache->head_dim,
            cache->block_size,
            cache->scratch_key_head
        );
        return cache->scratch_key_head;
    }

    if (!cache->key) {
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

    if (cache->int3_compressed && cache->value_int3 && cache->value_scales && cache->scratch_value_head) {
        const size_t qoff = (token_idx * cache->num_heads + head_idx) * cache->packed_head_bytes;
        const size_t soff = (token_idx * cache->num_heads + head_idx) * cache->blocks_per_head;
        dequantize_head_int3(
            cache->value_int3 + qoff,
            cache->value_scales + soff,
            cache->head_dim,
            cache->block_size,
            cache->scratch_value_head
        );
        return cache->scratch_value_head;
    }

    if (!cache->value) {
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
