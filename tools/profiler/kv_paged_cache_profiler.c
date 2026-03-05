#include <stdio.h>
#include <stdlib.h>

#include "vspec/attention/kv_paged_cache.h"

static void fill_token(float* dst, size_t n, float base) {
    for (size_t i = 0U; i < n; ++i) {
        dst[i] = base + (float)i * 0.001f;
    }
}

int main(void) {
    const size_t page_tokens = 32U;
    const size_t max_pages = 128U;
    const size_t num_heads = 8U;
    const size_t head_dim = 64U;
    const size_t stride = num_heads * head_dim;
    const size_t total = page_tokens * max_pages * stride;

    float* key_pages = (float*)malloc(total * sizeof(float));
    float* value_pages = (float*)malloc(total * sizeof(float));
    float* key_token = (float*)malloc(stride * sizeof(float));
    float* value_token = (float*)malloc(stride * sizeof(float));
    if (!key_pages || !value_pages || !key_token || !value_token) {
        free(key_pages);
        free(value_pages);
        free(key_token);
        free(value_token);
        return 2;
    }

    VspecKVPagedCache cache;
    if (!vspec_kv_paged_cache_init(&cache, key_pages, value_pages, page_tokens, max_pages, num_heads, head_dim)) {
        free(key_pages);
        free(value_pages);
        free(key_token);
        free(value_token);
        return 3;
    }

    (void)vspec_kv_paged_open_session(&cache, 1001U);
    for (size_t t = 0U; t < 120U; ++t) {
        fill_token(key_token, stride, (float)t);
        fill_token(value_token, stride, (float)t + 0.5f);
        (void)vspec_kv_paged_append(&cache, 1001U, key_token, value_token);
    }

    (void)vspec_kv_paged_open_session(&cache, 1002U);
    (void)vspec_kv_paged_reuse_prefix(&cache, 1002U, 1001U, 64U);
    for (size_t t = 0U; t < 96U; ++t) {
        fill_token(key_token, stride, 1000.0f + (float)t);
        fill_token(value_token, stride, 2000.0f + (float)t);
        (void)vspec_kv_paged_append(&cache, 1002U, key_token, value_token);
    }

    const float* key_ptr = NULL;
    const float* value_ptr = NULL;
    int ok = vspec_kv_paged_get_token_ptr(&cache, 1002U, 48U, 2U, &key_ptr, &value_ptr);

    VspecKVPagedStats stats;
    vspec_kv_paged_stats(&cache, &stats);

    printf("[kv-paged] active_sessions=%zu free_pages=%zu\n", stats.active_sessions, stats.free_pages);
    printf("[kv-paged] alloc=%llu reuse=%llu evict=%llu append=%llu\n",
        (unsigned long long)stats.page_alloc_count,
        (unsigned long long)stats.page_reuse_count,
        (unsigned long long)stats.evict_session_count,
        (unsigned long long)stats.append_token_count);
    printf("[kv-paged] session1001_tokens=%zu session1002_tokens=%zu\n",
        vspec_kv_paged_session_tokens(&cache, 1001U),
        vspec_kv_paged_session_tokens(&cache, 1002U));
    printf("[kv-paged] token_probe_ok=%d key0=%.6f value0=%.6f\n",
        ok,
        (ok && key_ptr) ? key_ptr[0] : 0.0f,
        (ok && value_ptr) ? value_ptr[0] : 0.0f);

    (void)vspec_kv_paged_close_session(&cache, 1001U);
    (void)vspec_kv_paged_close_session(&cache, 1002U);

    free(key_pages);
    free(value_pages);
    free(key_token);
    free(value_token);
    return 0;
}