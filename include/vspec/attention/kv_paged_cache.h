#ifndef VSPEC_ATTENTION_KV_PAGED_CACHE_H
#define VSPEC_ATTENTION_KV_PAGED_CACHE_H

#include <stddef.h>
#include <stdint.h>

#define VSPEC_KV_PAGED_MAX_SESSIONS 128U
#define VSPEC_KV_PAGED_MAX_PAGES 4096U
#define VSPEC_KV_PAGED_MAX_PAGES_PER_SESSION 1024U

typedef struct VspecKVPagedStats {
    uint64_t page_alloc_count;
    uint64_t page_reuse_count;
    uint64_t evict_session_count;
    uint64_t append_token_count;
    size_t free_pages;
    size_t active_sessions;
} VspecKVPagedStats;

typedef struct VspecKVPagedCache {
    float* key_pages;
    float* value_pages;

    size_t page_tokens;
    size_t max_pages;
    size_t num_heads;
    size_t head_dim;

    VspecKVPagedStats stats;

    struct {
        uint8_t allocated;
        uint16_t owner_slot;
        uint16_t used_tokens;
        uint16_t ref_count;
        uint64_t last_touch_step;
    } page_meta[VSPEC_KV_PAGED_MAX_PAGES];

    struct {
        uint8_t active;
        uint64_t session_id;
        uint64_t last_touch_step;
        size_t total_tokens;
        size_t page_count;
        uint16_t pages[VSPEC_KV_PAGED_MAX_PAGES_PER_SESSION];
    } sessions[VSPEC_KV_PAGED_MAX_SESSIONS];

    uint64_t clock_step;
} VspecKVPagedCache;

int vspec_kv_paged_cache_init(
    VspecKVPagedCache* cache,
    float* key_pages,
    float* value_pages,
    size_t page_tokens,
    size_t max_pages,
    size_t num_heads,
    size_t head_dim
);

int vspec_kv_paged_open_session(VspecKVPagedCache* cache, uint64_t session_id);
int vspec_kv_paged_close_session(VspecKVPagedCache* cache, uint64_t session_id);

int vspec_kv_paged_reuse_prefix(
    VspecKVPagedCache* cache,
    uint64_t target_session_id,
    uint64_t source_session_id,
    size_t max_tokens
);

int vspec_kv_paged_append(
    VspecKVPagedCache* cache,
    uint64_t session_id,
    const float* key_token,
    const float* value_token
);

int vspec_kv_paged_get_token_ptr(
    const VspecKVPagedCache* cache,
    uint64_t session_id,
    size_t token_idx,
    size_t head_idx,
    const float** out_key,
    const float** out_value
);

size_t vspec_kv_paged_session_tokens(const VspecKVPagedCache* cache, uint64_t session_id);
void vspec_kv_paged_stats(const VspecKVPagedCache* cache, VspecKVPagedStats* out_stats);

#endif