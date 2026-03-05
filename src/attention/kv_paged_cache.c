#include "vspec/attention/kv_paged_cache.h"

#include <string.h>

static size_t vspec_kv_stride(const VspecKVPagedCache* cache) {
    return cache->num_heads * cache->head_dim;
}

static size_t vspec_kv_page_span(const VspecKVPagedCache* cache) {
    return cache->page_tokens * vspec_kv_stride(cache);
}

static int vspec_find_session_slot(const VspecKVPagedCache* cache, uint64_t session_id) {
    if (!cache || session_id == 0U) {
        return -1;
    }
    for (size_t i = 0U; i < VSPEC_KV_PAGED_MAX_SESSIONS; ++i) {
        if (cache->sessions[i].active && cache->sessions[i].session_id == session_id) {
            return (int)i;
        }
    }
    return -1;
}

static int vspec_find_free_session_slot(const VspecKVPagedCache* cache) {
    if (!cache) {
        return -1;
    }
    for (size_t i = 0U; i < VSPEC_KV_PAGED_MAX_SESSIONS; ++i) {
        if (!cache->sessions[i].active) {
            return (int)i;
        }
    }
    return -1;
}

static void vspec_release_session_pages(VspecKVPagedCache* cache, int session_slot) {
    if (!cache || session_slot < 0) {
        return;
    }

    size_t count = cache->sessions[session_slot].page_count;
    for (size_t i = 0U; i < count; ++i) {
        uint16_t page_idx = cache->sessions[session_slot].pages[i];
        if (page_idx >= cache->max_pages) {
            continue;
        }

        if (cache->page_meta[page_idx].ref_count > 0U) {
            cache->page_meta[page_idx].ref_count -= 1U;
        }
        if (cache->page_meta[page_idx].ref_count == 0U) {
            cache->page_meta[page_idx].allocated = 0U;
            cache->page_meta[page_idx].owner_slot = 0U;
            cache->page_meta[page_idx].used_tokens = 0U;
        }
    }
    (void)memset(&cache->sessions[session_slot], 0, sizeof(cache->sessions[session_slot]));
}

static int vspec_evict_lru_session(VspecKVPagedCache* cache, int protect_session_slot) {
    if (!cache) {
        return 0;
    }

    int victim = -1;
    uint64_t oldest_touch = UINT64_MAX;
    for (size_t i = 0U; i < VSPEC_KV_PAGED_MAX_SESSIONS; ++i) {
        if (!cache->sessions[i].active) {
            continue;
        }
        if ((int)i == protect_session_slot) {
            continue;
        }
        if (cache->sessions[i].last_touch_step < oldest_touch) {
            oldest_touch = cache->sessions[i].last_touch_step;
            victim = (int)i;
        }
    }

    if (victim < 0) {
        return 0;
    }

    vspec_release_session_pages(cache, victim);
    cache->stats.evict_session_count += 1U;
    return 1;
}

static int vspec_alloc_page(VspecKVPagedCache* cache, int owner_slot) {
    if (!cache || owner_slot < 0) {
        return -1;
    }

    for (size_t i = 0U; i < cache->max_pages; ++i) {
        if (!cache->page_meta[i].allocated) {
            cache->page_meta[i].allocated = 1U;
            cache->page_meta[i].owner_slot = (uint16_t)owner_slot;
            cache->page_meta[i].used_tokens = 0U;
            cache->page_meta[i].ref_count = 1U;
            cache->page_meta[i].last_touch_step = cache->clock_step;
            cache->stats.page_alloc_count += 1U;
            return (int)i;
        }
    }

    if (vspec_evict_lru_session(cache, owner_slot)) {
        for (size_t i = 0U; i < cache->max_pages; ++i) {
            if (!cache->page_meta[i].allocated) {
                cache->page_meta[i].allocated = 1U;
                cache->page_meta[i].owner_slot = (uint16_t)owner_slot;
                cache->page_meta[i].used_tokens = 0U;
                cache->page_meta[i].ref_count = 1U;
                cache->page_meta[i].last_touch_step = cache->clock_step;
                cache->stats.page_alloc_count += 1U;
                return (int)i;
            }
        }
    }

    return -1;
}

int vspec_kv_paged_cache_init(
    VspecKVPagedCache* cache,
    float* key_pages,
    float* value_pages,
    size_t page_tokens,
    size_t max_pages,
    size_t num_heads,
    size_t head_dim
) {
    if (!cache || !key_pages || !value_pages || page_tokens == 0U || max_pages == 0U || num_heads == 0U || head_dim == 0U) {
        return 0;
    }
    if (max_pages > VSPEC_KV_PAGED_MAX_PAGES) {
        return 0;
    }

    (void)memset(cache, 0, sizeof(*cache));
    cache->key_pages = key_pages;
    cache->value_pages = value_pages;
    cache->page_tokens = page_tokens;
    cache->max_pages = max_pages;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->clock_step = 1U;
    return 1;
}

int vspec_kv_paged_open_session(VspecKVPagedCache* cache, uint64_t session_id) {
    if (!cache || session_id == 0U) {
        return 0;
    }
    if (vspec_find_session_slot(cache, session_id) >= 0) {
        return 1;
    }

    int slot = vspec_find_free_session_slot(cache);
    if (slot < 0) {
        if (!vspec_evict_lru_session(cache, -1)) {
            return 0;
        }
        slot = vspec_find_free_session_slot(cache);
        if (slot < 0) {
            return 0;
        }
    }

    cache->sessions[slot].active = 1U;
    cache->sessions[slot].session_id = session_id;
    cache->sessions[slot].last_touch_step = cache->clock_step;
    return 1;
}

int vspec_kv_paged_close_session(VspecKVPagedCache* cache, uint64_t session_id) {
    int slot = vspec_find_session_slot(cache, session_id);
    if (slot < 0) {
        return 0;
    }
    vspec_release_session_pages(cache, slot);
    return 1;
}

int vspec_kv_paged_reuse_prefix(
    VspecKVPagedCache* cache,
    uint64_t target_session_id,
    uint64_t source_session_id,
    size_t max_tokens
) {
    if (!cache || target_session_id == 0U || source_session_id == 0U) {
        return 0;
    }

    int source_slot = vspec_find_session_slot(cache, source_session_id);
    if (source_slot < 0 || !cache->sessions[source_slot].active) {
        return 0;
    }

    if (!vspec_kv_paged_open_session(cache, target_session_id)) {
        return 0;
    }
    int target_slot = vspec_find_session_slot(cache, target_session_id);
    if (target_slot < 0) {
        return 0;
    }

    if (cache->sessions[target_slot].page_count != 0U) {
        return 0;
    }

    size_t copied_tokens = 0U;
    for (size_t i = 0U; i < cache->sessions[source_slot].page_count; ++i) {
        if (cache->sessions[target_slot].page_count >= VSPEC_KV_PAGED_MAX_PAGES_PER_SESSION) {
            break;
        }

        uint16_t page_idx = cache->sessions[source_slot].pages[i];
        if (page_idx >= cache->max_pages) {
            continue;
        }

        size_t page_tokens = cache->page_meta[page_idx].used_tokens;
        if (max_tokens > 0U && copied_tokens >= max_tokens) {
            break;
        }

        if (max_tokens > 0U && copied_tokens + page_tokens > max_tokens) {
            break;
        }

        cache->sessions[target_slot].pages[cache->sessions[target_slot].page_count++] = page_idx;
        cache->sessions[target_slot].total_tokens += page_tokens;
        cache->page_meta[page_idx].ref_count += 1U;
        copied_tokens += page_tokens;
        cache->stats.page_reuse_count += 1U;
    }

    cache->sessions[target_slot].last_touch_step = cache->clock_step;
    return 1;
}

int vspec_kv_paged_append(
    VspecKVPagedCache* cache,
    uint64_t session_id,
    const float* key_token,
    const float* value_token
) {
    if (!cache || !key_token || !value_token) {
        return 0;
    }

    int session_slot = vspec_find_session_slot(cache, session_id);
    if (session_slot < 0) {
        if (!vspec_kv_paged_open_session(cache, session_id)) {
            return 0;
        }
        session_slot = vspec_find_session_slot(cache, session_id);
    }
    if (session_slot < 0) {
        return 0;
    }

    cache->clock_step += 1U;
    cache->sessions[session_slot].last_touch_step = cache->clock_step;

    int page_idx = -1;
    if (cache->sessions[session_slot].page_count > 0U) {
        page_idx = cache->sessions[session_slot].pages[cache->sessions[session_slot].page_count - 1U];
        if ((size_t)page_idx >= cache->max_pages) {
            page_idx = -1;
        }
    }

    if (page_idx >= 0) {
        if (cache->page_meta[page_idx].used_tokens >= cache->page_tokens || cache->page_meta[page_idx].ref_count > 1U) {
            page_idx = -1;
        }
    }

    if (page_idx < 0) {
        if (cache->sessions[session_slot].page_count >= VSPEC_KV_PAGED_MAX_PAGES_PER_SESSION) {
            return 0;
        }
        page_idx = vspec_alloc_page(cache, session_slot);
        if (page_idx < 0) {
            return 0;
        }
        cache->sessions[session_slot].pages[cache->sessions[session_slot].page_count++] = (uint16_t)page_idx;
    }

    const size_t stride = vspec_kv_stride(cache);
    const size_t page_span = vspec_kv_page_span(cache);
    const size_t token_offset = (size_t)cache->page_meta[page_idx].used_tokens * stride;
    const size_t page_offset = (size_t)page_idx * page_span;

    float* key_dst = cache->key_pages + page_offset + token_offset;
    float* value_dst = cache->value_pages + page_offset + token_offset;

    for (size_t i = 0U; i < stride; ++i) {
        key_dst[i] = key_token[i];
        value_dst[i] = value_token[i];
    }

    cache->page_meta[page_idx].used_tokens += 1U;
    cache->page_meta[page_idx].last_touch_step = cache->clock_step;
    cache->sessions[session_slot].total_tokens += 1U;
    cache->stats.append_token_count += 1U;
    return 1;
}

int vspec_kv_paged_get_token_ptr(
    const VspecKVPagedCache* cache,
    uint64_t session_id,
    size_t token_idx,
    size_t head_idx,
    const float** out_key,
    const float** out_value
) {
    if (!cache || !out_key || !out_value || head_idx >= cache->num_heads) {
        return 0;
    }

    int session_slot = vspec_find_session_slot(cache, session_id);
    if (session_slot < 0 || !cache->sessions[session_slot].active) {
        return 0;
    }
    if (token_idx >= cache->sessions[session_slot].total_tokens) {
        return 0;
    }

    size_t remain = token_idx;
    uint16_t page_idx = 0U;
    size_t token_in_page = 0U;
    for (size_t i = 0U; i < cache->sessions[session_slot].page_count; ++i) {
        uint16_t p = cache->sessions[session_slot].pages[i];
        if (p >= cache->max_pages) {
            return 0;
        }
        size_t used = cache->page_meta[p].used_tokens;
        if (remain < used) {
            page_idx = p;
            token_in_page = remain;
            break;
        }
        remain -= used;
    }

    const size_t stride = vspec_kv_stride(cache);
    const size_t page_span = vspec_kv_page_span(cache);
    const size_t offset = (size_t)page_idx * page_span + token_in_page * stride + head_idx * cache->head_dim;

    *out_key = cache->key_pages + offset;
    *out_value = cache->value_pages + offset;
    return 1;
}

size_t vspec_kv_paged_session_tokens(const VspecKVPagedCache* cache, uint64_t session_id) {
    int slot = vspec_find_session_slot(cache, session_id);
    if (slot < 0) {
        return 0U;
    }
    return cache->sessions[slot].total_tokens;
}

void vspec_kv_paged_stats(const VspecKVPagedCache* cache, VspecKVPagedStats* out_stats) {
    if (!cache || !out_stats) {
        return;
    }

    *out_stats = cache->stats;
    out_stats->free_pages = 0U;
    out_stats->active_sessions = 0U;
    for (size_t i = 0U; i < cache->max_pages; ++i) {
        if (!cache->page_meta[i].allocated) {
            out_stats->free_pages += 1U;
        }
    }
    for (size_t i = 0U; i < VSPEC_KV_PAGED_MAX_SESSIONS; ++i) {
        if (cache->sessions[i].active) {
            out_stats->active_sessions += 1U;
        }
    }
}