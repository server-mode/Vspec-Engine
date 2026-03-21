#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vspec/python/capi.h"
#include "vspec/attention/kv_paged_cache.h"
#include "vspec/compat/pytorch_loader.h"
#include "vspec/compat/safetensors_parser.h"
#include "vspec/compat/graph_rewrite.h"
#include "vspec/compat/weight_mapper.h"
#include "vspec/graph/ir.h"
#include "vspec/runtime/continuous_batch.h"
#include "vspec/runtime/decode_session.h"
#include "vspec/runtime/sampling_core.h"
#include "vspec/runtime/plugin/plugin_api.h"

#define VSPEC_PY_MAX_KV_CACHE_HANDLES 16
#define VSPEC_PY_MAX_DECODE_HANDLES 32
#define VSPEC_PY_MAX_CONT_BATCH_HANDLES 16

typedef struct VspecPyKVPagedHandle {
    int active;
    float* key_pages;
    float* value_pages;
    size_t page_tokens;
    size_t max_pages;
    size_t num_heads;
    size_t head_dim;
    VspecKVPagedCache cache;
} VspecPyKVPagedHandle;

static VspecPyKVPagedHandle g_kv_handles[VSPEC_PY_MAX_KV_CACHE_HANDLES];

typedef struct VspecPyDecodeHandle {
    int active;
    VspecDecodeSession session;
} VspecPyDecodeHandle;

static VspecPyDecodeHandle g_decode_handles[VSPEC_PY_MAX_DECODE_HANDLES];

typedef struct VspecPyContinuousBatchHandle {
    int active;
    VspecContinuousBatcher batcher;
} VspecPyContinuousBatchHandle;

static VspecPyContinuousBatchHandle g_cont_batch_handles[VSPEC_PY_MAX_CONT_BATCH_HANDLES];

static VspecPyKVPagedHandle* vspec_py_get_kv_handle(int handle_id) {
    if (handle_id <= 0 || handle_id > VSPEC_PY_MAX_KV_CACHE_HANDLES) {
        return NULL;
    }
    if (!g_kv_handles[handle_id - 1].active) {
        return NULL;
    }
    return &g_kv_handles[handle_id - 1];
}

static VspecPyDecodeHandle* vspec_py_get_decode_handle(int handle_id) {
    if (handle_id <= 0 || handle_id > VSPEC_PY_MAX_DECODE_HANDLES) {
        return NULL;
    }
    if (!g_decode_handles[handle_id - 1].active) {
        return NULL;
    }
    return &g_decode_handles[handle_id - 1];
}

static VspecPyContinuousBatchHandle* vspec_py_get_cont_batch_handle(int handle_id) {
    if (handle_id <= 0 || handle_id > VSPEC_PY_MAX_CONT_BATCH_HANDLES) {
        return NULL;
    }
    if (!g_cont_batch_handles[handle_id - 1].active) {
        return NULL;
    }
    return &g_cont_batch_handles[handle_id - 1];
}

const char* vspec_py_version(void) {
    return "0.1.0-week10";
}

int vspec_py_load_manifest_count(const char* path) {
    VspecCompatModel m;
    if (!vspec_pytorch_load_manifest(path, &m)) {
        return -1;
    }
    return (int)m.tensor_count;
}

int vspec_py_parse_safetensors_count(const char* path) {
    VspecCompatModel m;
    if (!vspec_safetensors_parse_header_file(path, &m)) {
        return -1;
    }
    return (int)m.tensor_count;
}

int vspec_py_rewrite_demo_final_nodes(void) {
    VspecGraph g;
    VspecGraphRewriteStats st = {0};

    vspec_graph_init(&g);
    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ACT_RELU, 2, 0, 3);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 3, 4, 5);

    vspec_graph_rewrite_fuse_linear_relu(&g, &st);
    vspec_graph_rewrite_compact(&g, &st);

    return (int)g.node_count;
}

int vspec_py_generate(const char* prompt, char* out, size_t out_size) {
    if (!out || out_size == 0U) {
        return 0;
    }

    const char* p = prompt ? prompt : "";
    const int n = snprintf(out, out_size, "vspec> %s", p);
    if (n < 0) {
        out[0] = '\0';
        return 0;
    }
    return 1;
}

int vspec_py_weight_canonical_name(const char* raw_name, char* out_name, size_t out_name_size) {
    return vspec_weight_canonical_name(raw_name, out_name, out_name_size);
}

int vspec_py_safetensors_tensor_descriptor(
    const char* path,
    const char* tensor_name,
    char* out_dtype,
    size_t out_dtype_size,
    size_t* out_shape,
    size_t shape_cap,
    size_t* out_ndim,
    uint64_t* out_data_start,
    uint64_t* out_data_end
) {
    VspecCompatTensorInfo info;
    if (!path || !tensor_name) {
        return 0;
    }
    if (!vspec_safetensors_find_tensor_file(path, tensor_name, &info)) {
        return 0;
    }
    if (out_dtype && out_dtype_size > 0U) {
        (void)snprintf(out_dtype, out_dtype_size, "%s", info.dtype);
    }
    if (out_shape && shape_cap > 0U) {
        const size_t limit = (info.ndim < shape_cap) ? info.ndim : shape_cap;
        for (size_t i = 0U; i < limit; ++i) {
            out_shape[i] = info.shape[i];
        }
    }
    if (out_ndim) {
        *out_ndim = info.ndim;
    }
    if (out_data_start) {
        *out_data_start = info.data_offset_start;
    }
    if (out_data_end) {
        *out_data_end = info.data_offset_end;
    }
    return 1;
}

int vspec_py_sample_candidate(
    const int* token_ids,
    const float* scores,
    size_t count,
    int greedy,
    uint64_t random_bits,
    int* out_token_id
) {
    return vspec_sampling_select_candidate(token_ids, scores, count, greedy, random_bits, out_token_id);
}

int vspec_py_kv_cache_create(size_t page_tokens, size_t max_pages, size_t num_heads, size_t head_dim) {
    const size_t stride = num_heads * head_dim;
    const size_t page_span = page_tokens * stride;
    for (int i = 0; i < VSPEC_PY_MAX_KV_CACHE_HANDLES; ++i) {
        VspecPyKVPagedHandle* handle = &g_kv_handles[i];
        if (handle->active) {
            continue;
        }
        (void)memset(handle, 0, sizeof(*handle));
        handle->key_pages = (float*)malloc(max_pages * page_span * sizeof(float));
        handle->value_pages = (float*)malloc(max_pages * page_span * sizeof(float));
        if (!handle->key_pages || !handle->value_pages) {
            free(handle->key_pages);
            free(handle->value_pages);
            (void)memset(handle, 0, sizeof(*handle));
            return 0;
        }
        handle->page_tokens = page_tokens;
        handle->max_pages = max_pages;
        handle->num_heads = num_heads;
        handle->head_dim = head_dim;
        if (!vspec_kv_paged_cache_init(&handle->cache, handle->key_pages, handle->value_pages, page_tokens, max_pages, num_heads, head_dim)) {
            free(handle->key_pages);
            free(handle->value_pages);
            (void)memset(handle, 0, sizeof(*handle));
            return 0;
        }
        handle->active = 1;
        return i + 1;
    }
    return 0;
}

void vspec_py_kv_cache_destroy(int handle_id) {
    VspecPyKVPagedHandle* handle = vspec_py_get_kv_handle(handle_id);
    if (!handle) {
        return;
    }
    free(handle->key_pages);
    free(handle->value_pages);
    (void)memset(handle, 0, sizeof(*handle));
}

int vspec_py_kv_cache_reset(int handle_id) {
    VspecPyKVPagedHandle* handle = vspec_py_get_kv_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_kv_paged_cache_init(
        &handle->cache,
        handle->key_pages,
        handle->value_pages,
        handle->page_tokens,
        handle->max_pages,
        handle->num_heads,
        handle->head_dim
    );
}

int vspec_py_kv_cache_append(int handle_id, uint64_t session_id, const float* key_token, const float* value_token) {
    VspecPyKVPagedHandle* handle = vspec_py_get_kv_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_kv_paged_append(&handle->cache, session_id, key_token, value_token);
}

size_t vspec_py_kv_cache_session_tokens(int handle_id, uint64_t session_id) {
    VspecPyKVPagedHandle* handle = vspec_py_get_kv_handle(handle_id);
    if (!handle) {
        return 0U;
    }
    return vspec_kv_paged_session_tokens(&handle->cache, session_id);
}

size_t vspec_py_kv_cache_read(
    int handle_id,
    uint64_t session_id,
    float* out_keys,
    float* out_values,
    size_t max_tokens
) {
    VspecPyKVPagedHandle* handle = vspec_py_get_kv_handle(handle_id);
    const size_t stride = handle ? (handle->num_heads * handle->head_dim) : 0U;
    size_t total_tokens = 0U;

    if (!handle || !out_keys || !out_values || stride == 0U) {
        return 0U;
    }

    total_tokens = vspec_kv_paged_session_tokens(&handle->cache, session_id);
    if (max_tokens > 0U && total_tokens > max_tokens) {
        total_tokens = max_tokens;
    }

    for (size_t t = 0U; t < total_tokens; ++t) {
        for (size_t h = 0U; h < handle->num_heads; ++h) {
            const float* key_ptr = NULL;
            const float* value_ptr = NULL;
            if (!vspec_kv_paged_get_token_ptr(&handle->cache, session_id, t, h, &key_ptr, &value_ptr)) {
                return t;
            }
            for (size_t d = 0U; d < handle->head_dim; ++d) {
                const size_t base = t * stride + h * handle->head_dim + d;
                out_keys[base] = key_ptr[d];
                out_values[base] = value_ptr[d];
            }
        }
    }

    return total_tokens;
}

int vspec_py_decode_session_create(
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_tokens,
    size_t token_quantum
) {
    for (int i = 0; i < VSPEC_PY_MAX_DECODE_HANDLES; ++i) {
        VspecPyDecodeHandle* handle = &g_decode_handles[i];
        if (handle->active) {
            continue;
        }
        (void)memset(handle, 0, sizeof(*handle));
        vspec_decode_session_init(
            &handle->session,
            total_vram_bytes,
            max_active,
            max_batch_tokens,
            token_quantum
        );
        handle->active = 1;
        return i + 1;
    }
    return 0;
}

void vspec_py_decode_session_destroy(int handle_id) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return;
    }
    if (vspec_decode_session_is_active(&handle->session)) {
        (void)vspec_decode_session_cancel(&handle->session);
    }
    (void)memset(handle, 0, sizeof(*handle));
}

int vspec_py_decode_session_begin(
    int handle_id,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority
) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_decode_session_begin(&handle->session, reserve_bytes, prompt_tokens, max_new_tokens, priority);
}

size_t vspec_py_decode_session_next_quota(int handle_id) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return 0U;
    }
    return vspec_decode_session_next_quota(&handle->session);
}

int vspec_py_decode_session_commit(int handle_id, size_t generated_tokens, int reached_eos) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_decode_session_commit(&handle->session, generated_tokens, reached_eos);
}

int vspec_py_decode_session_cancel(int handle_id) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_decode_session_cancel(&handle->session);
}

int vspec_py_decode_session_is_active(int handle_id) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_decode_session_is_active(&handle->session);
}

size_t vspec_py_decode_session_remaining_tokens(int handle_id) {
    VspecPyDecodeHandle* handle = vspec_py_get_decode_handle(handle_id);
    if (!handle) {
        return 0U;
    }
    return vspec_decode_session_remaining_tokens(&handle->session);
}

int vspec_py_continuous_batch_create(
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_items,
    size_t max_batch_tokens,
    size_t prefill_quantum,
    size_t decode_quantum
) {
    for (int i = 0; i < VSPEC_PY_MAX_CONT_BATCH_HANDLES; ++i) {
        VspecPyContinuousBatchHandle* handle = &g_cont_batch_handles[i];
        if (handle->active) {
            continue;
        }
        (void)memset(handle, 0, sizeof(*handle));
        vspec_continuous_batch_init(
            &handle->batcher,
            total_vram_bytes,
            max_active,
            max_batch_items,
            max_batch_tokens,
            prefill_quantum,
            decode_quantum
        );
        handle->active = 1;
        return i + 1;
    }
    return 0;
}

void vspec_py_continuous_batch_destroy(int handle_id) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    if (!handle) {
        return;
    }
    (void)memset(handle, 0, sizeof(*handle));
}

int vspec_py_continuous_batch_submit(
    int handle_id,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority,
    uint64_t* out_request_id
) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_continuous_batch_submit(
        &handle->batcher,
        reserve_bytes,
        prompt_tokens,
        max_new_tokens,
        priority,
        out_request_id
    );
}

size_t vspec_py_continuous_batch_next(
    int handle_id,
    uint64_t* out_request_ids,
    uint32_t* out_phases,
    size_t* out_quotas,
    size_t* out_prompt_cursors,
    size_t cap
) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    VspecContinuousBatchItem items[64];
    size_t count = 0U;
    if (!handle || !out_request_ids || !out_phases || !out_quotas || !out_prompt_cursors || cap == 0U) {
        return 0U;
    }
    if (cap > 64U) {
        cap = 64U;
    }
    count = vspec_continuous_batch_next_batch(&handle->batcher, items, cap);
    for (size_t i = 0U; i < count; ++i) {
        out_request_ids[i] = items[i].request_id;
        out_phases[i] = items[i].phase;
        out_quotas[i] = items[i].token_quota;
        out_prompt_cursors[i] = items[i].prompt_cursor;
    }
    return count;
}

int vspec_py_continuous_batch_commit_prefill(int handle_id, uint64_t request_id, size_t consumed_tokens) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_continuous_batch_commit_prefill(&handle->batcher, request_id, consumed_tokens);
}

int vspec_py_continuous_batch_commit_decode(int handle_id, uint64_t request_id, size_t generated_tokens, int reached_eos) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_continuous_batch_commit_decode(&handle->batcher, request_id, generated_tokens, reached_eos);
}

int vspec_py_continuous_batch_cancel(int handle_id, uint64_t request_id) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_continuous_batch_cancel(&handle->batcher, request_id);
}

int vspec_py_continuous_batch_stats(
    int handle_id,
    uint64_t* out_prefill_steps,
    uint64_t* out_decode_steps,
    uint64_t* out_prefill_tokens,
    uint64_t* out_decode_tokens,
    size_t* out_active_prefill,
    size_t* out_active_decode,
    size_t* out_reserved_vram
) {
    VspecPyContinuousBatchHandle* handle = vspec_py_get_cont_batch_handle(handle_id);
    VspecContinuousBatchStats stats;
    if (!handle) {
        return 0;
    }
    vspec_continuous_batch_stats(&handle->batcher, &stats);
    if (out_prefill_steps) {
        *out_prefill_steps = stats.prefill_steps;
    }
    if (out_decode_steps) {
        *out_decode_steps = stats.decode_steps;
    }
    if (out_prefill_tokens) {
        *out_prefill_tokens = stats.prefill_tokens;
    }
    if (out_decode_tokens) {
        *out_decode_tokens = stats.decode_tokens;
    }
    if (out_active_prefill) {
        *out_active_prefill = stats.active_prefill_requests;
    }
    if (out_active_decode) {
        *out_active_decode = stats.active_decode_requests;
    }
    if (out_reserved_vram) {
        *out_reserved_vram = stats.reserved_vram_bytes;
    }
    return 1;
}

int vspec_py_runtime_adaptive_step(
    const char* token_text,
    float token_entropy,
    float attention_entropy_collapse,
    float latency_ms,
    float vram_pressure,
    float quality_drift,
    uint32_t layer_type,
    uint8_t* out_target_bits,
    uint8_t* out_skip_compute,
    uint8_t* out_reduce_attention_depth,
    uint8_t* out_enable_kv_compression,
    uint8_t* out_routed_bits,
    uint32_t* out_attention_depth_hint,
    uint32_t* out_token_tier,
    float* out_token_importance,
    uint32_t* out_kv_action
) {
    (void)token_text;
    (void)layer_type;

    float pressure = vram_pressure;
    float collapse = attention_entropy_collapse;
    float drift = quality_drift;
    float entropy = token_entropy;
    uint8_t target_bits = 4U;
    uint8_t skip_compute = 0U;
    uint8_t reduce_attention_depth = 0U;
    uint8_t enable_kv_compression = 0U;
    uint8_t routed_bits = 4U;
    uint32_t attention_depth_hint = 8U;
    uint32_t token_tier = 1U;
    float token_importance = 0.5f;
    uint32_t kv_action = 0U;

    if (entropy < 0.9f) {
        token_tier = 2U;
        token_importance = 0.78f;
        attention_depth_hint = 16U;
    } else if (entropy > 1.8f) {
        token_tier = 0U;
        token_importance = 0.24f;
        attention_depth_hint = 4U;
    }

    if (pressure >= 0.88f || latency_ms >= 28.0f) {
        target_bits = 2U;
        skip_compute = 1U;
        reduce_attention_depth = 1U;
        enable_kv_compression = 1U;
        kv_action = 2U;
    } else if (pressure >= 0.72f || latency_ms >= 18.0f) {
        target_bits = 3U;
        reduce_attention_depth = 1U;
        enable_kv_compression = 1U;
        kv_action = 1U;
    } else {
        target_bits = 4U;
        kv_action = 0U;
    }

    if (drift >= 0.65f || collapse >= 0.72f) {
        if (target_bits < 4U) {
            target_bits = 4U;
        }
        skip_compute = 0U;
    }

    routed_bits = target_bits;
    if (token_tier == 2U && routed_bits < 4U) {
        routed_bits = (uint8_t)(routed_bits + 1U);
    }
    if (token_tier == 0U && routed_bits > 2U) {
        routed_bits = (uint8_t)(routed_bits - 1U);
    }

    if (out_target_bits) {
        *out_target_bits = target_bits;
    }
    if (out_skip_compute) {
        *out_skip_compute = skip_compute;
    }
    if (out_reduce_attention_depth) {
        *out_reduce_attention_depth = reduce_attention_depth;
    }
    if (out_enable_kv_compression) {
        *out_enable_kv_compression = enable_kv_compression;
    }
    if (out_routed_bits) {
        *out_routed_bits = routed_bits;
    }
    if (out_attention_depth_hint) {
        *out_attention_depth_hint = attention_depth_hint;
    }
    if (out_token_tier) {
        *out_token_tier = token_tier;
    }
    if (out_token_importance) {
        *out_token_importance = token_importance;
    }
    if (out_kv_action) {
        *out_kv_action = kv_action;
    }

    return 1;
}

int vspec_py_plugin_load_dynamic(const char* path, const char* symbol_name, char* out_msg, size_t out_msg_size) {
    return vspec_plugin_load_dynamic(path, symbol_name, out_msg, out_msg_size);
}

int vspec_py_plugin_unload_dynamic(const char* name, char* out_msg, size_t out_msg_size) {
    return vspec_plugin_unload_dynamic(name, out_msg, out_msg_size);
}
