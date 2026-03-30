#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "vspec/python/capi.h"
#include "vspec/attention/kv_paged_cache.h"
#include "vspec/compat/pytorch_loader.h"
#include "vspec/compat/safetensors_parser.h"
#include "vspec/compat/graph_rewrite.h"
#include "vspec/compat/weight_mapper.h"
#include "vspec/graph/ir.h"
#include "vspec/runtime/continuous_batch.h"
#include "vspec/runtime/decode_session.h"
#include "vspec/runtime/native_inference.h"
#include "vspec/runtime/runtime.h"
#include "vspec/runtime/sampling_core.h"
#include "vspec/runtime/plugin/plugin_api.h"

#define VSPEC_PY_MAX_KV_CACHE_HANDLES 16
#define VSPEC_PY_MAX_DECODE_HANDLES 32
#define VSPEC_PY_MAX_CONT_BATCH_HANDLES 16
#define VSPEC_PY_MAX_NATIVE_LOOP_HANDLES 16
#define VSPEC_PY_MAX_NATIVE_FORWARD_HANDLES 8

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

typedef struct VspecPyNativeDecodeLoopHandle {
    int active;
    VspecDecodeSession session;
    uint64_t graph_signature;
    uint64_t graph_reuse_hits;
    uint64_t graph_reuse_misses;
    uint64_t graph_captures;
    uint64_t graph_replays;
    uint64_t graph_cache_slots[16];
    size_t graph_cache_count;
    size_t graph_cache_cursor;
    int graph_capture_enabled;
    int graph_replay_active;
    uint64_t steps;
    int started;
} VspecPyNativeDecodeLoopHandle;

static VspecPyNativeDecodeLoopHandle g_native_loop_handles[VSPEC_PY_MAX_NATIVE_LOOP_HANDLES];

typedef struct VspecPyNativeForwardHandle {
    int active;
    VspecCompatModel model;
    VspecNativeForwardContext ctx;
    char model_path[1024];
} VspecPyNativeForwardHandle;

static VspecPyNativeForwardHandle g_native_forward_handles[VSPEC_PY_MAX_NATIVE_FORWARD_HANDLES];

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

static VspecPyNativeDecodeLoopHandle* vspec_py_get_native_loop_handle(int handle_id) {
    if (handle_id <= 0 || handle_id > VSPEC_PY_MAX_NATIVE_LOOP_HANDLES) {
        return NULL;
    }
    if (!g_native_loop_handles[handle_id - 1].active) {
        return NULL;
    }
    return &g_native_loop_handles[handle_id - 1];
}

static VspecPyNativeForwardHandle* vspec_py_get_native_forward_handle(int handle_id) {
    if (handle_id <= 0 || handle_id > VSPEC_PY_MAX_NATIVE_FORWARD_HANDLES) {
        return NULL;
    }
    if (!g_native_forward_handles[handle_id - 1].active) {
        return NULL;
    }
    return &g_native_forward_handles[handle_id - 1];
}

typedef struct VspecPyOutputGuardState {
    int initialized;
    float strictness;
    uint64_t observed_fragments;
} VspecPyOutputGuardState;

static VspecPyOutputGuardState g_py_output_guard = {0, 0.72f, 0U};
static int g_py_runtime_initialized = 0;

static void vspec_py_runtime_ensure_init(void) {
    if (!g_py_runtime_initialized) {
        vspec_runtime_init_default();
        g_py_runtime_initialized = 1;
    }
}

static size_t vspec_py_text_len(const char* s) {
    return s ? strlen(s) : 0U;
}

static int vspec_py_contains_bad_seq(const char* s) {
    if (!s || !s[0]) {
        return 0;
    }
    if (strstr(s, "http") || strstr(s, "www") || strstr(s, "__") || strstr(s, "==") || strstr(s, "@@")) {
        return 1;
    }
    return 0;
}

static float vspec_py_punct_ratio(const char* s) {
    if (!s || !s[0]) {
        return 0.0f;
    }
    size_t punct = 0U;
    size_t total = 0U;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        const unsigned char ch = *p;
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
            continue;
        }
        total += 1U;
        if (!isalnum(ch)) {
            punct += 1U;
        }
    }
    if (total == 0U) {
        return 0.0f;
    }
    return (float)punct / (float)total;
}

static int vspec_py_long_alpha_blob(const char* s) {
    if (!s || !s[0]) {
        return 0;
    }
    const size_t n = strlen(s);
    if (n < 24U) {
        return 0;
    }
    size_t alpha = 0U;
    size_t spaces = 0U;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        if (isalpha(*p)) {
            alpha += 1U;
        }
        if (*p == ' ') {
            spaces += 1U;
        }
    }
    return (spaces == 0U && alpha >= (n * 9U) / 10U) ? 1 : 0;
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

int vspec_py_runtime_output_guard_init(float strictness) {
    if (strictness < 0.0f) {
        strictness = 0.0f;
    }
    if (strictness > 1.0f) {
        strictness = 1.0f;
    }
    g_py_output_guard.strictness = strictness;
    g_py_output_guard.observed_fragments = 0U;
    g_py_output_guard.initialized = 1;
    return 1;
}

int vspec_py_runtime_output_guard_allow(const char* text_fragment) {
    if (!g_py_output_guard.initialized) {
        return 1;
    }
    if (!text_fragment || !text_fragment[0]) {
        return 1;
    }
    if (vspec_py_contains_bad_seq(text_fragment)) {
        return 0;
    }
    if (vspec_py_long_alpha_blob(text_fragment)) {
        return 0;
    }
    const size_t n = vspec_py_text_len(text_fragment);
    const float punct_ratio = vspec_py_punct_ratio(text_fragment);
    const float ratio_cap = 0.35f - (0.15f * g_py_output_guard.strictness);
    if (n >= 12U && punct_ratio > ratio_cap) {
        return 0;
    }
    return 1;
}

float vspec_py_runtime_output_guard_score_adjustment(const char* text_fragment) {
    if (!g_py_output_guard.initialized || !text_fragment || !text_fragment[0]) {
        return 0.0f;
    }
    float score = 0.0f;
    if (vspec_py_contains_bad_seq(text_fragment)) {
        score -= 1.25f;
    }
    if (vspec_py_long_alpha_blob(text_fragment)) {
        score -= 1.10f;
    }
    const float punct_ratio = vspec_py_punct_ratio(text_fragment);
    if (punct_ratio > 0.22f) {
        score -= (0.6f + punct_ratio);
    }
    return score;
}

int vspec_py_runtime_output_guard_observe(const char* text_fragment) {
    if (!g_py_output_guard.initialized) {
        return 1;
    }
    if (text_fragment && text_fragment[0]) {
        g_py_output_guard.observed_fragments += 1U;
    }
    return 1;
}

int vspec_py_runtime_anf_available(void) {
    vspec_py_runtime_ensure_init();
    return vspec_runtime_anf_available();
}

int vspec_py_runtime_anf_observe_activations(const float* activations, size_t count) {
    if (!activations || count == 0U) {
        return 0;
    }
    vspec_py_runtime_ensure_init();
    if (!vspec_runtime_anf_available()) {
        return 0;
    }
    vspec_runtime_anf_observe_token_activations(activations, count);
    return 1;
}

int vspec_py_runtime_anf_observe_quality(float residual_rms, float attention_entropy_collapse, float activation_norm_drift) {
    vspec_py_runtime_ensure_init();
    if (!vspec_runtime_anf_available()) {
        return 0;
    }
    vspec_runtime_behavior_observe_quality(residual_rms, attention_entropy_collapse, activation_norm_drift);
    return 1;
}

int vspec_py_runtime_anf_report(
    int* out_anf_available,
    int* out_anf_mode,
    float* out_hot_ratio,
    uint32_t* out_hot_neurons,
    uint32_t* out_tokens_observed,
    float* out_hot_ratio_avg,
    float* out_skip_ratio_avg,
    uint32_t* out_cache_updates,
    float* out_error_wave_avg,
    float* out_contamination_avg,
    uint32_t* out_cascade_depth,
    uint32_t* out_cascade_depth_max,
    uint32_t* out_forced_fallback_count,
    uint32_t* out_silent_stop_count
) {
    VspecRuntimeBehaviorReport report;

    vspec_py_runtime_ensure_init();
    vspec_runtime_behavior_report(&report);

    if (out_anf_available) {
        *out_anf_available = report.anf_available;
    }
    if (out_anf_mode) {
        *out_anf_mode = report.anf_mode;
    }
    if (out_hot_ratio) {
        *out_hot_ratio = report.anf_hot_ratio;
    }
    if (out_hot_neurons) {
        *out_hot_neurons = report.anf_hot_neurons;
    }
    if (out_tokens_observed) {
        *out_tokens_observed = report.anf_tokens_observed;
    }
    if (out_hot_ratio_avg) {
        *out_hot_ratio_avg = report.anf_hot_ratio_avg;
    }
    if (out_skip_ratio_avg) {
        *out_skip_ratio_avg = report.anf_skip_ratio_avg;
    }
    if (out_cache_updates) {
        *out_cache_updates = report.anf_cache_updates;
    }
    if (out_error_wave_avg) {
        *out_error_wave_avg = report.anf_error_wave_avg;
    }
    if (out_contamination_avg) {
        *out_contamination_avg = report.anf_contamination_avg;
    }
    if (out_cascade_depth) {
        *out_cascade_depth = report.anf_cascade_depth;
    }
    if (out_cascade_depth_max) {
        *out_cascade_depth_max = report.anf_cascade_depth_max;
    }
    if (out_forced_fallback_count) {
        *out_forced_fallback_count = report.anf_forced_fallback_count;
    }
    if (out_silent_stop_count) {
        *out_silent_stop_count = report.anf_silent_stop_count;
    }

    return 1;
}

int vspec_py_native_forward_create(const char* model_path, uint64_t seed) {
    if (!model_path || !model_path[0]) {
        return 0;
    }

    for (int i = 0; i < VSPEC_PY_MAX_NATIVE_FORWARD_HANDLES; ++i) {
        VspecPyNativeForwardHandle* handle = &g_native_forward_handles[i];
        if (handle->active) {
            continue;
        }

        (void)memset(handle, 0, sizeof(*handle));
        if (!vspec_safetensors_parse_header_file(model_path, &handle->model)) {
            (void)memset(handle, 0, sizeof(*handle));
            return 0;
        }

        if (!vspec_native_forward_init(&handle->ctx, &handle->model, model_path, seed)) {
            (void)memset(handle, 0, sizeof(*handle));
            return 0;
        }

        (void)snprintf(handle->model_path, sizeof(handle->model_path), "%s", model_path);
        handle->active = 1;
        return i + 1;
    }
    return 0;
}

void vspec_py_native_forward_destroy(int handle_id) {
    VspecPyNativeForwardHandle* handle = vspec_py_get_native_forward_handle(handle_id);
    if (!handle) {
        return;
    }
    (void)memset(handle, 0, sizeof(*handle));
}

int vspec_py_native_forward_step(
    int handle_id,
    const char* prompt,
    size_t produced_tokens,
    const int* candidate_ids,
    const float* base_scores,
    size_t candidate_count,
    float blend,
    float* out_scores
) {
    VspecPyNativeForwardHandle* handle = vspec_py_get_native_forward_handle(handle_id);
    if (!handle || !prompt || !out_scores || !candidate_ids || candidate_count == 0U) {
        return 0;
    }

    if (blend < 0.0f) {
        blend = 0.0f;
    }
    if (blend > 1.0f) {
        blend = 1.0f;
    }

    float native_scores[VSPEC_NATIVE_TOKEN_COUNT];
    int native_ids[VSPEC_NATIVE_TOKEN_COUNT];
    for (size_t i = 0U; i < VSPEC_NATIVE_TOKEN_COUNT; ++i) {
        native_ids[i] = (int)i;
        native_scores[i] = 0.0f;
    }

    if (!vspec_native_forward_step(
            &handle->ctx,
            prompt,
            produced_tokens,
            native_ids,
            VSPEC_NATIVE_TOKEN_COUNT,
            native_scores)) {
        return 0;
    }

    for (size_t i = 0U; i < candidate_count; ++i) {
        float base = base_scores ? base_scores[i] : 0.0f;
        const int cid = candidate_ids[i];
        float bonus = 0.0f;
        if (cid >= 0 && (size_t)cid < VSPEC_NATIVE_TOKEN_COUNT) {
            bonus = native_scores[(size_t)cid];
        }
        out_scores[i] = base + (blend * bonus);
    }

    return 1;
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

int vspec_py_native_decode_loop_create(
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_tokens,
    size_t token_quantum
) {
    for (int i = 0; i < VSPEC_PY_MAX_NATIVE_LOOP_HANDLES; ++i) {
        VspecPyNativeDecodeLoopHandle* handle = &g_native_loop_handles[i];
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
        {
            const char* enabled = getenv("VSPEC_CUDA_GRAPH_CAPTURE");
            handle->graph_capture_enabled = 1;
            if (enabled && enabled[0] != '\0') {
                if (strcmp(enabled, "0") == 0 || strcmp(enabled, "false") == 0 || strcmp(enabled, "False") == 0 || strcmp(enabled, "no") == 0 || strcmp(enabled, "off") == 0) {
                    handle->graph_capture_enabled = 0;
                }
            }
        }
        handle->active = 1;
        return i + 1;
    }
    return 0;
}

void vspec_py_native_decode_loop_destroy(int handle_id) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return;
    }
    if (vspec_decode_session_is_active(&handle->session)) {
        (void)vspec_decode_session_cancel(&handle->session);
    }
    (void)memset(handle, 0, sizeof(*handle));
}

int vspec_py_native_decode_loop_begin(
    int handle_id,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority,
    uint64_t graph_signature
) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return 0;
    }

    if (handle->graph_capture_enabled) {
        int seen = 0;
        for (size_t i = 0U; i < handle->graph_cache_count; ++i) {
            if (handle->graph_cache_slots[i] == graph_signature) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            handle->graph_captures += 1U;
            if (handle->graph_cache_count < 16U) {
                handle->graph_cache_slots[handle->graph_cache_count++] = graph_signature;
            } else {
                handle->graph_cache_slots[handle->graph_cache_cursor] = graph_signature;
                handle->graph_cache_cursor = (handle->graph_cache_cursor + 1U) % 16U;
            }
        }
        handle->graph_replay_active = (graph_signature != 0U) ? 1 : 0;
    } else {
        handle->graph_replay_active = 0;
    }

    if (handle->started) {
        if (handle->graph_signature == graph_signature) {
            handle->graph_reuse_hits += 1U;
        } else {
            handle->graph_reuse_misses += 1U;
        }
    } else {
        handle->graph_reuse_misses += 1U;
    }
    handle->graph_signature = graph_signature;
    handle->started = 1;
    return vspec_decode_session_begin(&handle->session, reserve_bytes, prompt_tokens, max_new_tokens, priority);
}

size_t vspec_py_native_decode_loop_next_quota(int handle_id) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return 0U;
    }
    return vspec_decode_session_next_quota(&handle->session);
}

int vspec_py_native_decode_loop_commit(int handle_id, size_t generated_tokens, int reached_eos) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return 0;
    }
    if (handle->graph_capture_enabled && handle->graph_replay_active) {
        const uint64_t replay_steps = (generated_tokens > 0U) ? (uint64_t)generated_tokens : 1U;
        handle->graph_replays += replay_steps;
    }
    handle->steps += 1U;
    return vspec_decode_session_commit(&handle->session, generated_tokens, reached_eos);
}

int vspec_py_native_decode_loop_cancel(int handle_id) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return 0;
    }
    return vspec_decode_session_cancel(&handle->session);
}

int vspec_py_native_decode_loop_stats(
    int handle_id,
    uint64_t* out_graph_signature,
    uint64_t* out_graph_reuse_hits,
    uint64_t* out_graph_reuse_misses,
    uint64_t* out_steps
) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return 0;
    }
    if (out_graph_signature) {
        *out_graph_signature = handle->graph_signature;
    }
    if (out_graph_reuse_hits) {
        *out_graph_reuse_hits = handle->graph_reuse_hits;
    }
    if (out_graph_reuse_misses) {
        *out_graph_reuse_misses = handle->graph_reuse_misses;
    }
    if (out_steps) {
        *out_steps = handle->steps;
    }
    return 1;
}

int vspec_py_native_decode_loop_graph_cache_stats(
    int handle_id,
    uint64_t* out_graph_captures,
    uint64_t* out_graph_replays,
    uint64_t* out_cached_signatures,
    int* out_graph_capture_enabled
) {
    VspecPyNativeDecodeLoopHandle* handle = vspec_py_get_native_loop_handle(handle_id);
    if (!handle) {
        return 0;
    }
    if (out_graph_captures) {
        *out_graph_captures = handle->graph_captures;
    }
    if (out_graph_replays) {
        *out_graph_replays = handle->graph_replays;
    }
    if (out_cached_signatures) {
        *out_cached_signatures = (uint64_t)handle->graph_cache_count;
    }
    if (out_graph_capture_enabled) {
        *out_graph_capture_enabled = handle->graph_capture_enabled;
    }
    return 1;
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
