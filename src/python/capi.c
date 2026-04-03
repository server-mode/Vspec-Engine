#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

#include "vspec/python/capi.h"
#include "vspec/attention/kv_paged_cache.h"
#include "vspec/compat/pytorch_loader.h"
#include "vspec/compat/safetensors_parser.h"
#include "vspec/compat/graph_rewrite.h"
#include "vspec/compat/weight_mapper.h"
#include "vspec/graph/ir.h"
#include "vspec/model/qwen_ops.h"
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
#define VSPEC_PY_NATIVE_FULL_MAX_LAYERS 128
#define VSPEC_PY_NATIVE_FILE_CACHE_MAX 32

typedef struct VspecPyTensorLocation {
    int ready;
    VspecCompatTensorInfo info;
    char file_path[1024];
    uint64_t data_base;
} VspecPyTensorLocation;

typedef struct VspecPyNativeFileCacheEntry {
    int used;
    char file_path[1024];
    FILE* file;
} VspecPyNativeFileCacheEntry;

typedef struct VspecPyNativeLayerTensorSet {
    int ready;
    VspecPyTensorLocation input_layernorm;
    VspecPyTensorLocation post_attention_layernorm;
    VspecPyTensorLocation q_proj;
    VspecPyTensorLocation k_proj;
    VspecPyTensorLocation v_proj;
    VspecPyTensorLocation o_proj;
    VspecPyTensorLocation gate_proj;
    VspecPyTensorLocation up_proj;
    VspecPyTensorLocation down_proj;
} VspecPyNativeLayerTensorSet;

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
    int token_forward_ready;
    int tied_lm_head;
    uint64_t data_base;
    VspecCompatTensorInfo embed_info;
    VspecCompatTensorInfo lm_head_info;
    int full_forward_ready;
    size_t full_layer_count;
    size_t full_layer_limit;
    size_t full_context_limit;
    size_t hidden_size;
    size_t head_dim;
    size_t num_heads;
    size_t num_kv_heads;
    VspecPyTensorLocation full_embed;
    VspecPyTensorLocation full_lm_head;
    VspecPyTensorLocation final_norm_info;
    VspecPyNativeLayerTensorSet* full_layers;
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

static int vspec_py_native_forward_prepare_token_cache(VspecPyNativeForwardHandle* handle);

static uint64_t vspec_py_read_u64_le_local(const unsigned char b[8]) {
    uint64_t v = 0U;
    for (size_t i = 0U; i < 8U; ++i) {
        v |= ((uint64_t)b[i]) << (8U * i);
    }
    return v;
}

static uint16_t vspec_py_read_u16_le_local(const unsigned char b[2]) {
    return (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8U));
}

static float vspec_py_f16_to_f32_local(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t frac = (uint32_t)h & 0x3FFu;
    if (exp == 0U) {
        if (frac == 0U) {
            return sign ? -0.0f : 0.0f;
        }
        {
            const float m = (float)frac / 1024.0f;
            const float v = ldexpf(m, -14);
            return sign ? -v : v;
        }
    }
    if (exp == 31U) {
        return sign ? -65504.0f : 65504.0f;
    }
    {
        const float m = 1.0f + ((float)frac / 1024.0f);
        const float v = ldexpf(m, (int)exp - 15);
        return sign ? -v : v;
    }
}

static float vspec_py_bf16_to_f32_local(uint16_t b16) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)b16) << 16U;
    return v.f;
}

static size_t vspec_py_dtype_bytes_local(const char* dtype) {
    if (!dtype) return 0U;
    if (strcmp(dtype, "F16") == 0) return 2U;
    if (strcmp(dtype, "BF16") == 0) return 2U;
    if (strcmp(dtype, "F32") == 0) return 4U;
    return 0U;
}

static float vspec_py_decode_scalar_local(const unsigned char* bytes, const char* dtype) {
    if (!bytes || !dtype) {
        return 0.0f;
    }
    if (strcmp(dtype, "F16") == 0) {
        return vspec_py_f16_to_f32_local(vspec_py_read_u16_le_local(bytes));
    }
    if (strcmp(dtype, "BF16") == 0) {
        return vspec_py_bf16_to_f32_local(vspec_py_read_u16_le_local(bytes));
    }
    if (strcmp(dtype, "F32") == 0) {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = (uint32_t)bytes[0]
            | ((uint32_t)bytes[1] << 8U)
            | ((uint32_t)bytes[2] << 16U)
            | ((uint32_t)bytes[3] << 24U);
        return v.f;
    }
    return 0.0f;
}

static int vspec_py_seek64_local(FILE* f, uint64_t offset) {
    if (!f) {
        return 0;
    }
#ifdef _WIN32
    return (_fseeki64(f, (long long)offset, SEEK_SET) == 0) ? 1 : 0;
#else
    return (fseeko(f, (off_t)offset, SEEK_SET) == 0) ? 1 : 0;
#endif
}

static int vspec_py_native_forward_find_tensor(
    const VspecCompatModel* model,
    const char* canonical_name,
    VspecCompatTensorInfo* out_info
) {
    if (!model || !canonical_name || !out_info) {
        return 0;
    }
    for (size_t i = 0U; i < model->tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        if (!vspec_weight_canonical_name(model->tensors[i].name, canonical, sizeof(canonical))) {
            continue;
        }
        if (strcmp(canonical, canonical_name) == 0) {
            *out_info = model->tensors[i];
            return 1;
        }
    }
    return 0;
}

static int vspec_py_file_exists_local(const char* path) {
    struct _stat st;
    if (!path || !path[0]) {
        return 0;
    }
    if (_stat(path, &st) != 0) {
        return 0;
    }
    return ((st.st_mode & _S_IFREG) != 0) ? 1 : 0;
}

static int vspec_py_dirname_local(const char* path, char* out_dir, size_t out_cap) {
    const char* slash = NULL;
    const char* bslash = NULL;
    const char* sep = NULL;
    size_t n = 0U;
    if (!path || !path[0] || !out_dir || out_cap == 0U) {
        return 0;
    }
    slash = strrchr(path, '/');
    bslash = strrchr(path, '\\');
    sep = (slash && bslash) ? ((slash > bslash) ? slash : bslash) : (slash ? slash : bslash);
    if (!sep) {
        return 0;
    }
    n = (size_t)(sep - path);
    if (n == 0U || n >= out_cap) {
        return 0;
    }
    (void)memcpy(out_dir, path, n);
    out_dir[n] = '\0';
    return 1;
}

static int vspec_py_has_safetensors_suffix_local(const char* name) {
    size_t n = 0U;
    static const char* suffix = ".safetensors";
    size_t m = strlen(suffix);
    if (!name) {
        return 0;
    }
    n = strlen(name);
    if (n < m) {
        return 0;
    }
#ifdef _WIN32
    return (_stricmp(name + (n - m), suffix) == 0) ? 1 : 0;
#else
    return (strcmp(name + (n - m), suffix) == 0) ? 1 : 0;
#endif
}

static int vspec_py_read_data_base_local(const char* file_path, uint64_t* out_data_base) {
    FILE* f = NULL;
    unsigned char hdr[8];
    if (!file_path || !out_data_base) {
        return 0;
    }
    f = fopen(file_path, "rb");
    if (!f) {
        return 0;
    }
    if (fread(hdr, 1, 8U, f) != 8U) {
        fclose(f);
        return 0;
    }
    fclose(f);
    *out_data_base = 8ULL + vspec_py_read_u64_le_local(hdr);
    return 1;
}

static int vspec_py_find_tensor_location_in_file(
    const char* file_path,
    const char* canonical_name,
    VspecPyTensorLocation* out_loc
) {
    VspecCompatModel model;
    if (!file_path || !canonical_name || !out_loc) {
        return 0;
    }
    if (!vspec_safetensors_parse_header_file(file_path, &model)) {
        return 0;
    }
    for (size_t i = 0U; i < model.tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        if (!vspec_weight_canonical_name(model.tensors[i].name, canonical, sizeof(canonical))) {
            continue;
        }
        if (strcmp(canonical, canonical_name) == 0) {
            uint64_t data_base = 0U;
            if (!vspec_py_read_data_base_local(file_path, &data_base)) {
                return 0;
            }
            (void)memset(out_loc, 0, sizeof(*out_loc));
            out_loc->ready = 1;
            out_loc->info = model.tensors[i];
            out_loc->data_base = data_base;
            (void)snprintf(out_loc->file_path, sizeof(out_loc->file_path), "%s", file_path);
            return 1;
        }
    }
    return 0;
}

static int vspec_py_find_tensor_location_any_file(
    const VspecPyNativeForwardHandle* handle,
    const char* canonical_name,
    VspecPyTensorLocation* out_loc
) {
    char dir_path[1024];
    if (!handle || !canonical_name || !out_loc || !handle->model_path[0]) {
        return 0;
    }

    if (vspec_py_find_tensor_location_in_file(handle->model_path, canonical_name, out_loc)) {
        return 1;
    }

    if (!vspec_py_dirname_local(handle->model_path, dir_path, sizeof(dir_path))) {
        return 0;
    }

#ifdef _WIN32
    {
        char pattern[1200];
        WIN32_FIND_DATAA data;
        HANDLE h = INVALID_HANDLE_VALUE;
        if (snprintf(pattern, sizeof(pattern), "%s\\*.safetensors", dir_path) >= (int)sizeof(pattern)) {
            return 0;
        }
        h = FindFirstFileA(pattern, &data);
        if (h == INVALID_HANDLE_VALUE) {
            return 0;
        }
        do {
            char file_path[1200];
            if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
                continue;
            }
            if (!vspec_py_has_safetensors_suffix_local(data.cFileName)) {
                continue;
            }
            if (snprintf(file_path, sizeof(file_path), "%s\\%s", dir_path, data.cFileName) >= (int)sizeof(file_path)) {
                continue;
            }
            if (_stricmp(file_path, handle->model_path) == 0) {
                continue;
            }
            if (vspec_py_find_tensor_location_in_file(file_path, canonical_name, out_loc)) {
                FindClose(h);
                return 1;
            }
        } while (FindNextFileA(h, &data));
        FindClose(h);
    }
#else
    {
        DIR* d = opendir(dir_path);
        if (!d) {
            return 0;
        }
        for (;;) {
            struct dirent* e = readdir(d);
            char file_path[1200];
            if (!e) {
                break;
            }
            if (!vspec_py_has_safetensors_suffix_local(e->d_name)) {
                continue;
            }
            if (snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, e->d_name) >= (int)sizeof(file_path)) {
                continue;
            }
            if (strcmp(file_path, handle->model_path) == 0) {
                continue;
            }
            if (vspec_py_find_tensor_location_in_file(file_path, canonical_name, out_loc)) {
                closedir(d);
                return 1;
            }
        }
        closedir(d);
    }
#endif
    return 0;
}

static FILE* vspec_py_get_cached_file_local(
    VspecPyNativeFileCacheEntry* cache,
    size_t cache_cap,
    const char* file_path
) {
    size_t free_idx = (size_t)(-1);
    if (!cache || cache_cap == 0U || !file_path || !file_path[0]) {
        return NULL;
    }
    for (size_t i = 0U; i < cache_cap; ++i) {
        if (cache[i].used && cache[i].file && strcmp(cache[i].file_path, file_path) == 0) {
            return cache[i].file;
        }
        if (!cache[i].used && free_idx == (size_t)(-1)) {
            free_idx = i;
        }
    }
    if (free_idx == (size_t)(-1)) {
        return NULL;
    }
    cache[free_idx].file = fopen(file_path, "rb");
    if (!cache[free_idx].file) {
        return NULL;
    }
    cache[free_idx].used = 1;
    (void)snprintf(cache[free_idx].file_path, sizeof(cache[free_idx].file_path), "%s", file_path);
    return cache[free_idx].file;
}

static void vspec_py_close_cached_files_local(VspecPyNativeFileCacheEntry* cache, size_t cache_cap) {
    if (!cache) {
        return;
    }
    for (size_t i = 0U; i < cache_cap; ++i) {
        if (cache[i].used && cache[i].file) {
            fclose(cache[i].file);
        }
        cache[i].used = 0;
        cache[i].file = NULL;
        cache[i].file_path[0] = '\0';
    }
}

static void vspec_py_native_forward_release_full_cache(VspecPyNativeForwardHandle* handle) {
    if (!handle) {
        return;
    }
    if (handle->full_layers) {
        free(handle->full_layers);
        handle->full_layers = NULL;
    }
    handle->full_forward_ready = 0;
    handle->full_layer_count = 0U;
    handle->full_layer_limit = 0U;
    handle->full_context_limit = 0U;
    handle->hidden_size = 0U;
    handle->head_dim = 0U;
    handle->num_heads = 0U;
    handle->num_kv_heads = 0U;
    (void)memset(&handle->final_norm_info, 0, sizeof(handle->final_norm_info));
}

static int vspec_py_env_true_local(const char* name) {
    const char* v = getenv(name);
    if (!v || !v[0]) {
        return 0;
    }
    return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static size_t vspec_py_env_size_local(const char* name, size_t default_value, size_t max_value) {
    const char* v = getenv(name);
    size_t out = default_value;
    if (v && v[0]) {
        unsigned long long parsed = strtoull(v, NULL, 10);
        out = (size_t)parsed;
    }
    if (out > max_value) {
        out = max_value;
    }
    return out;
}

#define VSPEC_NATIVE_DBG(...) do { if (vspec_py_env_true_local("VSPEC_NATIVE_DEBUG")) { fprintf(stderr, __VA_ARGS__); } } while (0)

static size_t vspec_py_infer_head_dim_local(size_t q_dim, size_t kv_dim) {
    static const size_t preferred[] = {256U, 192U, 160U, 128U, 96U, 80U, 64U, 48U, 40U, 32U, 16U, 8U};
    for (size_t i = 0U; i < sizeof(preferred) / sizeof(preferred[0]); ++i) {
        const size_t d = preferred[i];
        if (d == 0U) {
            continue;
        }
        if ((q_dim % d) == 0U && (kv_dim % d) == 0U) {
            return d;
        }
    }
    return 0U;
}

static int vspec_py_native_forward_prepare_full_cache(VspecPyNativeForwardHandle* handle) {
    size_t detected_layers = 0U;
    size_t ready_layers = 0U;
    size_t hidden = 0U;
    VspecPyNativeLayerTensorSet* layers = NULL;

    if (!handle) {
        return 0;
    }

    vspec_py_native_forward_release_full_cache(handle);

    if (!vspec_py_env_true_local("VSPEC_NATIVE_FULL_TRANSFORMER")) {
        return 0;
    }
    if (!handle->token_forward_ready && !vspec_py_native_forward_prepare_token_cache(handle)) {
        return 0;
    }

    if (!vspec_py_find_tensor_location_any_file(handle, "model.embed_tokens.weight", &handle->full_embed)) {
        return 0;
    }
    if (!vspec_py_find_tensor_location_any_file(handle, "lm_head.weight", &handle->full_lm_head)) {
        handle->full_lm_head = handle->full_embed;
    }

    if (!handle->full_embed.ready || handle->full_embed.info.ndim < 2U) {
        return 0;
    }
    if (!handle->full_lm_head.ready || handle->full_lm_head.info.ndim < 2U) {
        return 0;
    }
    if (handle->full_lm_head.info.shape[1] != handle->full_embed.info.shape[1]) {
        return 0;
    }

    (void)vspec_py_find_tensor_location_any_file(handle, "model.norm.weight", &handle->final_norm_info);

    hidden = handle->full_embed.info.shape[1];
    if (hidden == 0U) {
        return 0;
    }

    detected_layers = VSPEC_PY_NATIVE_FULL_MAX_LAYERS;
    if (detected_layers == 0U || detected_layers > VSPEC_PY_NATIVE_FULL_MAX_LAYERS) {
        return 0;
    }

    layers = (VspecPyNativeLayerTensorSet*)calloc(detected_layers, sizeof(VspecPyNativeLayerTensorSet));
    if (!layers) {
        return 0;
    }

    for (size_t li = 0U; li < detected_layers; ++li) {
        char name[VSPEC_COMPAT_NAME_MAX];
        VspecPyNativeLayerTensorSet* layer = &layers[li];

        (void)snprintf(name, sizeof(name), "model.layers.%zu.input_layernorm.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->input_layernorm)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.post_attention_layernorm.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->post_attention_layernorm)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.self_attn.q_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->q_proj)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.self_attn.k_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->k_proj)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.self_attn.v_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->v_proj)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.self_attn.o_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->o_proj)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.mlp.gate_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->gate_proj)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.mlp.up_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->up_proj)) {
            break;
        }
        (void)snprintf(name, sizeof(name), "model.layers.%zu.mlp.down_proj.weight", li);
        if (!vspec_py_find_tensor_location_any_file(handle, name, &layer->down_proj)) {
            break;
        }

        if (layer->input_layernorm.info.ndim < 1U || layer->post_attention_layernorm.info.ndim < 1U) {
            break;
        }
        if (layer->input_layernorm.info.shape[0] != hidden || layer->post_attention_layernorm.info.shape[0] != hidden) {
            break;
        }
        if (layer->q_proj.info.ndim < 2U || layer->k_proj.info.ndim < 2U || layer->v_proj.info.ndim < 2U || layer->o_proj.info.ndim < 2U) {
            break;
        }
        if (layer->q_proj.info.shape[1] != hidden || layer->k_proj.info.shape[1] != hidden || layer->v_proj.info.shape[1] != hidden) {
            break;
        }
        if (layer->o_proj.info.shape[0] != hidden || layer->o_proj.info.shape[1] != layer->q_proj.info.shape[0]) {
            break;
        }
        if (layer->gate_proj.info.ndim < 2U || layer->up_proj.info.ndim < 2U || layer->down_proj.info.ndim < 2U) {
            break;
        }
        if (layer->gate_proj.info.shape[1] != hidden || layer->up_proj.info.shape[1] != hidden) {
            break;
        }
        if (layer->gate_proj.info.shape[0] != layer->up_proj.info.shape[0]) {
            break;
        }
        if (layer->down_proj.info.shape[0] != hidden || layer->down_proj.info.shape[1] != layer->gate_proj.info.shape[0]) {
            break;
        }

        layer->ready = 1;
        ready_layers += 1U;
    }

    if (ready_layers == 0U) {
        free(layers);
        return 0;
    }

    detected_layers = ready_layers;

    if (handle->final_norm_info.ready && handle->final_norm_info.info.ndim >= 1U && handle->final_norm_info.info.shape[0] != hidden) {
        (void)memset(&handle->final_norm_info, 0, sizeof(handle->final_norm_info));
    }

    {
        const size_t q_dim = layers[0].q_proj.info.shape[0];
        const size_t kv_dim = layers[0].k_proj.info.shape[0];
        const size_t head_dim = vspec_py_infer_head_dim_local(q_dim, kv_dim);
        if (head_dim == 0U) {
            free(layers);
            return 0;
        }
        handle->head_dim = head_dim;
        handle->num_heads = q_dim / head_dim;
        handle->num_kv_heads = kv_dim / head_dim;
        if (handle->num_heads == 0U || handle->num_kv_heads == 0U) {
            free(layers);
            return 0;
        }
    }

    handle->hidden_size = hidden;
    handle->full_layer_count = detected_layers;
    handle->full_layer_limit = vspec_py_env_size_local("VSPEC_NATIVE_FULL_LAYER_LIMIT", 0U, detected_layers);
    handle->full_context_limit = vspec_py_env_size_local("VSPEC_NATIVE_FULL_CONTEXT_LIMIT", 0U, 16384U);
    handle->full_layers = layers;
    handle->full_forward_ready = 1;
    return 1;
}

static int vspec_py_native_forward_prepare_token_cache(VspecPyNativeForwardHandle* handle) {
    if (!handle || !handle->model_path[0]) {
        VSPEC_NATIVE_DBG("[native][prepare_token_cache] invalid_handle_or_model_path\n");
        return 0;
    }

    if (!vspec_py_find_tensor_location_any_file(handle, "model.embed_tokens.weight", &handle->full_embed)) {
        VSPEC_NATIVE_DBG("[native][prepare_token_cache] embed_not_found model_path=%s\n", handle->model_path);
        return 0;
    }
    if (!vspec_py_find_tensor_location_any_file(handle, "lm_head.weight", &handle->full_lm_head)) {
        handle->full_lm_head = handle->full_embed;
        handle->tied_lm_head = 1;
        VSPEC_NATIVE_DBG("[native][prepare_token_cache] lm_head_not_found_using_tied_embed file=%s\n", handle->full_embed.file_path);
    } else {
        handle->tied_lm_head = 0;
    }

    handle->embed_info = handle->full_embed.info;
    handle->lm_head_info = handle->full_lm_head.info;
    handle->data_base = handle->full_embed.data_base;

    if (handle->embed_info.ndim < 2U || handle->lm_head_info.ndim < 2U) {
        VSPEC_NATIVE_DBG("[native][prepare_token_cache] bad_ndim embed_ndim=%zu lm_ndim=%zu\n", handle->embed_info.ndim, handle->lm_head_info.ndim);
        return 0;
    }
    if (handle->embed_info.shape[0] == 0U || handle->embed_info.shape[1] == 0U) {
        VSPEC_NATIVE_DBG("[native][prepare_token_cache] bad_embed_shape=[%zu,%zu]\n", handle->embed_info.shape[0], handle->embed_info.shape[1]);
        return 0;
    }
    if (handle->lm_head_info.shape[0] != handle->embed_info.shape[0] || handle->lm_head_info.shape[1] != handle->embed_info.shape[1]) {
        VSPEC_NATIVE_DBG(
            "[native][prepare_token_cache] shape_mismatch embed=[%zu,%zu] lm=[%zu,%zu] lm_file=%s\n",
            handle->embed_info.shape[0],
            handle->embed_info.shape[1],
            handle->lm_head_info.shape[0],
            handle->lm_head_info.shape[1],
            handle->full_lm_head.file_path);
        return 0;
    }
    if (vspec_py_dtype_bytes_local(handle->embed_info.dtype) == 0U || vspec_py_dtype_bytes_local(handle->lm_head_info.dtype) == 0U) {
        VSPEC_NATIVE_DBG("[native][prepare_token_cache] unsupported_dtype embed=%s lm=%s\n", handle->embed_info.dtype, handle->lm_head_info.dtype);
        return 0;
    }

    handle->token_forward_ready = 1;
    VSPEC_NATIVE_DBG(
        "[native][prepare_token_cache] ready embed_file=%s lm_file=%s embed=[%zu,%zu]\n",
        handle->full_embed.file_path,
        handle->full_lm_head.file_path,
        handle->embed_info.shape[0],
        handle->embed_info.shape[1]);
    return 1;
}

static int vspec_py_native_forward_read_row(
    FILE* f,
    uint64_t data_base,
    const VspecCompatTensorInfo* info,
    int row_id,
    float* out_row,
    size_t dim
) {
    const size_t elem_bytes = vspec_py_dtype_bytes_local(info ? info->dtype : NULL);
    unsigned char* raw = NULL;
    uint64_t row_off = 0U;

    if (!f || !info || !out_row || dim == 0U || elem_bytes == 0U || row_id < 0) {
        return 0;
    }
    if ((uint64_t)row_id >= info->shape[0] || info->shape[1] != dim) {
        return 0;
    }

    raw = (unsigned char*)malloc(dim * elem_bytes);
    if (!raw) {
        return 0;
    }

    row_off = data_base + info->data_offset_start + ((uint64_t)(size_t)row_id * (uint64_t)dim * (uint64_t)elem_bytes);
    if (!vspec_py_seek64_local(f, row_off)) {
        free(raw);
        return 0;
    }
    if (fread(raw, 1, dim * elem_bytes, f) != dim * elem_bytes) {
        free(raw);
        return 0;
    }

    for (size_t i = 0U; i < dim; ++i) {
        out_row[i] = vspec_py_decode_scalar_local(&raw[i * elem_bytes], info->dtype);
    }
    free(raw);
    return 1;
}

static int vspec_py_native_forward_read_vector(
    FILE* f,
    uint64_t data_base,
    const VspecCompatTensorInfo* info,
    float* out_vec,
    size_t dim
) {
    const size_t elem_bytes = vspec_py_dtype_bytes_local(info ? info->dtype : NULL);
    unsigned char* raw = NULL;
    uint64_t off = 0U;

    if (!f || !info || !out_vec || dim == 0U || elem_bytes == 0U) {
        return 0;
    }
    if (info->ndim < 1U || info->shape[0] != dim) {
        return 0;
    }

    raw = (unsigned char*)malloc(dim * elem_bytes);
    if (!raw) {
        return 0;
    }

    off = data_base + info->data_offset_start;
    if (!vspec_py_seek64_local(f, off)) {
        free(raw);
        return 0;
    }
    if (fread(raw, 1, dim * elem_bytes, f) != dim * elem_bytes) {
        free(raw);
        return 0;
    }
    for (size_t i = 0U; i < dim; ++i) {
        out_vec[i] = vspec_py_decode_scalar_local(&raw[i * elem_bytes], info->dtype);
    }
    free(raw);
    return 1;
}

static int vspec_py_native_forward_matvec(
    FILE* f,
    uint64_t data_base,
    const VspecCompatTensorInfo* info,
    const float* input,
    size_t input_dim,
    float* output,
    size_t output_dim,
    float* row_buf,
    unsigned char* raw_buf,
    size_t raw_buf_cap
) {
    const size_t elem_bytes = vspec_py_dtype_bytes_local(info ? info->dtype : NULL);
    const size_t row_bytes = input_dim * elem_bytes;
    if (!f || !info || !input || !output || !row_buf || !raw_buf || input_dim == 0U || output_dim == 0U || elem_bytes == 0U) {
        return 0;
    }
    if (info->ndim < 2U || info->shape[0] < output_dim || info->shape[1] != input_dim) {
        return 0;
    }
    if (raw_buf_cap < row_bytes) {
        return 0;
    }

    for (size_t row = 0U; row < output_dim; ++row) {
        uint64_t row_off = data_base + info->data_offset_start + ((uint64_t)row * (uint64_t)input_dim * (uint64_t)elem_bytes);
        double acc = 0.0;
        if (!vspec_py_seek64_local(f, row_off)) {
            return 0;
        }
        if (fread(raw_buf, 1, row_bytes, f) != row_bytes) {
            return 0;
        }
        for (size_t j = 0U; j < input_dim; ++j) {
            row_buf[j] = vspec_py_decode_scalar_local(&raw_buf[j * elem_bytes], info->dtype);
        }
        for (size_t j = 0U; j < input_dim; ++j) {
            acc += (double)input[j] * (double)row_buf[j];
        }
        output[row] = (float)acc;
    }
    return 1;
}

static int vspec_py_native_forward_read_row_loc(
    VspecPyNativeFileCacheEntry* cache,
    size_t cache_cap,
    const VspecPyTensorLocation* loc,
    int row_id,
    float* out_row,
    size_t dim
) {
    FILE* f = NULL;
    int ok = 0;
    if (!loc || !loc->ready) {
        VSPEC_NATIVE_DBG("[native][read_row_loc] invalid_loc_or_not_ready\n");
        return 0;
    }
    f = vspec_py_get_cached_file_local(cache, cache_cap, loc->file_path);
    if (!f) {
        VSPEC_NATIVE_DBG("[native][read_row_loc] open_failed file=%s\n", loc->file_path);
        return 0;
    }
    ok = vspec_py_native_forward_read_row(f, loc->data_base, &loc->info, row_id, out_row, dim);
    if (!ok) {
        VSPEC_NATIVE_DBG(
            "[native][read_row_loc] read_failed file=%s row=%d dim=%zu tensor_shape=[%zu,%zu] dtype=%s data_base=%llu off_start=%llu\n",
            loc->file_path,
            row_id,
            dim,
            loc->info.shape[0],
            loc->info.shape[1],
            loc->info.dtype,
            (unsigned long long)loc->data_base,
            (unsigned long long)loc->info.data_offset_start);
    }
    return ok;
}

static int vspec_py_native_forward_read_vector_loc(
    VspecPyNativeFileCacheEntry* cache,
    size_t cache_cap,
    const VspecPyTensorLocation* loc,
    float* out_vec,
    size_t dim
) {
    FILE* f = NULL;
    if (!loc || !loc->ready) {
        return 0;
    }
    f = vspec_py_get_cached_file_local(cache, cache_cap, loc->file_path);
    if (!f) {
        return 0;
    }
    return vspec_py_native_forward_read_vector(f, loc->data_base, &loc->info, out_vec, dim);
}

static int vspec_py_native_forward_matvec_loc(
    VspecPyNativeFileCacheEntry* cache,
    size_t cache_cap,
    const VspecPyTensorLocation* loc,
    const float* input,
    size_t input_dim,
    float* output,
    size_t output_dim,
    float* row_buf,
    unsigned char* raw_buf,
    size_t raw_buf_cap
) {
    FILE* f = NULL;
    if (!loc || !loc->ready) {
        return 0;
    }
    f = vspec_py_get_cached_file_local(cache, cache_cap, loc->file_path);
    if (!f) {
        return 0;
    }
    return vspec_py_native_forward_matvec(
        f,
        loc->data_base,
        &loc->info,
        input,
        input_dim,
        output,
        output_dim,
        row_buf,
        raw_buf,
        raw_buf_cap);
}

static void vspec_py_apply_rotary_local(float* vec, size_t head_dim, size_t position) {
    const float base = 10000.0f;
    if (!vec || head_dim < 2U) {
        return;
    }
    for (size_t d = 0U; d + 1U < head_dim; d += 2U) {
        const float dim_scale = (float)d / (float)head_dim;
        const float inv_freq = powf(base, -dim_scale);
        const float angle = (float)position * inv_freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = vec[d];
        const float x1 = vec[d + 1U];
        vec[d] = (x0 * c) - (x1 * s);
        vec[d + 1U] = (x0 * s) + (x1 * c);
    }
}

static int vspec_py_native_forward_step_tokens_full_impl(
    VspecPyNativeForwardHandle* handle,
    const int* context_token_ids,
    size_t context_token_count,
    const int* candidate_ids,
    const float* base_scores,
    size_t candidate_count,
    float blend,
    float* out_scores
) {
    VspecPyNativeFileCacheEntry file_cache[VSPEC_PY_NATIVE_FILE_CACHE_MAX];
    float* states = NULL;
    float* norm_states = NULL;
    float* q_states = NULL;
    float* k_states = NULL;
    float* v_states = NULL;
    float* attn_states = NULL;
    float* norm_weight = NULL;
    float* post_norm_weight = NULL;
    float* final_norm_weight = NULL;
    float* gate = NULL;
    float* up = NULL;
    float* down = NULL;
    float* temp_out = NULL;
    float* row_buf = NULL;
    float* score_buf = NULL;
    unsigned char* raw_buf = NULL;
    size_t ctx_start = 0U;
    size_t ctx_count = context_token_count;
    size_t layer_count = 0U;
    size_t hidden = 0U;
    size_t q_dim = 0U;
    size_t kv_dim = 0U;
    size_t inter_dim = 0U;
    size_t head_dim = 0U;
    size_t num_heads = 0U;
    size_t num_kv_heads = 0U;
    size_t max_in_dim = 0U;
    size_t max_raw_bytes = 0U;
    int ok = 0;

    (void)memset(file_cache, 0, sizeof(file_cache));

    if (!handle || !context_token_ids || context_token_count == 0U || !candidate_ids || candidate_count == 0U || !out_scores) {
        return 0;
    }
    if (!handle->full_forward_ready && !vspec_py_native_forward_prepare_full_cache(handle)) {
        return 0;
    }
    if (!handle->full_forward_ready || !handle->full_layers || handle->full_layer_count == 0U) {
        return 0;
    }

    hidden = handle->hidden_size;
    head_dim = handle->head_dim;
    num_heads = handle->num_heads;
    num_kv_heads = handle->num_kv_heads;
    q_dim = handle->full_layers[0].q_proj.info.shape[0];
    kv_dim = handle->full_layers[0].k_proj.info.shape[0];
    inter_dim = handle->full_layers[0].gate_proj.info.shape[0];
    if (hidden == 0U || q_dim == 0U || kv_dim == 0U || inter_dim == 0U || head_dim == 0U || num_heads == 0U || num_kv_heads == 0U) {
        return 0;
    }
    if ((num_heads * head_dim) != q_dim || (num_kv_heads * head_dim) != kv_dim) {
        return 0;
    }

    if (handle->full_context_limit > 0U && ctx_count > handle->full_context_limit) {
        ctx_start = ctx_count - handle->full_context_limit;
        ctx_count = handle->full_context_limit;
    }
    if (ctx_count == 0U) {
        return 0;
    }

    layer_count = handle->full_layer_count;
    if (handle->full_layer_limit > 0U && layer_count > handle->full_layer_limit) {
        layer_count = handle->full_layer_limit;
    }
    if (layer_count == 0U) {
        return 0;
    }

    max_in_dim = hidden;
    if (q_dim > max_in_dim) {
        max_in_dim = q_dim;
    }
    if (kv_dim > max_in_dim) {
        max_in_dim = kv_dim;
    }
    if (inter_dim > max_in_dim) {
        max_in_dim = inter_dim;
    }
    max_raw_bytes = max_in_dim * 4U;

    states = (float*)malloc(ctx_count * hidden * sizeof(float));
    norm_states = (float*)malloc(ctx_count * hidden * sizeof(float));
    q_states = (float*)malloc(ctx_count * q_dim * sizeof(float));
    k_states = (float*)malloc(ctx_count * kv_dim * sizeof(float));
    v_states = (float*)malloc(ctx_count * kv_dim * sizeof(float));
    attn_states = (float*)malloc(ctx_count * q_dim * sizeof(float));
    norm_weight = (float*)malloc(hidden * sizeof(float));
    post_norm_weight = (float*)malloc(hidden * sizeof(float));
    final_norm_weight = (float*)malloc(hidden * sizeof(float));
    gate = (float*)malloc(inter_dim * sizeof(float));
    up = (float*)malloc(inter_dim * sizeof(float));
    down = (float*)malloc(hidden * sizeof(float));
    temp_out = (float*)malloc(hidden * sizeof(float));
    row_buf = (float*)malloc(max_in_dim * sizeof(float));
    score_buf = (float*)malloc(ctx_count * sizeof(float));
    raw_buf = (unsigned char*)malloc(max_raw_bytes);
    if (!states || !norm_states || !q_states || !k_states || !v_states || !attn_states || !norm_weight || !post_norm_weight || !final_norm_weight || !gate || !up || !down || !temp_out || !row_buf || !score_buf || !raw_buf) {
        goto cleanup;
    }

    if (handle->final_norm_info.ready && handle->final_norm_info.info.ndim >= 1U && handle->final_norm_info.info.shape[0] == hidden) {
        if (!vspec_py_native_forward_read_vector_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &handle->final_norm_info, final_norm_weight, hidden)) {
            goto cleanup;
        }
    } else {
        for (size_t i = 0U; i < hidden; ++i) {
            final_norm_weight[i] = 1.0f;
        }
    }

    for (size_t ti = 0U; ti < ctx_count; ++ti) {
        const int token_id = context_token_ids[ctx_start + ti];
        if (!vspec_py_native_forward_read_row_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &handle->full_embed, token_id, &states[ti * hidden], hidden)) {
            goto cleanup;
        }
    }

    for (size_t li = 0U; li < layer_count; ++li) {
        const VspecPyNativeLayerTensorSet* layer = &handle->full_layers[li];
        const float inv_sqrt_head = 1.0f / sqrtf((float)head_dim);
        if (!layer->ready) {
            goto cleanup;
        }

        if (!vspec_py_native_forward_read_vector_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->input_layernorm, norm_weight, hidden)) {
            goto cleanup;
        }
        if (!vspec_py_native_forward_read_vector_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->post_attention_layernorm, post_norm_weight, hidden)) {
            goto cleanup;
        }

        for (size_t ti = 0U; ti < ctx_count; ++ti) {
            float* state_row = &states[ti * hidden];
            float* norm_row = &norm_states[ti * hidden];
            float* q_row = &q_states[ti * q_dim];
            float* k_row = &k_states[ti * kv_dim];
            float* v_row = &v_states[ti * kv_dim];

            vspec_rmsnorm_f32(state_row, norm_weight, hidden, 1e-5f, norm_row);
            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->q_proj, norm_row, hidden, q_row, q_dim, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }
            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->k_proj, norm_row, hidden, k_row, kv_dim, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }
            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->v_proj, norm_row, hidden, v_row, kv_dim, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }

            for (size_t h = 0U; h < num_heads; ++h) {
                vspec_py_apply_rotary_local(&q_row[h * head_dim], head_dim, ctx_start + ti);
            }
            for (size_t h = 0U; h < num_kv_heads; ++h) {
                vspec_py_apply_rotary_local(&k_row[h * head_dim], head_dim, ctx_start + ti);
            }
        }

        for (size_t ti = 0U; ti < ctx_count; ++ti) {
            float* attn_row = &attn_states[ti * q_dim];
            for (size_t h = 0U; h < num_heads; ++h) {
                const size_t kvh = h % num_kv_heads;
                const float* q_head = &q_states[ti * q_dim + h * head_dim];
                float score_max = -1.0e30f;
                float score_sum = 0.0f;

                for (size_t sj = 0U; sj <= ti; ++sj) {
                    const float* k_head = &k_states[sj * kv_dim + kvh * head_dim];
                    float dot = 0.0f;
                    for (size_t d = 0U; d < head_dim; ++d) {
                        dot += q_head[d] * k_head[d];
                    }
                    score_buf[sj] = dot * inv_sqrt_head;
                    if (score_buf[sj] > score_max) {
                        score_max = score_buf[sj];
                    }
                }

                for (size_t sj = 0U; sj <= ti; ++sj) {
                    score_buf[sj] = expf(score_buf[sj] - score_max);
                    score_sum += score_buf[sj];
                }
                if (score_sum < 1e-12f) {
                    score_sum = 1e-12f;
                }

                for (size_t d = 0U; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t sj = 0U; sj <= ti; ++sj) {
                        const float w = score_buf[sj] / score_sum;
                        const float* v_head = &v_states[sj * kv_dim + kvh * head_dim];
                        acc += w * v_head[d];
                    }
                    attn_row[h * head_dim + d] = acc;
                }
            }

            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->o_proj, attn_row, q_dim, temp_out, hidden, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }
            for (size_t d = 0U; d < hidden; ++d) {
                states[ti * hidden + d] += temp_out[d];
            }
        }

        for (size_t ti = 0U; ti < ctx_count; ++ti) {
            float* state_row = &states[ti * hidden];
            float* norm_row = &norm_states[ti * hidden];
            vspec_rmsnorm_f32(state_row, post_norm_weight, hidden, 1e-5f, norm_row);

            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->gate_proj, norm_row, hidden, gate, inter_dim, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }
            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->up_proj, norm_row, hidden, up, inter_dim, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }
            vspec_silu_inplace(gate, inter_dim);
            vspec_mul_inplace(gate, up, inter_dim);
            if (!vspec_py_native_forward_matvec_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &layer->down_proj, gate, inter_dim, down, hidden, row_buf, raw_buf, max_raw_bytes)) {
                goto cleanup;
            }
            for (size_t d = 0U; d < hidden; ++d) {
                state_row[d] += down[d];
            }
        }
    }

    {
        float* last_norm = &norm_states[(ctx_count - 1U) * hidden];
        const float* last_state = &states[(ctx_count - 1U) * hidden];
        if (handle->final_norm_info.ready && handle->final_norm_info.info.ndim >= 1U && handle->final_norm_info.info.shape[0] == hidden) {
            vspec_rmsnorm_f32(last_state, final_norm_weight, hidden, 1e-5f, last_norm);
        } else {
            (void)memcpy(last_norm, last_state, hidden * sizeof(float));
        }

        for (size_t i = 0U; i < candidate_count; ++i) {
            const int cid = candidate_ids[i];
            float base = base_scores ? base_scores[i] : 0.0f;
            float native_logit = 0.0f;
            if (cid >= 0 && (uint64_t)cid < handle->full_lm_head.info.shape[0]) {
                if (!vspec_py_native_forward_read_row_loc(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX, &handle->full_lm_head, cid, row_buf, hidden)) {
                    goto cleanup;
                }
                for (size_t j = 0U; j < hidden; ++j) {
                    native_logit += last_norm[j] * row_buf[j];
                }
            }
            out_scores[i] = base + (blend * native_logit);
        }
    }

    ok = 1;

cleanup:
    vspec_py_close_cached_files_local(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX);
    free(states);
    free(norm_states);
    free(q_states);
    free(k_states);
    free(v_states);
    free(attn_states);
    free(norm_weight);
    free(post_norm_weight);
    free(final_norm_weight);
    free(gate);
    free(up);
    free(down);
    free(temp_out);
    free(row_buf);
    free(score_buf);
    free(raw_buf);
    return ok;
}

static int vspec_py_native_forward_step_tokens_fallback_impl(
    VspecPyNativeForwardHandle* handle,
    const int* context_token_ids,
    size_t context_token_count,
    const int* candidate_ids,
    const float* base_scores,
    size_t candidate_count,
    float blend,
    float* out_scores
) {
    VspecPyNativeFileCacheEntry file_cache[VSPEC_PY_NATIVE_FILE_CACHE_MAX];
    float* hidden_vec = NULL;
    float* row_vec = NULL;
    size_t hidden = 0U;
    int ok = 0;

    (void)memset(file_cache, 0, sizeof(file_cache));

    if (!handle || !context_token_ids || context_token_count == 0U || !candidate_ids || candidate_count == 0U || !out_scores) {
        VSPEC_NATIVE_DBG("[native][fallback_step] invalid_args\n");
        return 0;
    }
    if (!handle->token_forward_ready && !vspec_py_native_forward_prepare_token_cache(handle)) {
        VSPEC_NATIVE_DBG("[native][fallback_step] prepare_token_cache_failed\n");
        return 0;
    }

    hidden = handle->embed_info.shape[1];
    if (hidden == 0U) {
        VSPEC_NATIVE_DBG("[native][fallback_step] hidden_zero\n");
        return 0;
    }

    hidden_vec = (float*)calloc(hidden, sizeof(float));
    row_vec = (float*)malloc(hidden * sizeof(float));
    if (!hidden_vec || !row_vec) {
        VSPEC_NATIVE_DBG("[native][fallback_step] alloc_failed hidden=%zu\n", hidden);
        goto cleanup;
    }

    for (size_t i = 0U; i < context_token_count; ++i) {
        if (!vspec_py_native_forward_read_row_loc(
                file_cache,
                VSPEC_PY_NATIVE_FILE_CACHE_MAX,
                &handle->full_embed,
                context_token_ids[i],
                row_vec,
                hidden)) {
            VSPEC_NATIVE_DBG("[native][fallback_step] embed_read_failed token_id=%d\n", context_token_ids[i]);
            goto cleanup;
        }
        for (size_t j = 0U; j < hidden; ++j) {
            hidden_vec[j] += row_vec[j];
        }
    }

    {
        const float inv = 1.0f / (float)context_token_count;
        for (size_t j = 0U; j < hidden; ++j) {
            hidden_vec[j] *= inv;
        }
    }

    for (size_t i = 0U; i < candidate_count; ++i) {
        const int cid = candidate_ids[i];
        float base = base_scores ? base_scores[i] : 0.0f;
        float native_logit = 0.0f;

        if (cid >= 0 && (uint64_t)cid < handle->full_lm_head.info.shape[0]) {
            if (!vspec_py_native_forward_read_row_loc(
                    file_cache,
                    VSPEC_PY_NATIVE_FILE_CACHE_MAX,
                    &handle->full_lm_head,
                    cid,
                    row_vec,
                    hidden)) {
                VSPEC_NATIVE_DBG("[native][fallback_step] lm_head_read_failed candidate_id=%d\n", cid);
                goto cleanup;
            }
            for (size_t j = 0U; j < hidden; ++j) {
                native_logit += hidden_vec[j] * row_vec[j];
            }
        }
        out_scores[i] = base + (blend * native_logit);
    }

    ok = 1;

cleanup:
    vspec_py_close_cached_files_local(file_cache, VSPEC_PY_NATIVE_FILE_CACHE_MAX);
    free(hidden_vec);
    free(row_vec);
    return ok;
}

static int vspec_py_native_forward_step_tokens_impl(
    VspecPyNativeForwardHandle* handle,
    const int* context_token_ids,
    size_t context_token_count,
    const int* candidate_ids,
    const float* base_scores,
    size_t candidate_count,
    float blend,
    float* out_scores
) {
    if (vspec_py_env_true_local("VSPEC_NATIVE_FULL_TRANSFORMER")) {
        if (vspec_py_native_forward_step_tokens_full_impl(
                handle,
                context_token_ids,
                context_token_count,
                candidate_ids,
                base_scores,
                candidate_count,
                blend,
                out_scores)) {
            return 1;
        }
    }
    return vspec_py_native_forward_step_tokens_fallback_impl(
        handle,
        context_token_ids,
        context_token_count,
        candidate_ids,
        base_scores,
        candidate_count,
        blend,
        out_scores);
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
        (void)vspec_py_native_forward_prepare_token_cache(handle);
        (void)vspec_py_native_forward_prepare_full_cache(handle);
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
    vspec_py_native_forward_release_full_cache(handle);
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

int vspec_py_native_forward_step_tokens(
    int handle_id,
    const int* context_token_ids,
    size_t context_token_count,
    const int* candidate_ids,
    const float* base_scores,
    size_t candidate_count,
    float blend,
    float* out_scores
) {
    VspecPyNativeForwardHandle* handle = vspec_py_get_native_forward_handle(handle_id);
    if (!handle || !context_token_ids || context_token_count == 0U || !candidate_ids || candidate_count == 0U || !out_scores) {
        return 0;
    }

    if (blend < 0.0f) {
        blend = 0.0f;
    }
    if (blend > 1.0f) {
        blend = 1.0f;
    }

    if (!vspec_py_native_forward_step_tokens_impl(
            handle,
            context_token_ids,
            context_token_count,
            candidate_ids,
            base_scores,
            candidate_count,
            blend,
            out_scores)) {
        return 0;
    }

    return 1;
}

int vspec_py_native_forward_prefill_tokens(
    int handle_id,
    const int* context_token_ids,
    size_t context_token_count,
    size_t* out_processed_tokens
) {
    VspecPyNativeForwardHandle* handle = vspec_py_get_native_forward_handle(handle_id);
    int dummy_candidate = 0;
    float dummy_score = 0.0f;

    if (out_processed_tokens) {
        *out_processed_tokens = 0U;
    }
    if (!handle || !context_token_ids || context_token_count == 0U) {
        return 0;
    }

    if (!vspec_py_native_forward_step_tokens_impl(
            handle,
            context_token_ids,
            context_token_count,
            &dummy_candidate,
            NULL,
            1U,
            0.0f,
            &dummy_score)) {
        return 0;
    }

    if (out_processed_tokens) {
        *out_processed_tokens = context_token_count;
    }
    return 1;
}

int vspec_py_native_forward_topk_tokens(
    int handle_id,
    const int* context_token_ids,
    size_t context_token_count,
    size_t top_k,
    int* out_token_ids,
    float* out_scores,
    size_t out_capacity,
    size_t* out_count
) {
    VspecPyNativeForwardHandle* handle = vspec_py_get_native_forward_handle(handle_id);
    size_t vocab = 0U;
    int* chunk_ids = NULL;
    float* chunk_scores = NULL;
    size_t chunk_cap = 0U;
    size_t k = 0U;
    int ok = 0;

    if (out_count) {
        *out_count = 0U;
    }
    if (!handle || !context_token_ids || context_token_count == 0U || !out_token_ids || !out_scores || out_capacity == 0U) {
        return 0;
    }
    if (!handle->token_forward_ready && !vspec_py_native_forward_prepare_token_cache(handle)) {
        return 0;
    }

    vocab = handle->full_lm_head.info.shape[0];
    if (vocab == 0U) {
        return 0;
    }

    k = top_k;
    if (k == 0U) {
        k = 1U;
    }
    if (k > out_capacity) {
        k = out_capacity;
    }
    if (k > vocab) {
        k = vocab;
    }
    if (k == 0U) {
        return 0;
    }

    chunk_cap = vspec_py_env_size_local("VSPEC_NATIVE_LOGITS_PROVIDER_CHUNK", 2048U, 16384U);
    if (chunk_cap == 0U) {
        chunk_cap = 2048U;
    }
    if (chunk_cap > vocab) {
        chunk_cap = vocab;
    }

    chunk_ids = (int*)malloc(chunk_cap * sizeof(int));
    chunk_scores = (float*)malloc(chunk_cap * sizeof(float));
    if (!chunk_ids || !chunk_scores) {
        goto cleanup;
    }

    for (size_t i = 0U; i < k; ++i) {
        out_token_ids[i] = -1;
        out_scores[i] = -1.0e30f;
    }

    for (size_t base = 0U; base < vocab; base += chunk_cap) {
        size_t n = chunk_cap;
        if (base + n > vocab) {
            n = vocab - base;
        }
        for (size_t i = 0U; i < n; ++i) {
            chunk_ids[i] = (int)(base + i);
            chunk_scores[i] = 0.0f;
        }

        if (!vspec_py_native_forward_step_tokens_impl(
                handle,
                context_token_ids,
                context_token_count,
                chunk_ids,
                NULL,
                n,
                1.0f,
                chunk_scores)) {
            goto cleanup;
        }

        for (size_t i = 0U; i < n; ++i) {
            const int tid = chunk_ids[i];
            const float s = chunk_scores[i];
            for (size_t p = 0U; p < k; ++p) {
                if (s > out_scores[p]) {
                    for (size_t r = k - 1U; r > p; --r) {
                        out_scores[r] = out_scores[r - 1U];
                        out_token_ids[r] = out_token_ids[r - 1U];
                    }
                    out_scores[p] = s;
                    out_token_ids[p] = tid;
                    break;
                }
            }
        }
    }

    if (out_count) {
        *out_count = k;
    }
    ok = 1;

cleanup:
    free(chunk_ids);
    free(chunk_scores);
    return ok;
}

int vspec_py_native_forward_sample_topk_tokens(
    int handle_id,
    const int* context_token_ids,
    size_t context_token_count,
    size_t top_k,
    float temperature,
    int greedy,
    uint64_t random_bits,
    const int* repetition_token_ids,
    size_t repetition_token_count,
    float repetition_penalty,
    int* out_token_id,
    float* out_token_score
) {
    int* token_ids = NULL;
    float* scores = NULL;
    size_t out_count = 0U;
    size_t k = 0U;
    int sampled = -1;
    int ok = 0;

    if (!out_token_id) {
        return 0;
    }
    *out_token_id = -1;
    if (out_token_score) {
        *out_token_score = -1.0e30f;
    }

    if (!context_token_ids || context_token_count == 0U) {
        return 0;
    }

    k = top_k;
    if (k == 0U) {
        k = 1U;
    }
    if (k > 1024U) {
        k = 1024U;
    }

    token_ids = (int*)malloc(k * sizeof(int));
    scores = (float*)malloc(k * sizeof(float));
    if (!token_ids || !scores) {
        goto cleanup;
    }

    if (!vspec_py_native_forward_topk_tokens(
            handle_id,
            context_token_ids,
            context_token_count,
            k,
            token_ids,
            scores,
            k,
            &out_count)) {
        goto cleanup;
    }
    if (out_count == 0U) {
        goto cleanup;
    }

    if (temperature < 0.05f) {
        temperature = 0.05f;
    }
    if (temperature > 8.0f) {
        temperature = 8.0f;
    }

    if (repetition_penalty < 1.0f) {
        repetition_penalty = 1.0f;
    }
    if (repetition_penalty > 3.0f) {
        repetition_penalty = 3.0f;
    }

    if (repetition_token_ids && repetition_token_count > 0U && repetition_penalty > 1.0f) {
        for (size_t i = 0U; i < out_count; ++i) {
            const int tid = token_ids[i];
            int repeated = 0;
            for (size_t j = 0U; j < repetition_token_count; ++j) {
                if (repetition_token_ids[j] == tid) {
                    repeated = 1;
                    break;
                }
            }
            if (!repeated) {
                continue;
            }
            if (scores[i] > 0.0f) {
                scores[i] /= repetition_penalty;
            } else {
                scores[i] *= repetition_penalty;
            }
        }
    }

    for (size_t i = 0U; i < out_count; ++i) {
        scores[i] /= temperature;
    }

    if (!vspec_sampling_select_candidate(
            token_ids,
            scores,
            out_count,
            (greedy != 0) ? 1 : 0,
            random_bits,
            &sampled)) {
        goto cleanup;
    }

    *out_token_id = sampled;
    if (out_token_score) {
        float matched = -1.0e30f;
        for (size_t i = 0U; i < out_count; ++i) {
            if (token_ids[i] == sampled) {
                matched = scores[i];
                break;
            }
        }
        *out_token_score = matched;
    }
    ok = 1;

cleanup:
    free(token_ids);
    free(scores);
    return ok;
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
