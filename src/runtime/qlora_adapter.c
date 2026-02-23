#include "vspec/runtime/qlora_adapter.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VSPEC_QLORA_MAX_LAYERS 64U

typedef struct VspecQloraLayer {
    int active;
    VspecQloraLayerConfig cfg;
    float* matrix_a;
    float* matrix_b;
} VspecQloraLayer;

static VspecQloraLayer g_layers[VSPEC_QLORA_MAX_LAYERS];

static char* vspec_read_text_file(const char* path, size_t* out_size) {
    if (!path || path[0] == '\0') {
        return NULL;
    }
    FILE* file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long file_size = ftell(file);
    if (file_size <= 0) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char* buffer = (char*)malloc((size_t)file_size + 1U);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    size_t read_size = fread(buffer, 1, (size_t)file_size, file);
    fclose(file);
    if (read_size != (size_t)file_size) {
        free(buffer);
        return NULL;
    }
    buffer[read_size] = '\0';
    if (out_size) {
        *out_size = read_size;
    }
    return buffer;
}

static int vspec_load_float_blob(const char* path, size_t count, float* out_data) {
    if (!path || path[0] == '\0' || !out_data || count == 0U) {
        return 0;
    }
    FILE* file = fopen(path, "rb");
    if (!file) {
        return 0;
    }
    const size_t read_count = fread(out_data, sizeof(float), count, file);
    fclose(file);
    return read_count == count ? 1 : 0;
}

static int vspec_path_is_absolute(const char* path) {
    if (!path || path[0] == '\0') {
        return 0;
    }

    if (path[0] == '/' || path[0] == '\\') {
        return 1;
    }

    if (isalpha((unsigned char)path[0]) && path[1] == ':') {
        return 1;
    }

    return 0;
}

static int vspec_extract_parent_dir(const char* path, char* out_dir, size_t capacity) {
    if (!out_dir || capacity == 0U) {
        return 0;
    }
    out_dir[0] = '\0';
    if (!path || path[0] == '\0') {
        return 1;
    }

    const char* slash = strrchr(path, '/');
    const char* bslash = strrchr(path, '\\');
    const char* cut = slash;
    if (!cut || (bslash && bslash > cut)) {
        cut = bslash;
    }
    if (!cut) {
        return 1;
    }

    const size_t len = (size_t)(cut - path);
    if (len + 1U > capacity) {
        return 0;
    }
    (void)memcpy(out_dir, path, len);
    out_dir[len] = '\0';
    return 1;
}

static int vspec_join_manifest_relative_path(
    const char* manifest_dir,
    const char* path_text,
    char* out_path,
    size_t out_capacity
) {
    if (!path_text || !out_path || out_capacity == 0U) {
        return 0;
    }

    if (vspec_path_is_absolute(path_text) || !manifest_dir || manifest_dir[0] == '\0') {
        const size_t path_len = strlen(path_text);
        if (path_len + 1U > out_capacity) {
            return 0;
        }
        (void)memcpy(out_path, path_text, path_len + 1U);
        return 1;
    }

    const size_t base_len = strlen(manifest_dir);
    const size_t rel_len = strlen(path_text);
    const int has_trailing_sep =
        (base_len > 0U && (manifest_dir[base_len - 1U] == '/' || manifest_dir[base_len - 1U] == '\\')) ? 1 : 0;
    const size_t need_len = base_len + (has_trailing_sep ? 0U : 1U) + rel_len;
    if (need_len + 1U > out_capacity) {
        return 0;
    }

    (void)memcpy(out_path, manifest_dir, base_len);
    size_t offset = base_len;
    if (!has_trailing_sep) {
#if defined(_WIN32)
        out_path[offset++] = '\\';
#else
        out_path[offset++] = '/';
#endif
    }
    (void)memcpy(out_path + offset, path_text, rel_len);
    out_path[offset + rel_len] = '\0';
    return 1;
}

static const char* vspec_find_key_in_range(const char* begin, const char* end, const char* key) {
    if (!begin || !end || !key || begin >= end) {
        return NULL;
    }

    char quoted[64];
    (void)snprintf(quoted, sizeof(quoted), "\"%s\"", key);
    const size_t key_len = strlen(quoted);
    const char* cursor = begin;
    while (cursor + key_len < end) {
        const char* hit = strstr(cursor, quoted);
        if (!hit || hit >= end) {
            return NULL;
        }
        const char* colon = hit + key_len;
        while (colon < end && isspace((unsigned char)*colon)) {
            ++colon;
        }
        if (colon < end && *colon == ':') {
            return colon + 1;
        }
        cursor = hit + 1;
    }
    return NULL;
}

static int vspec_parse_uint_key(const char* begin, const char* end, const char* key, unsigned int* out_value) {
    const char* pos = vspec_find_key_in_range(begin, end, key);
    if (!pos || !out_value) {
        return 0;
    }
    while (pos < end && isspace((unsigned char)*pos)) {
        ++pos;
    }
    char* parse_end = NULL;
    unsigned long v = strtoul(pos, &parse_end, 10);
    if (!parse_end || parse_end == pos) {
        return 0;
    }
    *out_value = (unsigned int)v;
    return 1;
}

static int vspec_parse_size_key(const char* begin, const char* end, const char* key, size_t* out_value) {
    unsigned int tmp = 0U;
    if (!vspec_parse_uint_key(begin, end, key, &tmp)) {
        return 0;
    }
    if (!out_value) {
        return 0;
    }
    *out_value = (size_t)tmp;
    return 1;
}

static int vspec_parse_float_key(const char* begin, const char* end, const char* key, float* out_value) {
    const char* pos = vspec_find_key_in_range(begin, end, key);
    if (!pos || !out_value) {
        return 0;
    }
    while (pos < end && isspace((unsigned char)*pos)) {
        ++pos;
    }
    char* parse_end = NULL;
    float v = strtof(pos, &parse_end);
    if (!parse_end || parse_end == pos) {
        return 0;
    }
    *out_value = v;
    return 1;
}

static int vspec_parse_string_key(
    const char* begin,
    const char* end,
    const char* key,
    char* out_text,
    size_t out_capacity
) {
    const char* pos = vspec_find_key_in_range(begin, end, key);
    if (!pos || !out_text || out_capacity == 0U) {
        return 0;
    }
    while (pos < end && isspace((unsigned char)*pos)) {
        ++pos;
    }
    if (pos >= end || *pos != '"') {
        return 0;
    }
    ++pos;
    const char* close = pos;
    while (close < end && *close != '"') {
        ++close;
    }
    if (close >= end) {
        return 0;
    }
    const size_t len = (size_t)(close - pos);
    if (len + 1U > out_capacity) {
        return 0;
    }
    (void)memcpy(out_text, pos, len);
    out_text[len] = '\0';
    return 1;
}

static void vspec_qlora_free_layer(VspecQloraLayer* layer) {
    if (!layer) {
        return;
    }
    if (layer->matrix_a) {
        free(layer->matrix_a);
        layer->matrix_a = NULL;
    }
    if (layer->matrix_b) {
        free(layer->matrix_b);
        layer->matrix_b = NULL;
    }
    layer->active = 0;
    (void)memset(&layer->cfg, 0, sizeof(layer->cfg));
}

static VspecQloraLayer* vspec_qlora_find_layer(uint32_t layer_id) {
    for (size_t i = 0; i < VSPEC_QLORA_MAX_LAYERS; ++i) {
        if (g_layers[i].active && g_layers[i].cfg.layer_id == layer_id) {
            return &g_layers[i];
        }
    }
    return NULL;
}

static VspecQloraLayer* vspec_qlora_acquire_slot(uint32_t layer_id) {
    VspecQloraLayer* exists = vspec_qlora_find_layer(layer_id);
    if (exists) {
        return exists;
    }
    for (size_t i = 0; i < VSPEC_QLORA_MAX_LAYERS; ++i) {
        if (!g_layers[i].active) {
            return &g_layers[i];
        }
    }
    return NULL;
}

int vspec_qlora_adapter_add_layer(
    uint32_t layer_id,
    size_t in_dim,
    size_t rank,
    size_t out_dim,
    float alpha,
    const float* matrix_a,
    const float* matrix_b
) {
    if (layer_id == 0U || in_dim == 0U || rank == 0U || out_dim == 0U || !matrix_a || !matrix_b) {
        return 0;
    }

    VspecQloraLayer* slot = vspec_qlora_acquire_slot(layer_id);
    if (!slot) {
        return 0;
    }

    const size_t a_count = in_dim * rank;
    const size_t b_count = rank * out_dim;

    float* a_copy = (float*)malloc(a_count * sizeof(float));
    float* b_copy = (float*)malloc(b_count * sizeof(float));
    if (!a_copy || !b_copy) {
        if (a_copy) free(a_copy);
        if (b_copy) free(b_copy);
        return 0;
    }

    (void)memcpy(a_copy, matrix_a, a_count * sizeof(float));
    (void)memcpy(b_copy, matrix_b, b_count * sizeof(float));

    vspec_qlora_free_layer(slot);
    slot->active = 1;
    slot->cfg.layer_id = layer_id;
    slot->cfg.in_dim = in_dim;
    slot->cfg.rank = rank;
    slot->cfg.out_dim = out_dim;
    slot->cfg.alpha = alpha;
    slot->matrix_a = a_copy;
    slot->matrix_b = b_copy;
    return 1;
}

int vspec_qlora_adapter_load_file(const char* path) {
    if (!path || path[0] == '\0') {
        return 0;
    }

    FILE* file = fopen(path, "rb");
    if (!file) {
        return 0;
    }

    char token[64] = {0};
    int loaded = 0;

    while (fscanf(file, "%63s", token) == 1) {
        if (strcmp(token, "VSPEC_QLORA_V1") == 0) {
            continue;
        }
        if (strcmp(token, "LAYER") != 0) {
            continue;
        }

        unsigned int layer_id = 0U;
        size_t in_dim = 0U;
        size_t rank = 0U;
        size_t out_dim = 0U;
        float alpha = 1.0f;

        if (fscanf(file, "%u %zu %zu %zu %f", &layer_id, &in_dim, &rank, &out_dim, &alpha) != 5) {
            break;
        }

        const size_t a_count = in_dim * rank;
        const size_t b_count = rank * out_dim;
        float* matrix_a = (float*)malloc(a_count * sizeof(float));
        float* matrix_b = (float*)malloc(b_count * sizeof(float));
        if (!matrix_a || !matrix_b) {
            if (matrix_a) free(matrix_a);
            if (matrix_b) free(matrix_b);
            break;
        }

        int ok = 1;
        for (size_t i = 0; i < a_count; ++i) {
            if (fscanf(file, "%f", &matrix_a[i]) != 1) {
                ok = 0;
                break;
            }
        }
        for (size_t i = 0; ok && i < b_count; ++i) {
            if (fscanf(file, "%f", &matrix_b[i]) != 1) {
                ok = 0;
                break;
            }
        }

        if (ok) {
            loaded += vspec_qlora_adapter_add_layer(layer_id, in_dim, rank, out_dim, alpha, matrix_a, matrix_b);
        }

        free(matrix_a);
        free(matrix_b);

        if (!ok) {
            break;
        }
    }

    fclose(file);
    return loaded;
}

int vspec_qlora_adapter_load_manifest_json(const char* manifest_path) {
    char manifest_dir[1024] = {0};
    if (!vspec_extract_parent_dir(manifest_path, manifest_dir, sizeof(manifest_dir))) {
        return 0;
    }

    size_t text_size = 0U;
    char* text = vspec_read_text_file(manifest_path, &text_size);
    if (!text || text_size == 0U) {
        if (text) {
            free(text);
        }
        return 0;
    }

    int loaded = 0;
    const char* cursor = text;
    const char* end = text + text_size;

    while (cursor < end) {
        const char* marker = strstr(cursor, "\"layer_id\"");
        if (!marker || marker >= end) {
            break;
        }

        const char* obj_begin = marker;
        while (obj_begin > text && *obj_begin != '{') {
            --obj_begin;
        }
        if (*obj_begin != '{') {
            cursor = marker + 1;
            continue;
        }

        const char* obj_end = marker;
        while (obj_end < end && *obj_end != '}') {
            ++obj_end;
        }
        if (obj_end >= end || *obj_end != '}') {
            break;
        }

        unsigned int layer_id = 0U;
        size_t in_dim = 0U;
        size_t rank = 0U;
        size_t out_dim = 0U;
        float alpha = 1.0f;
        char a_path_text[512] = {0};
        char b_path_text[512] = {0};
        char a_path[1024] = {0};
        char b_path[1024] = {0};

        int ok = 1;
        ok = ok && vspec_parse_uint_key(obj_begin, obj_end, "layer_id", &layer_id);
        ok = ok && vspec_parse_size_key(obj_begin, obj_end, "in_dim", &in_dim);
        ok = ok && vspec_parse_size_key(obj_begin, obj_end, "rank", &rank);
        ok = ok && vspec_parse_size_key(obj_begin, obj_end, "out_dim", &out_dim);
        ok = ok && vspec_parse_float_key(obj_begin, obj_end, "alpha", &alpha);
        ok = ok && vspec_parse_string_key(obj_begin, obj_end, "a_path", a_path_text, sizeof(a_path_text));
        ok = ok && vspec_parse_string_key(obj_begin, obj_end, "b_path", b_path_text, sizeof(b_path_text));
        ok = ok && vspec_join_manifest_relative_path(manifest_dir, a_path_text, a_path, sizeof(a_path));
        ok = ok && vspec_join_manifest_relative_path(manifest_dir, b_path_text, b_path, sizeof(b_path));

        if (ok) {
            const size_t a_count = in_dim * rank;
            const size_t b_count = rank * out_dim;
            float* matrix_a = (float*)malloc(a_count * sizeof(float));
            float* matrix_b = (float*)malloc(b_count * sizeof(float));
            if (matrix_a && matrix_b &&
                vspec_load_float_blob(a_path, a_count, matrix_a) &&
                vspec_load_float_blob(b_path, b_count, matrix_b)) {
                loaded += vspec_qlora_adapter_add_layer(
                    layer_id,
                    in_dim,
                    rank,
                    out_dim,
                    alpha,
                    matrix_a,
                    matrix_b
                );
            }
            if (matrix_a) {
                free(matrix_a);
            }
            if (matrix_b) {
                free(matrix_b);
            }
        }

        cursor = obj_end + 1;
    }

    free(text);
    return loaded;
}

void vspec_qlora_adapter_clear(void) {
    for (size_t i = 0; i < VSPEC_QLORA_MAX_LAYERS; ++i) {
        if (g_layers[i].active || g_layers[i].matrix_a || g_layers[i].matrix_b) {
            vspec_qlora_free_layer(&g_layers[i]);
        }
    }
}

int vspec_qlora_adapter_has_layer(uint32_t layer_id) {
    return vspec_qlora_find_layer(layer_id) ? 1 : 0;
}

void vspec_qlora_adapter_apply_layer_f32(
    uint32_t layer_id,
    const float* input,
    size_t m,
    size_t k,
    size_t n,
    float* output
) {
    if (!input || !output || m == 0U || k == 0U || n == 0U) {
        return;
    }

    VspecQloraLayer* layer = vspec_qlora_find_layer(layer_id);
    if (!layer) {
        return;
    }

    if (layer->cfg.in_dim != k || layer->cfg.out_dim != n || layer->cfg.rank == 0U || !layer->matrix_a || !layer->matrix_b) {
        return;
    }

    const size_t rank = layer->cfg.rank;
    const float scale = layer->cfg.alpha / (float)rank;
    float* tmp = (float*)malloc(rank * sizeof(float));
    if (!tmp) {
        return;
    }

    for (size_t row = 0; row < m; ++row) {
        const float* in_row = input + (row * k);
        float* out_row = output + (row * n);

        for (size_t r = 0; r < rank; ++r) {
            float acc = 0.0f;
            for (size_t t = 0; t < k; ++t) {
                acc += in_row[t] * layer->matrix_a[t * rank + r];
            }
            tmp[r] = acc;
        }

        for (size_t col = 0; col < n; ++col) {
            float delta = 0.0f;
            for (size_t r = 0; r < rank; ++r) {
                delta += tmp[r] * layer->matrix_b[r * n + col];
            }
            out_row[col] += delta * scale;
        }
    }

    free(tmp);
}