#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "vspec/compat/safetensors_parser.h"
#include "vspec/compat/weight_mapper.h"

typedef struct TensorRef {
    const VspecCompatTensorInfo* info;
    uint64_t file_offset;
} TensorRef;

typedef struct TensorLocation {
    VspecCompatTensorInfo info;
    char file_path[1024];
    uint64_t data_base;
} TensorLocation;

static uint64_t read_u64_le_local(const unsigned char b[8]) {
    uint64_t v = 0U;
    for (size_t i = 0U; i < 8U; ++i) {
        v |= ((uint64_t)b[i]) << (8U * i);
    }
    return v;
}

static uint16_t read_u16_le_local(const unsigned char b[2]) {
    return (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8U));
}

static float f16_to_f32(uint16_t h) {
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

static float bf16_to_f32(uint16_t b16) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)b16) << 16U;
    return v.f;
}

static size_t dtype_bytes(const char* dtype) {
    if (!dtype) return 0U;
    if (strcmp(dtype, "F16") == 0) return 2U;
    if (strcmp(dtype, "BF16") == 0) return 2U;
    if (strcmp(dtype, "F32") == 0) return 4U;
    return 0U;
}

static float decode_scalar(const unsigned char* bytes, const char* dtype) {
    if (!bytes || !dtype) {
        return 0.0f;
    }
    if (strcmp(dtype, "F16") == 0) {
        return f16_to_f32(read_u16_le_local(bytes));
    }
    if (strcmp(dtype, "BF16") == 0) {
        return bf16_to_f32(read_u16_le_local(bytes));
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

static int find_canonical_tensors(const VspecCompatModel* model, TensorRef* embed, TensorRef* lm_head) {
    if (!model || !embed || !lm_head) {
        return 0;
    }
    embed->info = NULL;
    lm_head->info = NULL;
    embed->file_offset = 0U;
    lm_head->file_offset = 0U;

    for (size_t i = 0U; i < model->tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        if (!vspec_weight_canonical_name(model->tensors[i].name, canonical, sizeof(canonical))) {
            continue;
        }
        if (!embed->info && strcmp(canonical, "model.embed_tokens.weight") == 0) {
            embed->info = &model->tensors[i];
            continue;
        }
        if (!lm_head->info && strcmp(canonical, "lm_head.weight") == 0) {
            lm_head->info = &model->tensors[i];
            continue;
        }
    }
    return (embed->info != NULL && lm_head->info != NULL) ? 1 : 0;
}

static int is_directory_path(const char* path) {
    struct _stat st;
    if (!path || !path[0]) {
        return 0;
    }
    if (_stat(path, &st) != 0) {
        return 0;
    }
    return (st.st_mode & _S_IFDIR) != 0;
}

static int has_suffix_ci(const char* s, const char* suffix) {
    size_t n = 0U;
    size_t m = 0U;
    if (!s || !suffix) {
        return 0;
    }
    n = strlen(s);
    m = strlen(suffix);
    if (n < m) {
        return 0;
    }
#ifdef _WIN32
    return _stricmp(s + (n - m), suffix) == 0;
#else
    return strcmp(s + (n - m), suffix) == 0;
#endif
}

static int read_data_base(const char* file_path, uint64_t* out_data_base) {
    FILE* f = NULL;
    unsigned char hdr[8];
    uint64_t header_len = 0U;
    if (!file_path || !out_data_base) {
        return 0;
    }
    f = fopen(file_path, "rb");
    if (!f) {
        return 0;
    }
    if (fread(hdr, 1, 8, f) != 8U) {
        fclose(f);
        return 0;
    }
    header_len = read_u64_le_local(hdr);
    fclose(f);
    *out_data_base = 8ULL + header_len;
    return 1;
}

static int find_canonical_in_file(const char* file_path, const char* canonical_name, TensorLocation* out_loc) {
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
            if (!read_data_base(file_path, &data_base)) {
                return 0;
            }
            out_loc->info = model.tensors[i];
            out_loc->data_base = data_base;
            if (snprintf(out_loc->file_path, sizeof(out_loc->file_path), "%s", file_path) >= (int)sizeof(out_loc->file_path)) {
                return 0;
            }
            return 1;
        }
    }
    return 0;
}

static int resolve_tensor_locations(const char* model_path, TensorLocation* embed_loc, TensorLocation* lm_loc, int* out_tied_lm_head) {
    if (!model_path || !embed_loc || !lm_loc) {
        return 0;
    }
    if (out_tied_lm_head) {
        *out_tied_lm_head = 0;
    }

    memset(embed_loc, 0, sizeof(*embed_loc));
    memset(lm_loc, 0, sizeof(*lm_loc));

    if (!is_directory_path(model_path)) {
        const int ok_embed = find_canonical_in_file(model_path, "model.embed_tokens.weight", embed_loc);
        const int ok_lm = find_canonical_in_file(model_path, "lm_head.weight", lm_loc);
        if (ok_embed && ok_lm) {
            return 1;
        }
        if (ok_embed && !ok_lm) {
            *lm_loc = *embed_loc;
            if (out_tied_lm_head) {
                *out_tied_lm_head = 1;
            }
            return 1;
        }
        return 0;
    }

#ifdef _WIN32
    {
        char pattern[1200];
        WIN32_FIND_DATAA data;
        HANDLE h = INVALID_HANDLE_VALUE;
        if (snprintf(pattern, sizeof(pattern), "%s\\*.safetensors", model_path) >= (int)sizeof(pattern)) {
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
            if (!has_suffix_ci(data.cFileName, ".safetensors")) {
                continue;
            }
            if (snprintf(file_path, sizeof(file_path), "%s\\%s", model_path, data.cFileName) >= (int)sizeof(file_path)) {
                continue;
            }
            if (embed_loc->file_path[0] == '\0') {
                (void)find_canonical_in_file(file_path, "model.embed_tokens.weight", embed_loc);
            }
            if (lm_loc->file_path[0] == '\0') {
                (void)find_canonical_in_file(file_path, "lm_head.weight", lm_loc);
            }
            if (embed_loc->file_path[0] != '\0' && lm_loc->file_path[0] != '\0') {
                FindClose(h);
                return 1;
            }
        } while (FindNextFileA(h, &data));
        FindClose(h);
    }
#endif

    if (embed_loc->file_path[0] != '\0' && lm_loc->file_path[0] == '\0') {
        *lm_loc = *embed_loc;
        if (out_tied_lm_head) {
            *out_tied_lm_head = 1;
        }
        return 1;
    }
    return 0;
}

static int parse_token_ids(const char* s, int* out_ids, size_t cap, size_t* out_n) {
    char buf[4096];
    char* tok = NULL;
    size_t n = 0U;
    if (!s || !out_ids || cap == 0U || !out_n) {
        return 0;
    }
    if (snprintf(buf, sizeof(buf), "%s", s) >= (int)sizeof(buf)) {
        return 0;
    }
    tok = strtok(buf, ", ");
    while (tok && n < cap) {
        char* endp = NULL;
        long v = strtol(tok, &endp, 10);
        if (endp == tok || v < 0 || v > 10000000L) {
            return 0;
        }
        out_ids[n++] = (int)v;
        tok = strtok(NULL, ", ");
    }
    if (n == 0U) {
        return 0;
    }
    *out_n = n;
    return 1;
}

static int read_embedding_avg(
    FILE* f,
    const TensorRef* embed,
    const int* token_ids,
    size_t token_count,
    float* out_hidden,
    size_t hidden
) {
    const size_t elem_bytes = dtype_bytes(embed->info->dtype);
    const size_t vocab = embed->info->shape[0];
    unsigned char* row = NULL;
    if (!f || !embed || !embed->info || !token_ids || token_count == 0U || !out_hidden || hidden == 0U || elem_bytes == 0U) {
        return 0;
    }

    row = (unsigned char*)malloc(hidden * elem_bytes);
    if (!row) {
        return 0;
    }
    for (size_t j = 0U; j < hidden; ++j) {
        out_hidden[j] = 0.0f;
    }

    for (size_t t = 0U; t < token_count; ++t) {
        const int tid = token_ids[t];
        if (tid < 0 || (size_t)tid >= vocab) {
            free(row);
            return 0;
        }
        {
            const uint64_t row_off = embed->file_offset + ((uint64_t)(size_t)tid * (uint64_t)hidden * (uint64_t)elem_bytes);
            if (fseek(f, (long)row_off, SEEK_SET) != 0) {
                free(row);
                return 0;
            }
            if (fread(row, 1, hidden * elem_bytes, f) != hidden * elem_bytes) {
                free(row);
                return 0;
            }
        }
        for (size_t j = 0U; j < hidden; ++j) {
            out_hidden[j] += decode_scalar(&row[j * elem_bytes], embed->info->dtype);
        }
    }

    {
        const float inv = 1.0f / (float)token_count;
        for (size_t j = 0U; j < hidden; ++j) {
            out_hidden[j] *= inv;
        }
    }

    free(row);
    return 1;
}

static int topk_insert(int* ids, float* scores, size_t k, int id, float score) {
    if (!ids || !scores || k == 0U) return 0;
    for (size_t i = 0U; i < k; ++i) {
        if (score > scores[i]) {
            for (size_t j = k - 1U; j > i; --j) {
                ids[j] = ids[j - 1U];
                scores[j] = scores[j - 1U];
            }
            ids[i] = id;
            scores[i] = score;
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* token_ids_arg = NULL;
    int token_ids[1024];
    size_t token_count = 0U;
    size_t top_k = 8U;
    TensorLocation embed_loc;
    TensorLocation lm_loc;
    TensorRef embed;
    TensorRef lm_head;
    FILE* embed_f = NULL;
    FILE* lm_f = NULL;
    size_t hidden = 0U;
    size_t vocab = 0U;
    size_t elem_bytes = 0U;
    float* hidden_vec = NULL;
    unsigned char* lm_row = NULL;
    int* top_ids = NULL;
    float* top_scores = NULL;
    int tied_lm_head = 0;

    if (argc < 3) {
        fprintf(stderr, "usage: native_real_logits <safetensors-file|model-dir> <token_ids_csv> [top_k]\n");
        fprintf(stderr, "example: native_real_logits model.safetensors \"1,2,3,4\" 8\n");
        return 2;
    }

    model_path = argv[1];
    token_ids_arg = argv[2];
    if (argc >= 4) {
        long v = strtol(argv[3], NULL, 10);
        if (v > 0 && v <= 128) {
            top_k = (size_t)v;
        }
    }

    if (!parse_token_ids(token_ids_arg, token_ids, sizeof(token_ids) / sizeof(token_ids[0]), &token_count)) {
        fprintf(stderr, "[native-real] invalid token_ids_csv\n");
        return 3;
    }

    if (!resolve_tensor_locations(model_path, &embed_loc, &lm_loc, &tied_lm_head)) {
        fprintf(stderr, "[native-real] missing canonical tensors embed/lm_head\n");
        return 4;
    }

    embed.info = &embed_loc.info;
    lm_head.info = &lm_loc.info;
    if (embed.info->ndim < 2U || lm_head.info->ndim < 2U) {
        fprintf(stderr, "[native-real] unsupported tensor rank\n");
        return 6;
    }

    hidden = embed.info->shape[1];
    vocab = embed.info->shape[0];
    if (hidden == 0U || vocab == 0U) {
        fprintf(stderr, "[native-real] invalid embed shape\n");
        return 7;
    }
    if (lm_head.info->shape[0] != vocab || lm_head.info->shape[1] != hidden) {
        fprintf(stderr, "[native-real] lm_head shape not row-major [vocab,hidden], got [%zu,%zu]\n",
            lm_head.info->shape[0], lm_head.info->shape[1]);
        return 8;
    }

    elem_bytes = dtype_bytes(lm_head.info->dtype);
    if (elem_bytes == 0U || dtype_bytes(embed.info->dtype) == 0U) {
        fprintf(stderr, "[native-real] unsupported dtype (embed=%s lm_head=%s)\n", embed.info->dtype, lm_head.info->dtype);
        return 9;
    }

    embed_f = fopen(embed_loc.file_path, "rb");
    lm_f = fopen(lm_loc.file_path, "rb");
    if (!embed_f || !lm_f) {
        fprintf(stderr, "[native-real] open_failed\n");
        if (embed_f) fclose(embed_f);
        if (lm_f) fclose(lm_f);
        return 10;
    }

    embed.file_offset = embed_loc.data_base + embed.info->data_offset_start;
    lm_head.file_offset = lm_loc.data_base + lm_head.info->data_offset_start;

    hidden_vec = (float*)malloc(hidden * sizeof(float));
    lm_row = (unsigned char*)malloc(hidden * elem_bytes);
    top_ids = (int*)malloc(top_k * sizeof(int));
    top_scores = (float*)malloc(top_k * sizeof(float));
    if (!hidden_vec || !lm_row || !top_ids || !top_scores) {
        fprintf(stderr, "[native-real] alloc_failed\n");
        fclose(embed_f);
        fclose(lm_f);
        free(hidden_vec);
        free(lm_row);
        free(top_ids);
        free(top_scores);
        return 12;
    }

    for (size_t i = 0U; i < top_k; ++i) {
        top_ids[i] = -1;
        top_scores[i] = -FLT_MAX;
    }

    if (!read_embedding_avg(embed_f, &embed, token_ids, token_count, hidden_vec, hidden)) {
        fprintf(stderr, "[native-real] read_embedding_failed\n");
        fclose(embed_f);
        fclose(lm_f);
        free(hidden_vec);
        free(lm_row);
        free(top_ids);
        free(top_scores);
        return 13;
    }

    for (size_t vid = 0U; vid < vocab; ++vid) {
        const uint64_t row_off = lm_head.file_offset + ((uint64_t)vid * (uint64_t)hidden * (uint64_t)elem_bytes);
        float acc = 0.0f;
        if (fseek(lm_f, (long)row_off, SEEK_SET) != 0) {
            fprintf(stderr, "[native-real] fseek_lm_failed vid=%zu\n", vid);
            fclose(embed_f);
            fclose(lm_f);
            free(hidden_vec);
            free(lm_row);
            free(top_ids);
            free(top_scores);
            return 14;
        }
        if (fread(lm_row, 1, hidden * elem_bytes, lm_f) != hidden * elem_bytes) {
            fprintf(stderr, "[native-real] fread_lm_failed vid=%zu\n", vid);
            fclose(embed_f);
            fclose(lm_f);
            free(hidden_vec);
            free(lm_row);
            free(top_ids);
            free(top_scores);
            return 15;
        }
        for (size_t j = 0U; j < hidden; ++j) {
            acc += hidden_vec[j] * decode_scalar(&lm_row[j * elem_bytes], lm_head.info->dtype);
        }
        (void)topk_insert(top_ids, top_scores, top_k, (int)vid, acc);
    }

    printf("[native-real] mode=embed_mean_plus_lm_head\n");
    printf("[native-real] model_path=%s\n", model_path);
    printf("[native-real] embed_file=%s\n", embed_loc.file_path);
    printf("[native-real] lm_head_file=%s\n", lm_loc.file_path);
    printf("[native-real] tied_lm_head=%s\n", tied_lm_head ? "yes" : "no");
    printf("[native-real] embed=%s shape=[%zu,%zu]\n", embed.info->dtype, vocab, hidden);
    printf("[native-real] lm_head=%s shape=[%zu,%zu]\n", lm_head.info->dtype, lm_head.info->shape[0], lm_head.info->shape[1]);
    printf("[native-real] prompt_token_count=%zu top_k=%zu\n", token_count, top_k);
    for (size_t i = 0U; i < top_k; ++i) {
        if (top_ids[i] >= 0) {
            printf("[native-real] top%zu token_id=%d logit=%.6f\n", i + 1U, top_ids[i], top_scores[i]);
        }
    }

    fclose(embed_f);
    fclose(lm_f);
    free(hidden_vec);
    free(lm_row);
    free(top_ids);
    free(top_scores);
    return 0;
}
