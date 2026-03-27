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

typedef struct TensorLocation {
    VspecCompatTensorInfo info;
    char file_path[1024];
    uint64_t data_base;
} TensorLocation;

typedef struct ModelTensors {
    TensorLocation embed;
    TensorLocation lm_head;
    int tied_lm_head;
} ModelTensors;

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

static int resolve_tensors(const char* model_path, ModelTensors* out) {
    if (!model_path || !out) {
        return 0;
    }

    memset(out, 0, sizeof(*out));

    if (!is_directory_path(model_path)) {
        const int ok_embed = find_canonical_in_file(model_path, "model.embed_tokens.weight", &out->embed);
        const int ok_lm = find_canonical_in_file(model_path, "lm_head.weight", &out->lm_head);
        if (ok_embed && ok_lm) {
            return 1;
        }
        if (ok_embed && !ok_lm) {
            out->lm_head = out->embed;
            out->tied_lm_head = 1;
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
            if (out->embed.file_path[0] == '\0') {
                (void)find_canonical_in_file(file_path, "model.embed_tokens.weight", &out->embed);
            }
            if (out->lm_head.file_path[0] == '\0') {
                (void)find_canonical_in_file(file_path, "lm_head.weight", &out->lm_head);
            }
            if (out->embed.file_path[0] != '\0' && out->lm_head.file_path[0] != '\0') {
                FindClose(h);
                return 1;
            }
        } while (FindNextFileA(h, &data));
        FindClose(h);
    }
#endif

    if (out->embed.file_path[0] != '\0' && out->lm_head.file_path[0] == '\0') {
        out->lm_head = out->embed;
        out->tied_lm_head = 1;
        return 1;
    }
    return 0;
}

static int parse_token_ids(const char* s, int* out_ids, size_t cap, size_t* out_n) {
    char* buf = NULL;
    char* tok = NULL;
    size_t n = 0U;
    size_t len = 0U;

    if (!s || !out_ids || cap == 0U || !out_n) {
        return 0;
    }

    len = strlen(s);
    if (len == 0U || len > 1024U * 1024U) {
        return 0;
    }

    buf = (char*)malloc(len + 1U);
    if (!buf) {
        return 0;
    }
    memcpy(buf, s, len + 1U);

    tok = strtok(buf, ", ");
    while (tok && n < cap) {
        char* endp = NULL;
        long v = strtol(tok, &endp, 10);
        if (endp == tok || v < 0 || v > 100000000L) {
            free(buf);
            return 0;
        }
        out_ids[n++] = (int)v;
        tok = strtok(NULL, ", ");
    }

    free(buf);
    if (n == 0U) {
        return 0;
    }
    *out_n = n;
    return 1;
}

static int read_embed_row(
    FILE* f,
    const TensorLocation* embed,
    int token_id,
    float* out_row,
    size_t hidden
) {
    const size_t elem_bytes = dtype_bytes(embed->info.dtype);
    unsigned char* raw = NULL;
    const uint64_t vocab = embed->info.shape[0];

    if (!f || !embed || token_id < 0 || !out_row || hidden == 0U || elem_bytes == 0U) {
        return 0;
    }
    if ((uint64_t)token_id >= vocab) {
        return 0;
    }

    raw = (unsigned char*)malloc(hidden * elem_bytes);
    if (!raw) {
        return 0;
    }

    {
        const uint64_t base = embed->data_base + embed->info.data_offset_start;
        const uint64_t off = base + ((uint64_t)(uint32_t)token_id * (uint64_t)hidden * (uint64_t)elem_bytes);
        if (fseek(f, (long)off, SEEK_SET) != 0) {
            free(raw);
            return 0;
        }
        if (fread(raw, 1, hidden * elem_bytes, f) != hidden * elem_bytes) {
            free(raw);
            return 0;
        }
    }

    for (size_t j = 0U; j < hidden; ++j) {
        out_row[j] = decode_scalar(&raw[j * elem_bytes], embed->info.dtype);
    }

    free(raw);
    return 1;
}

static int argmax_next_token(
    FILE* lm_f,
    const TensorLocation* lm,
    const float* hidden_vec,
    size_t hidden,
    int* out_token,
    float* out_logit
) {
    const size_t elem_bytes = dtype_bytes(lm->info.dtype);
    const size_t vocab = lm->info.shape[0];
    const uint64_t base = lm->data_base + lm->info.data_offset_start;
    unsigned char* row = NULL;
    float best = -FLT_MAX;
    int best_id = -1;

    if (!lm_f || !lm || !hidden_vec || hidden == 0U || !out_token || !out_logit || elem_bytes == 0U) {
        return 0;
    }

    row = (unsigned char*)malloc(hidden * elem_bytes);
    if (!row) {
        return 0;
    }

    for (size_t vid = 0U; vid < vocab; ++vid) {
        const uint64_t off = base + ((uint64_t)vid * (uint64_t)hidden * (uint64_t)elem_bytes);
        float acc = 0.0f;
        if (fseek(lm_f, (long)off, SEEK_SET) != 0) {
            free(row);
            return 0;
        }
        if (fread(row, 1, hidden * elem_bytes, lm_f) != hidden * elem_bytes) {
            free(row);
            return 0;
        }
        for (size_t j = 0U; j < hidden; ++j) {
            acc += hidden_vec[j] * decode_scalar(&row[j * elem_bytes], lm->info.dtype);
        }
        if (acc > best) {
            best = acc;
            best_id = (int)vid;
        }
    }

    free(row);
    if (best_id < 0) {
        return 0;
    }

    *out_token = best_id;
    *out_logit = best;
    return 1;
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* token_ids_csv = NULL;
    int prompt_ids[16384];
    size_t prompt_n = 0U;
    int max_new_tokens = 64;
    int eos_token_id = -1;
    ModelTensors mt;
    FILE* embed_f = NULL;
    FILE* lm_f = NULL;
    size_t hidden = 0U;
    size_t vocab = 0U;
    float* sum_vec = NULL;
    float* row_vec = NULL;
    float* hidden_vec = NULL;
    int* generated = NULL;
    size_t gen_n = 0U;
    size_t context_count = 0U;

    if (argc < 3) {
        fprintf(stderr, "usage: native_real_decode <safetensors-file|model-dir> <token_ids_csv> [max_new_tokens] [eos_token_id]\n");
        return 2;
    }

    model_path = argv[1];
    token_ids_csv = argv[2];
    if (argc >= 4) {
        const int v = atoi(argv[3]);
        if (v > 0 && v <= 1024) {
            max_new_tokens = v;
        }
    }
    if (argc >= 5) {
        eos_token_id = atoi(argv[4]);
    }

    if (!parse_token_ids(token_ids_csv, prompt_ids, sizeof(prompt_ids) / sizeof(prompt_ids[0]), &prompt_n)) {
        fprintf(stderr, "[native-real-decode] invalid token_ids_csv\n");
        return 3;
    }

    if (!resolve_tensors(model_path, &mt)) {
        fprintf(stderr, "[native-real-decode] missing canonical tensors embed/lm_head\n");
        return 4;
    }

    if (mt.embed.info.ndim < 2U || mt.lm_head.info.ndim < 2U) {
        fprintf(stderr, "[native-real-decode] unsupported tensor rank\n");
        return 5;
    }

    hidden = mt.embed.info.shape[1];
    vocab = mt.embed.info.shape[0];
    if (hidden == 0U || vocab == 0U) {
        fprintf(stderr, "[native-real-decode] invalid embed shape\n");
        return 6;
    }
    if (mt.lm_head.info.shape[0] != vocab || mt.lm_head.info.shape[1] != hidden) {
        fprintf(stderr, "[native-real-decode] lm_head shape mismatch\n");
        return 7;
    }
    if (dtype_bytes(mt.embed.info.dtype) == 0U || dtype_bytes(mt.lm_head.info.dtype) == 0U) {
        fprintf(stderr, "[native-real-decode] unsupported dtype\n");
        return 8;
    }

    embed_f = fopen(mt.embed.file_path, "rb");
    lm_f = fopen(mt.lm_head.file_path, "rb");
    if (!embed_f || !lm_f) {
        fprintf(stderr, "[native-real-decode] open_failed\n");
        if (embed_f) fclose(embed_f);
        if (lm_f) fclose(lm_f);
        return 9;
    }

    sum_vec = (float*)calloc(hidden, sizeof(float));
    row_vec = (float*)malloc(hidden * sizeof(float));
    hidden_vec = (float*)malloc(hidden * sizeof(float));
    generated = (int*)malloc((size_t)max_new_tokens * sizeof(int));
    if (!sum_vec || !row_vec || !hidden_vec || !generated) {
        fprintf(stderr, "[native-real-decode] alloc_failed\n");
        fclose(embed_f);
        fclose(lm_f);
        free(sum_vec);
        free(row_vec);
        free(hidden_vec);
        free(generated);
        return 10;
    }

    for (size_t i = 0U; i < prompt_n; ++i) {
        if (!read_embed_row(embed_f, &mt.embed, prompt_ids[i], row_vec, hidden)) {
            fprintf(stderr, "[native-real-decode] invalid prompt token id=%d\n", prompt_ids[i]);
            fclose(embed_f);
            fclose(lm_f);
            free(sum_vec);
            free(row_vec);
            free(hidden_vec);
            free(generated);
            return 11;
        }
        for (size_t j = 0U; j < hidden; ++j) {
            sum_vec[j] += row_vec[j];
        }
        context_count += 1U;
    }

    for (int step = 0; step < max_new_tokens; ++step) {
        int next_id = -1;
        float next_logit = 0.0f;

        for (size_t j = 0U; j < hidden; ++j) {
            hidden_vec[j] = sum_vec[j] / (float)context_count;
        }

        if (!argmax_next_token(lm_f, &mt.lm_head, hidden_vec, hidden, &next_id, &next_logit)) {
            fprintf(stderr, "[native-real-decode] argmax_failed step=%d\n", step);
            fclose(embed_f);
            fclose(lm_f);
            free(sum_vec);
            free(row_vec);
            free(hidden_vec);
            free(generated);
            return 12;
        }

        generated[gen_n++] = next_id;
        printf("[native-real-decode] step=%d token_id=%d logit=%.6f\n", step + 1, next_id, next_logit);

        if (eos_token_id >= 0 && next_id == eos_token_id) {
            break;
        }

        if (!read_embed_row(embed_f, &mt.embed, next_id, row_vec, hidden)) {
            fprintf(stderr, "[native-real-decode] generated token out of range id=%d\n", next_id);
            fclose(embed_f);
            fclose(lm_f);
            free(sum_vec);
            free(row_vec);
            free(hidden_vec);
            free(generated);
            return 13;
        }
        for (size_t j = 0U; j < hidden; ++j) {
            sum_vec[j] += row_vec[j];
        }
        context_count += 1U;
    }

    printf("[native-real-decode] mode=embed_mean_plus_lm_head_greedy\n");
    printf("[native-real-decode] model_path=%s\n", model_path);
    printf("[native-real-decode] embed_file=%s\n", mt.embed.file_path);
    printf("[native-real-decode] lm_head_file=%s\n", mt.lm_head.file_path);
    printf("[native-real-decode] tied_lm_head=%s\n", mt.tied_lm_head ? "yes" : "no");
    printf("[native-real-decode] generated_count=%zu\n", gen_n);
    printf("[native-real-decode] generated_ids=");
    for (size_t i = 0U; i < gen_n; ++i) {
        printf("%s%d", (i == 0U) ? "" : ",", generated[i]);
    }
    printf("\n");

    fclose(embed_f);
    fclose(lm_f);
    free(sum_vec);
    free(row_vec);
    free(hidden_vec);
    free(generated);
    return 0;
}
