#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#include <errno.h>
#else
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "vspec/runtime/runtime.h"

static int write_float_blob(const char* path, const float* data, size_t count) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        return 0;
    }
    const size_t written = fwrite(data, sizeof(float), count, file);
    fclose(file);
    return written == count ? 1 : 0;
}

static int write_manifest(const char* path) {
    static const char* manifest_text =
        "{\n"
        "  \"version\": 1,\n"
        "  \"layers\": [\n"
        "    {\n"
        "      \"layer_id\": 101,\n"
        "      \"in_dim\": 4,\n"
        "      \"rank\": 2,\n"
        "      \"out_dim\": 3,\n"
        "      \"alpha\": 2.0,\n"
        "      \"a_path\": \"qlora_l101_a.bin\",\n"
        "      \"b_path\": \"qlora_l101_b.bin\"\n"
        "    },\n"
        "    {\n"
        "      \"layer_id\": 102,\n"
        "      \"in_dim\": 4,\n"
        "      \"rank\": 2,\n"
        "      \"out_dim\": 3,\n"
        "      \"alpha\": 2.0,\n"
        "      \"a_path\": \"qlora_l102_a.bin\",\n"
        "      \"b_path\": \"qlora_l102_b.bin\"\n"
        "    }\n"
        "  ]\n"
        "}\n";

    FILE* file = fopen(path, "wb");
    if (!file) {
        return 0;
    }
    const size_t len = strlen(manifest_text);
    const size_t written = fwrite(manifest_text, 1, len, file);
    fclose(file);
    return written == len ? 1 : 0;
}

static int write_text_file(const char* path, const char* text) {
    if (!path || !text) {
        return 0;
    }
    FILE* file = fopen(path, "wb");
    if (!file) {
        return 0;
    }
    const size_t len = strlen(text);
    const size_t written = fwrite(text, 1, len, file);
    fclose(file);
    return written == len ? 1 : 0;
}

static int make_dir_if_missing(const char* dir_path) {
    if (!dir_path || dir_path[0] == '\0') {
        return 0;
    }
#if defined(_WIN32)
    if (_mkdir(dir_path) == 0) {
        return 1;
    }
#else
    if (mkdir(dir_path, 0755) == 0) {
        return 1;
    }
#endif
    return errno == EEXIST ? 1 : 0;
}

static void apply_expected(
    const float* input,
    size_t m,
    size_t k,
    const float* a,
    size_t rank,
    const float* b,
    size_t n,
    float alpha,
    float* out
) {
    const float scale = alpha / (float)rank;
    float* tmp = (float*)malloc(rank * sizeof(float));
    if (!tmp) {
        return;
    }

    for (size_t row = 0; row < m; ++row) {
        const float* in_row = input + (row * k);
        float* out_row = out + (row * n);
        for (size_t r = 0; r < rank; ++r) {
            float acc = 0.0f;
            for (size_t t = 0; t < k; ++t) {
                acc += in_row[t] * a[t * rank + r];
            }
            tmp[r] = acc;
        }
        for (size_t col = 0; col < n; ++col) {
            float delta = 0.0f;
            for (size_t r = 0; r < rank; ++r) {
                delta += tmp[r] * b[r * n + col];
            }
            out_row[col] += delta * scale;
        }
    }

    free(tmp);
}

int main(void) {
    const size_t m = 2;
    const size_t k = 4;
    const size_t rank = 2;
    const size_t n = 3;

    const float input[] = {
        1.0f, 0.5f, -1.0f, 2.0f,
        -0.5f, 1.5f, 0.0f, 1.0f
    };

    const float a101[] = {
        0.10f, -0.20f,
        0.05f, 0.10f,
        -0.10f, 0.05f,
        0.20f, 0.15f
    };
    const float b101[] = {
        0.30f, -0.20f, 0.10f,
        -0.10f, 0.25f, 0.20f
    };

    const float a102[] = {
        0.20f, 0.10f,
        -0.10f, 0.20f,
        0.15f, -0.05f,
        0.05f, 0.10f
    };
    const float b102[] = {
        0.05f, 0.15f, -0.10f,
        0.20f, -0.05f, 0.30f
    };

    const char* base_dir = "qlora_manifest_assets";
    char manifest_path[256] = {0};
    char missing_blob_manifest_path[256] = {0};
    char missing_key_manifest_path[256] = {0};
    char a101_path[256] = {0};
    char b101_path[256] = {0};
    char a102_path[256] = {0};
    char b102_path[256] = {0};
    const float alpha = 2.0f;

    if (!make_dir_if_missing(base_dir)) {
        printf("[qlora_manifest_smoke] failed to prepare asset directory\n");
        return 1;
    }

    (void)snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", base_dir);
    (void)snprintf(missing_blob_manifest_path, sizeof(missing_blob_manifest_path), "%s/manifest_missing_blob.json", base_dir);
    (void)snprintf(missing_key_manifest_path, sizeof(missing_key_manifest_path), "%s/manifest_missing_key.json", base_dir);
    (void)snprintf(a101_path, sizeof(a101_path), "%s/qlora_l101_a.bin", base_dir);
    (void)snprintf(b101_path, sizeof(b101_path), "%s/qlora_l101_b.bin", base_dir);
    (void)snprintf(a102_path, sizeof(a102_path), "%s/qlora_l102_a.bin", base_dir);
    (void)snprintf(b102_path, sizeof(b102_path), "%s/qlora_l102_b.bin", base_dir);

    if (!write_float_blob(a101_path, a101, k * rank) ||
        !write_float_blob(b101_path, b101, rank * n) ||
        !write_float_blob(a102_path, a102, k * rank) ||
        !write_float_blob(b102_path, b102, rank * n) ||
        !write_manifest(manifest_path)) {
        printf("[qlora_manifest_smoke] failed to create test files\n");
        return 1;
    }

    vspec_runtime_init_default();
    vspec_runtime_qlora_clear();

    const int loaded = vspec_runtime_qlora_load_manifest_json(manifest_path);
    printf("[qlora_manifest_smoke] loaded=%d\n", loaded);

    if (loaded != 2 || !vspec_qlora_adapter_has_layer(101U) || !vspec_qlora_adapter_has_layer(102U)) {
        printf("[qlora_manifest_smoke] layer presence check failed\n");
        vspec_runtime_qlora_clear();
        return 2;
    }

    float* output_actual = (float*)calloc(m * n, sizeof(float));
    float* output_expected = (float*)calloc(m * n, sizeof(float));
    if (!output_actual || !output_expected) {
        if (output_actual) {
            free(output_actual);
        }
        if (output_expected) {
            free(output_expected);
        }
        vspec_runtime_qlora_clear();
        return 4;
    }

    vspec_qlora_adapter_apply_layer_f32(101U, input, m, k, n, output_actual);
    apply_expected(input, m, k, a101, rank, b101, n, alpha, output_expected);

    float max_abs_err = 0.0f;
    for (size_t i = 0; i < (m * n); ++i) {
        const float err = fabsf(output_actual[i] - output_expected[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    printf("[qlora_manifest_smoke] max_abs_err=%.8f\n", max_abs_err);
    if (max_abs_err > 1e-5f) {
        printf("[qlora_manifest_smoke] numerical check failed\n");
        free(output_actual);
        free(output_expected);
        vspec_runtime_qlora_clear();
        return 3;
    }

    free(output_actual);
    free(output_expected);

    {
        static const char* missing_blob_manifest =
            "{\n"
            "  \"version\": 1,\n"
            "  \"layers\": [\n"
            "    {\n"
            "      \"layer_id\": 201,\n"
            "      \"in_dim\": 4,\n"
            "      \"rank\": 2,\n"
            "      \"out_dim\": 3,\n"
            "      \"alpha\": 2.0,\n"
            "      \"a_path\": \"qlora_l101_a.bin\",\n"
            "      \"b_path\": \"missing_b.bin\"\n"
            "    }\n"
            "  ]\n"
            "}\n";

        static const char* missing_key_manifest =
            "{\n"
            "  \"version\": 1,\n"
            "  \"layers\": [\n"
            "    {\n"
            "      \"layer_id\": 202,\n"
            "      \"in_dim\": 4,\n"
            "      \"rank\": 2,\n"
            "      \"out_dim\": 3,\n"
            "      \"alpha\": 2.0,\n"
            "      \"a_path\": \"qlora_l101_a.bin\"\n"
            "    }\n"
            "  ]\n"
            "}\n";

        if (!write_text_file(missing_blob_manifest_path, missing_blob_manifest) ||
            !write_text_file(missing_key_manifest_path, missing_key_manifest)) {
            printf("[qlora_manifest_smoke] failed to create negative manifests\n");
            vspec_runtime_qlora_clear();
            return 5;
        }

        vspec_runtime_qlora_clear();
        if (vspec_runtime_qlora_load_manifest_json(missing_blob_manifest_path) != 0 ||
            vspec_qlora_adapter_has_layer(201U)) {
            printf("[qlora_manifest_smoke] missing-blob regression failed\n");
            vspec_runtime_qlora_clear();
            return 6;
        }

        vspec_runtime_qlora_clear();
        if (vspec_runtime_qlora_load_manifest_json(missing_key_manifest_path) != 0 ||
            vspec_qlora_adapter_has_layer(202U)) {
            printf("[qlora_manifest_smoke] missing-key regression failed\n");
            vspec_runtime_qlora_clear();
            return 7;
        }
    }

    vspec_runtime_qlora_clear();
    printf("[qlora_manifest_smoke] status=pass\n");
    return 0;
}