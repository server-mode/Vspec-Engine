#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/quant/int4.h"
#include "vspec/quant/pack.h"
#include "vspec/runtime/qlora_adapter.h"
#include "vspec/runtime/runtime.h"

static float frand_sym(unsigned int* seed) {
    *seed = (*seed * 1664525u) + 1013904223u;
    const float x = (float)((*seed >> 8) & 0xFFFFu) / 65535.0f;
    return (x * 2.0f) - 1.0f;
}

static void dense_matmul(const float* a, const float* w_kxn, size_t m, size_t k, size_t n, float* out) {
    for (size_t i = 0; i < m; ++i) {
        const float* a_row = a + (i * k);
        float* o_row = out + (i * n);
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t t = 0; t < k; ++t) {
                acc += a_row[t] * w_kxn[t * n + j];
            }
            o_row[j] = acc;
        }
    }
}

static float mse(const float* pred, const float* target, size_t count) {
    double acc = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double d = (double)pred[i] - (double)target[i];
        acc += d * d;
    }
    return (count > 0U) ? (float)(acc / (double)count) : 0.0f;
}

static float cosine_similarity(const float* a, const float* b, size_t count) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double va = (double)a[i];
        const double vb = (double)b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0f;
    }
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

int main(void) {
    const size_t m = 32;
    const size_t k = 128;
    const size_t n = 96;
    const size_t rank = 8;
    const uint32_t layer_id = 1U;

    unsigned int seed = 7u;

    float* input = (float*)malloc(m * k * sizeof(float));
    float* w_base = (float*)malloc(k * n * sizeof(float));
    float* w_target = (float*)malloc(k * n * sizeof(float));
    float* out_fp32_target = (float*)malloc(m * n * sizeof(float));
    float* out_fp32_base = (float*)malloc(m * n * sizeof(float));
    float* out_int4 = (float*)malloc(m * n * sizeof(float));
    float* out_int3 = (float*)malloc(m * n * sizeof(float));
    float* out_ultimate = (float*)malloc(m * n * sizeof(float));

    float* lora_a = (float*)malloc(k * rank * sizeof(float));
    float* lora_b = (float*)malloc(rank * n * sizeof(float));

    uint8_t* w4 = (uint8_t*)malloc(n * vspec_int4_packed_bytes(k));
    uint8_t* w3 = (uint8_t*)malloc(n * vspec_quant_packed_bytes(k, 3));
    int8_t* qtmp4 = (int8_t*)malloc(k * sizeof(int8_t));
    int8_t* qtmp3 = (int8_t*)malloc(k * sizeof(int8_t));
    float* scales4 = (float*)malloc(n * sizeof(float));
    float* scales3 = (float*)malloc(n * sizeof(float));

    if (!input || !w_base || !w_target || !out_fp32_target || !out_fp32_base || !out_int4 || !out_int3 || !out_ultimate ||
        !lora_a || !lora_b || !w4 || !w3 || !qtmp4 || !qtmp3 || !scales4 || !scales3) {
        return 1;
    }

    for (size_t i = 0; i < m * k; ++i) {
        input[i] = frand_sym(&seed) * 1.2f;
    }
    for (size_t i = 0; i < k * n; ++i) {
        w_base[i] = frand_sym(&seed) * 0.4f;
    }
    for (size_t i = 0; i < k * rank; ++i) {
        lora_a[i] = frand_sym(&seed) * 0.07f;
    }
    for (size_t i = 0; i < rank * n; ++i) {
        lora_b[i] = frand_sym(&seed) * 0.07f;
    }

    for (size_t t = 0; t < k; ++t) {
        for (size_t j = 0; j < n; ++j) {
            float delta = 0.0f;
            for (size_t r = 0; r < rank; ++r) {
                delta += lora_a[t * rank + r] * lora_b[r * n + j];
            }
            w_target[t * n + j] = w_base[t * n + j] + (delta * (8.0f / (float)rank));
        }
    }

    dense_matmul(input, w_base, m, k, n, out_fp32_base);
    dense_matmul(input, w_target, m, k, n, out_fp32_target);

    for (size_t j = 0; j < n; ++j) {
        float max_abs = 1e-6f;
        for (size_t t = 0; t < k; ++t) {
            const float v = fabsf(w_base[t * n + j]);
            if (v > max_abs) {
                max_abs = v;
            }
        }

        scales4[j] = max_abs / 7.0f;
        scales3[j] = max_abs / 3.0f;
        if (scales4[j] <= 0.0f) scales4[j] = 1e-6f;
        if (scales3[j] <= 0.0f) scales3[j] = 1e-6f;

        for (size_t t = 0; t < k; ++t) {
            int q4 = (int)lroundf(w_base[t * n + j] / scales4[j]);
            int q3 = (int)lroundf(w_base[t * n + j] / scales3[j]);
            if (q4 < -8) q4 = -8;
            if (q4 > 7) q4 = 7;
            if (q3 < -4) q3 = -4;
            if (q3 > 3) q3 = 3;
            qtmp4[t] = (int8_t)q4;
            qtmp3[t] = (int8_t)q3;
        }

        vspec_int4_pack(qtmp4, k, &w4[j * vspec_int4_packed_bytes(k)]);
        vspec_quant_pack_signed(qtmp3, k, 3, &w3[j * vspec_quant_packed_bytes(k, 3)]);
    }

    vspec_int4_matmul_ref_f32_q4(input, m, k, w4, n, scales4, out_int4);
    vspec_int3_matmul_ref_f32_q3(input, m, k, w3, n, scales3, out_int3);

    (void)memcpy(out_ultimate, out_int4, m * n * sizeof(float));
    vspec_qlora_adapter_clear();
    (void)vspec_qlora_adapter_add_layer(layer_id, k, rank, n, 8.0f, lora_a, lora_b);
    vspec_qlora_adapter_apply_layer_f32(layer_id, input, m, k, n, out_ultimate);

    const float mse_int4 = mse(out_int4, out_fp32_target, m * n);
    const float mse_int3 = mse(out_int3, out_fp32_target, m * n);
    const float mse_ultimate = mse(out_ultimate, out_fp32_target, m * n);

    const float cos_int4 = cosine_similarity(out_int4, out_fp32_target, m * n);
    const float cos_int3 = cosine_similarity(out_int3, out_fp32_target, m * n);
    const float cos_ultimate = cosine_similarity(out_ultimate, out_fp32_target, m * n);

    VspecRuntimeHwConfig cfg;
    vspec_runtime_hw_config_default(&cfg);
    cfg.enable_ultimate_mode = 1;
    cfg.enable_outlier_aware = 1;
    cfg.enable_qlora_adapter = 1;
    cfg.quality_bias = 0.90f;

    VspecRuntimeUltimateState ultimate;
    vspec_runtime_ultimate_init(&ultimate, &cfg);
    VspecQuantType recommended = vspec_runtime_ultimate_recommend_quant(&ultimate, input, m * k);

    printf("[ultimate_quality] shape m=%zu k=%zu n=%zu rank=%zu\n", m, k, n, rank);
    printf("[ultimate_quality] recommended_quant=%d outlier_ratio=%.4f\n", (int)recommended, ultimate.latest_outlier_ratio);
    printf("[ultimate_quality] mse int3=%.6f int4=%.6f ultimate+qlora=%.6f\n", mse_int3, mse_int4, mse_ultimate);
    printf("[ultimate_quality] cos int3=%.6f int4=%.6f ultimate+qlora=%.6f\n", cos_int3, cos_int4, cos_ultimate);

    free(input);
    free(w_base);
    free(w_target);
    free(out_fp32_target);
    free(out_fp32_base);
    free(out_int4);
    free(out_int3);
    free(out_ultimate);
    free(lora_a);
    free(lora_b);
    free(w4);
    free(w3);
    free(qtmp4);
    free(qtmp3);
    free(scales4);
    free(scales3);

    return 0;
}