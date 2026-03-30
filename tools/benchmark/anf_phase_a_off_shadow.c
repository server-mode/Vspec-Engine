#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vspec/runtime/runtime.h"

#if defined(_WIN32)
#include <windows.h>
#endif

enum {
    VSPEC_LANEA_SCORE_DIM = 20,
    VSPEC_LANEA_MAX_TOKENS = 128,
    VSPEC_LANEA_MAX_SAMPLES = 8192
};

static uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static float prng_signed(uint32_t seed) {
    const uint32_t h = hash_u32(seed);
    const float unit = (float)(h & 0xFFFFU) / 65535.0f;
    return (unit * 2.0f) - 1.0f;
}

static int cmp_float_asc(const void* a, const void* b) {
    const float fa = *(const float*)a;
    const float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static double now_ms(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) {
        (void)QueryPerformanceFrequency(&freq);
    }
    (void)QueryPerformanceCounter(&t);
    return ((double)t.QuadPart * 1000.0) / (double)freq.QuadPart;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

static float percentile(float* v, size_t n, float p) {
    if (!v || n == 0U) {
        return 0.0f;
    }
    qsort(v, n, sizeof(float), cmp_float_asc);
    {
        size_t idx = (size_t)((((double)n - 1.0) * (double)p) + 0.5);
        if (idx >= n) idx = n - 1U;
        return v[idx];
    }
}

static void synth_scores(const char* prompt, size_t token_step, float out[VSPEC_LANEA_SCORE_DIM]) {
    uint32_t base = 2166136261U;
    for (const unsigned char* p = (const unsigned char*)prompt; p && *p; ++p) {
        base ^= (uint32_t)(*p);
        base *= 16777619U;
    }
    for (size_t i = 0U; i < VSPEC_LANEA_SCORE_DIM; ++i) {
        const uint32_t s = base ^ (uint32_t)(token_step * 131U) ^ (uint32_t)(i * 2654435761U);
        out[i] = 2.2f * prng_signed(s) + 0.05f * (float)(i % 5U);
    }
}

static float mse(const float* a, const float* b, size_t n) {
    double acc = 0.0;
    if (!a || !b || n == 0U) {
        return 0.0f;
    }
    for (size_t i = 0U; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return (float)(acc / (double)n);
}

static int run_mode(
    const char* mode,
    const char* const* prompts,
    size_t prompt_count,
    size_t tokens_per_prompt,
    float* out_latency_samples,
    size_t max_samples,
    size_t* out_sample_count,
    float* out_quality_mse,
    VspecRuntimeBehaviorReport* out_report
) {
    const size_t repeat_per_sample = 64U;
    size_t sample_count = 0U;
    double mse_acc = 0.0;
    size_t mse_n = 0U;
    float baseline[VSPEC_LANEA_SCORE_DIM];
    float shadow_scores[VSPEC_LANEA_SCORE_DIM];

#if defined(_WIN32)
    (void)_putenv_s("VSPEC_ENABLE_ANF", "1");
    (void)_putenv_s("VSPEC_ANF_MODE", mode ? mode : "off");
#else
    (void)setenv("VSPEC_ENABLE_ANF", "1", 1);
    (void)setenv("VSPEC_ANF_MODE", mode ? mode : "off", 1);
#endif

    vspec_runtime_init_default();

    for (size_t p = 0U; p < prompt_count; ++p) {
        for (size_t t = 0U; t < tokens_per_prompt; ++t) {
            const double begin = now_ms();
            for (size_t rep = 0U; rep < repeat_per_sample; ++rep) {
                synth_scores(prompts[p], t + rep, shadow_scores);
                if (strcmp(mode, "shadow") == 0) {
                    vspec_runtime_anf_observe_token_activations(shadow_scores, VSPEC_LANEA_SCORE_DIM);
                }
            }
            const double end = now_ms();

            synth_scores(prompts[p], t, baseline);
            mse_acc += (double)mse(baseline, shadow_scores, VSPEC_LANEA_SCORE_DIM);
            mse_n += 1U;

            if (sample_count < max_samples) {
                const float total_ms = (float)(end - begin);
                out_latency_samples[sample_count++] = total_ms / (float)repeat_per_sample;
            }
        }
    }

    if (out_sample_count) {
        *out_sample_count = sample_count;
    }
    if (out_quality_mse) {
        *out_quality_mse = (mse_n > 0U) ? (float)(mse_acc / (double)mse_n) : 0.0f;
    }
    if (out_report) {
        vspec_runtime_behavior_report(out_report);
    }
    return 1;
}

int main(void) {
    static const char* kPrompts[] = {
        "Tinh tong 245 + 367 va giai thich ngan.",
        "Viet mot doan van 2 cau ve nang luong mat troi.",
        "Neu gia la 1.10$ va giam 20% thi con bao nhieu?",
        "Liet ke 3 buoc debug loi memory leak.",
        "Tom tat su khac nhau giua CPU va GPU trong 2 cau.",
        "Cho vi du ve queue va stack trong lap trinh.",
        "Dich cau nay sang tieng Anh: toi dang toi uu he thong.",
        "Viet checklist release nho cho mot ban build moi."
    };

    float off_samples[VSPEC_LANEA_MAX_SAMPLES];
    float shadow_samples[VSPEC_LANEA_MAX_SAMPLES];
    size_t off_n = 0U;
    size_t shadow_n = 0U;
    float off_mse = 0.0f;
    float shadow_mse = 0.0f;
    VspecRuntimeBehaviorReport off_report;
    VspecRuntimeBehaviorReport shadow_report;

    if (!run_mode("off", kPrompts, sizeof(kPrompts) / sizeof(kPrompts[0]), VSPEC_LANEA_MAX_TOKENS,
            off_samples, VSPEC_LANEA_MAX_SAMPLES, &off_n, &off_mse, &off_report)) {
        printf("[anf_phase_a_off_shadow] status=fail off_run\n");
        return 2;
    }

    if (!run_mode("shadow", kPrompts, sizeof(kPrompts) / sizeof(kPrompts[0]), VSPEC_LANEA_MAX_TOKENS,
            shadow_samples, VSPEC_LANEA_MAX_SAMPLES, &shadow_n, &shadow_mse, &shadow_report)) {
        printf("[anf_phase_a_off_shadow] status=fail shadow_run\n");
        return 3;
    }

    {
        const float off_median = percentile(off_samples, off_n, 0.50f);
        const float off_p95 = percentile(off_samples, off_n, 0.95f);
        const float shadow_median = percentile(shadow_samples, shadow_n, 0.50f);
        const float shadow_p95 = percentile(shadow_samples, shadow_n, 0.95f);
        const float overhead_pct = (off_median > 1e-9f) ? ((shadow_median - off_median) / off_median) * 100.0f : 0.0f;

        printf("[anf_phase_a_off_shadow] samples=%zu prompts=%zu tokens_per_prompt=%d\n", shadow_n, sizeof(kPrompts) / sizeof(kPrompts[0]), VSPEC_LANEA_MAX_TOKENS);
        printf("[anf_phase_a_off_shadow] off_median_ms=%.6f off_p95_ms=%.6f\n", off_median, off_p95);
        printf("[anf_phase_a_off_shadow] shadow_median_ms=%.6f shadow_p95_ms=%.6f overhead_pct=%.2f\n", shadow_median, shadow_p95, overhead_pct);
        printf("[anf_phase_a_off_shadow] mse_off=%.10f mse_shadow=%.10f\n", off_mse, shadow_mse);
        printf("[anf_phase_a_off_shadow] shadow_tokens=%u hot_ratio_avg=%.4f hot_ratio_p95=%.4f last_hot=%u mode=%d\n",
            shadow_report.anf_tokens_observed,
            shadow_report.anf_hot_ratio_avg,
            shadow_report.anf_hot_ratio_p95,
            shadow_report.anf_hot_neurons,
            shadow_report.anf_mode);
    }

    printf("[anf_phase_a_off_shadow] status=pass\n");
    return 0;
}
