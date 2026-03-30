#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#include "vspec/compat/safetensors_parser.h"
#include "vspec/runtime/native_inference.h"
#include "vspec/runtime/runtime.h"

enum {
    VSPEC_PHASEB_MAX_PROMPTS = 8,
    VSPEC_PHASEB_MAX_TOKENS = 128,
    VSPEC_PHASEB_MAX_SAMPLES = VSPEC_PHASEB_MAX_PROMPTS * VSPEC_PHASEB_MAX_TOKENS,
};

static void set_env_local(const char* key, const char* value) {
#if defined(_WIN32)
    (void)_putenv_s(key, value ? value : "");
#else
    (void)setenv(key, value ? value : "", 1);
#endif
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

static int cmp_float_asc(const void* lhs, const void* rhs) {
    const float a = *(const float*)lhs;
    const float b = *(const float*)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static float percentile(float* values, size_t count, float p) {
    size_t idx;
    if (!values || count == 0U) {
        return 0.0f;
    }
    qsort(values, count, sizeof(float), cmp_float_asc);
    idx = (size_t)((((double)count - 1.0) * (double)p) + 0.5);
    if (idx >= count) {
        idx = count - 1U;
    }
    return values[idx];
}

static float mse(const float* a, const float* b, size_t count) {
    double s = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)(s / (double)(count > 0U ? count : 1U));
}

static int run_mode(
    const char* mode,
    int tcc_enable,
    const VspecCompatModel* model,
    const char* model_file,
    const char* const* prompts,
    size_t prompt_count,
    size_t tokens_per_prompt,
    float* out_latencies,
    float* out_scores,
    size_t out_capacity,
    VspecRuntimeBehaviorReport* out_report,
    size_t* out_steps
) {
    static const int candidate_ids[VSPEC_NATIVE_TOKEN_COUNT] = {
        VSPEC_NATIVE_TOKEN_THE,
        VSPEC_NATIVE_TOKEN_ANSWER,
        VSPEC_NATIVE_TOKEN_IS,
        VSPEC_NATIVE_TOKEN_5,
        VSPEC_NATIVE_TOKEN_9,
        VSPEC_NATIVE_TOKEN_CENTS,
        VSPEC_NATIVE_TOKEN_DOT,
        VSPEC_NATIVE_TOKEN_I,
        VSPEC_NATIVE_TOKEN_CAN,
        VSPEC_NATIVE_TOKEN_HELP,
        VSPEC_NATIVE_TOKEN_WITH,
        VSPEC_NATIVE_TOKEN_MATH,
        VSPEC_NATIVE_TOKEN_USING,
        VSPEC_NATIVE_TOKEN_NATIVE,
        VSPEC_NATIVE_TOKEN_INFERENCE,
        VSPEC_NATIVE_TOKEN_MULTI,
        VSPEC_NATIVE_TOKEN_MODEL,
        VSPEC_NATIVE_TOKEN_ENGINE,
        VSPEC_NATIVE_TOKEN_TODAY,
        VSPEC_NATIVE_TOKEN_MODELS
    };

    VspecNativeForwardContext fctx;
    size_t step = 0U;

    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", mode);
    set_env_local("VSPEC_ANF_TCC_ENABLE", tcc_enable ? "1" : "0");
    set_env_local("VSPEC_ANF_MAX_HOT_RATIO", "0.25");
    set_env_local("VSPEC_ANF_MIN_HOT_NEURONS", "2");
    set_env_local("VSPEC_ANF_MAX_HOT_NEURONS", "8");
    set_env_local("VSPEC_ANF_ACTIVATION_THRESHOLD", "0.80");

    vspec_runtime_init_default();
    if (!vspec_native_forward_init(&fctx, model, model_file, 7U)) {
        return 0;
    }

    for (size_t p = 0U; p < prompt_count; ++p) {
        for (size_t t = 0U; t < tokens_per_prompt; ++t) {
            float scores[VSPEC_NATIVE_TOKEN_COUNT];
            const double begin = now_ms();
            if (!vspec_native_forward_step(
                    &fctx,
                    prompts[p],
                    t,
                    candidate_ids,
                    VSPEC_NATIVE_TOKEN_COUNT,
                    scores)) {
                return 0;
            }
            if (step < out_capacity) {
                out_latencies[step] = (float)(now_ms() - begin);
                (void)memcpy(out_scores + (step * VSPEC_NATIVE_TOKEN_COUNT),
                    scores,
                    VSPEC_NATIVE_TOKEN_COUNT * sizeof(float));
                step += 1U;
            }
        }
    }

    if (out_steps) {
        *out_steps = step;
    }
    if (out_report) {
        vspec_runtime_behavior_report(out_report);
    }
    return 1;
}

int main(void) {
    static const char* prompts[VSPEC_PHASEB_MAX_PROMPTS] = {
        "Tinh tong 245 + 367 va giai thich ngan.",
        "Viet mot doan van 2 cau ve nang luong mat troi.",
        "Neu gia la 1.10$ va giam 20% thi con bao nhieu?",
        "Liet ke 3 buoc debug loi memory leak.",
        "Tom tat su khac nhau giua CPU va GPU trong 2 cau.",
        "Cho vi du ve queue va stack trong lap trinh.",
        "Dich cau nay sang tieng Anh: toi dang toi uu he thong.",
        "Viet checklist release nho cho mot ban build moi."
    };

    const char* model_file = getenv("VSPEC_PHASEB_MODEL_FILE");
    const char* toks_env = getenv("VSPEC_PHASEB_TOKENS");
    size_t tokens_per_prompt = 64U;
    VspecCompatModel model;
    float shadow_lat[VSPEC_PHASEB_MAX_SAMPLES];
    float active_lat[VSPEC_PHASEB_MAX_SAMPLES];
    float shadow_scores[VSPEC_PHASEB_MAX_SAMPLES * VSPEC_NATIVE_TOKEN_COUNT];
    float active_scores[VSPEC_PHASEB_MAX_SAMPLES * VSPEC_NATIVE_TOKEN_COUNT];
    size_t shadow_steps = 0U;
    size_t active_steps = 0U;
    VspecRuntimeBehaviorReport shadow_report;
    VspecRuntimeBehaviorReport active_report;
    float shadow_med;
    float shadow_p95;
    float active_med;
    float active_p95;
    float speedup;
    float quality_mse;
    float quality_rel;
    int pass;

    if (!model_file || model_file[0] == '\0') {
        model_file = "sample.safetensors";
    }
    if (toks_env && toks_env[0] != '\0') {
        unsigned long v = strtoul(toks_env, NULL, 10);
        if (v > 0UL && v <= VSPEC_PHASEB_MAX_TOKENS) {
            tokens_per_prompt = (size_t)v;
        }
    }

    if (!vspec_safetensors_parse_header_file(model_file, &model)) {
        printf("[anf_phase_b_laneB] status=fail parse model_file=%s\n", model_file);
        return 2;
    }

    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", "shadow");
    set_env_local("VSPEC_ANF_TCC_ENABLE", "0");
    vspec_runtime_init_default();
    if (!vspec_runtime_anf_available()) {
        printf("[anf_phase_b_laneB] anf_available=0 (compile-time disabled)\n");
        printf("[anf_phase_b_laneB] status=pass\n");
        return 0;
    }

    if (!run_mode("shadow", 0, &model, model_file, prompts, VSPEC_PHASEB_MAX_PROMPTS,
            tokens_per_prompt, shadow_lat, shadow_scores, VSPEC_PHASEB_MAX_SAMPLES,
            &shadow_report, &shadow_steps)) {
        printf("[anf_phase_b_laneB] status=fail run shadow\n");
        return 3;
    }

    if (!run_mode("active", 1, &model, model_file, prompts, VSPEC_PHASEB_MAX_PROMPTS,
            tokens_per_prompt, active_lat, active_scores, VSPEC_PHASEB_MAX_SAMPLES,
            &active_report, &active_steps)) {
        printf("[anf_phase_b_laneB] status=fail run active+tcc\n");
        return 4;
    }

    if (shadow_steps == 0U || active_steps == 0U || shadow_steps != active_steps) {
        printf("[anf_phase_b_laneB] status=fail invalid sample count\n");
        return 5;
    }

    shadow_med = percentile(shadow_lat, shadow_steps, 0.50f);
    shadow_p95 = percentile(shadow_lat, shadow_steps, 0.95f);
    active_med = percentile(active_lat, active_steps, 0.50f);
    active_p95 = percentile(active_lat, active_steps, 0.95f);
    speedup = (active_med > 1e-9f) ? (shadow_med / active_med) : 0.0f;
    quality_mse = mse(shadow_scores, active_scores, active_steps * VSPEC_NATIVE_TOKEN_COUNT);
    quality_rel = quality_mse / 0.05f;

    printf("[anf_phase_b_laneB] model=%s prompts=%d tokens_per_prompt=%zu samples=%zu\n",
        model_file,
        VSPEC_PHASEB_MAX_PROMPTS,
        tokens_per_prompt,
        active_steps);
    printf("[anf_phase_b_laneB] shadow_median_ms=%.6f shadow_p95_ms=%.6f\n", shadow_med, shadow_p95);
    printf("[anf_phase_b_laneB] active_tcc_median_ms=%.6f active_tcc_p95_ms=%.6f speedup_vs_shadow=%.4fx\n",
        active_med,
        active_p95,
        speedup);
    printf("[anf_phase_b_laneB] skip_ratio_avg=%.4f skip_ratio_last=%.4f changed_last=%.4f confidence=%.4f cache_updates=%u\n",
        active_report.anf_skip_ratio_avg,
        active_report.anf_skip_ratio_last,
        active_report.anf_changed_ratio_last,
        active_report.anf_pattern_confidence,
        (unsigned)active_report.anf_cache_updates);
    printf("[anf_phase_b_laneB] quality_mse_active_vs_shadow=%.8f quality_rel_to_gate=%.4f\n",
        quality_mse,
        quality_rel);

    pass = 1;
    if (active_report.anf_skip_ratio_avg < 0.35f) {
        pass = 0;
    }
    if (quality_rel > 1.0f) {
        pass = 0;
    }
    if (active_med > (shadow_med * 1.10f)) {
        pass = 0;
    }

    printf("[anf_phase_b_laneB] status=%s\n", pass ? "pass" : "fail");
    return pass ? 0 : 1;
}
