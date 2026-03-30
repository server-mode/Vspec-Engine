#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#define VSPEC_BENCH_API __declspec(dllimport)
#else
#define VSPEC_BENCH_API
#endif

VSPEC_BENCH_API int vspec_cuda_fused_linear_int4_bridge(
    const float* input,
    const unsigned char* packed_weight,
    const float* scales,
    const float* zero_points,
    size_t m,
    size_t k,
    size_t n,
    size_t n_blocks,
    float* output
);

VSPEC_BENCH_API int vspec_cuda_fused_linear_hybrid_bridge(
    const float* input,
    const unsigned char* packed_weight_int2,
    const float* scales_int2,
    const unsigned char* packed_weight_int4,
    const float* scales_int4,
    const float* zero_points_int4,
    size_t m,
    size_t k,
    size_t n,
    const uint32_t* hot_indices,
    size_t hot_count,
    size_t n_blocks_int4,
    float* output
);

VSPEC_BENCH_API int vspec_cuda_fused_linear_hybrid_profile_bridge(
    const float* input,
    const unsigned char* packed_weight_int2,
    const float* scales_int2,
    const unsigned char* packed_weight_int4,
    const float* scales_int4,
    const float* zero_points_int4,
    size_t m,
    size_t k,
    size_t n,
    const uint32_t* hot_indices,
    size_t hot_count,
    size_t n_blocks_int4,
    float* output,
    float* out_h2d_ms,
    float* out_kernel_ms,
    float* out_d2h_ms,
    float* out_total_gpu_ms
);

static float rand_uniform(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
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

static size_t env_size_or_default(const char* name, size_t def) {
    const char* s = getenv(name);
    if (!s || s[0] == '\0') {
        return def;
    }
    {
        unsigned long long v = strtoull(s, NULL, 10);
        return (v > 0ULL) ? (size_t)v : def;
    }
}

static float env_float_or_default(const char* name, float def) {
    const char* s = getenv(name);
    if (!s || s[0] == '\0') {
        return def;
    }
    {
        const float v = (float)atof(s);
        return (v > 0.0f) ? v : def;
    }
}

static void ensure_env_default(const char* name, const char* value) {
    const char* cur = getenv(name);
    if (cur && cur[0] != '\0') {
        return;
    }
#if defined(_WIN32)
    (void)_putenv_s(name, value ? value : "");
#else
    (void)setenv(name, value ? value : "", 0);
#endif
}

static void quantize_pack_int2_rowwise(
    const float* w,
    size_t n,
    size_t k,
    unsigned char* packed,
    float* scales
) {
    const size_t packed_k = (k + 3U) / 4U;
    memset(packed, 0, n * packed_k * sizeof(unsigned char));
    for (size_t row = 0U; row < n; ++row) {
        const float* wr = w + row * k;
        float max_abs = 1e-6f;
        for (size_t i = 0U; i < k; ++i) {
            const float a = fabsf(wr[i]);
            if (a > max_abs) max_abs = a;
        }
        float s = max_abs / 1.5f;
        if (s < 1e-6f) s = 1e-6f;
        scales[row] = s;
        unsigned char* pr = packed + row * packed_k;
        for (size_t i = 0U; i < k; ++i) {
            const float qf = wr[i] / s + 1.5f;
            int q = (int)floorf(qf + 0.5f);
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            const size_t b = i >> 2U;
            const unsigned char lane = (unsigned char)(i & 0x3U);
            pr[b] |= (unsigned char)((q & 0x3) << (lane * 2U));
        }
    }
}

static void quantize_pack_int4_rowwise(
    const float* w,
    size_t n,
    size_t k,
    unsigned char* packed,
    float* scales,
    float* zero_points
) {
    const size_t packed_k = (k + 1U) / 2U;
    memset(packed, 0, n * packed_k * sizeof(unsigned char));
    for (size_t row = 0U; row < n; ++row) {
        const float* wr = w + row * k;
        float max_abs = 1e-6f;
        for (size_t i = 0U; i < k; ++i) {
            const float a = fabsf(wr[i]);
            if (a > max_abs) max_abs = a;
        }
        float s = max_abs / 7.0f;
        if (s < 1e-6f) s = 1e-6f;
        scales[row] = s;
        zero_points[row] = 0.0f;
        unsigned char* pr = packed + row * packed_k;
        for (size_t i = 0U; i < k; ++i) {
            const float qf = wr[i] / s + 8.0f;
            int q = (int)floorf(qf + 0.5f);
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            const size_t b = i >> 1U;
            if (i & 1U) {
                pr[b] |= (unsigned char)((q & 0xF) << 4U);
            } else {
                pr[b] |= (unsigned char)(q & 0xF);
            }
        }
    }
}

static float mse(const float* a, const float* b, size_t count) {
    double s = 0.0;
    for (size_t i = 0U; i < count; ++i) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)(s / (double)(count > 0U ? count : 1U));
}

int main(void) {
    const size_t m = env_size_or_default("VSPEC_LANEA_M", 1U);
    const size_t k = env_size_or_default("VSPEC_LANEA_K", 2048U);
    const size_t n = env_size_or_default("VSPEC_LANEA_N", 4096U);
    const size_t count = m * n;
    const size_t packed_k2 = (k + 3U) / 4U;
    const size_t packed_k4 = (k + 1U) / 2U;
    const float hot_ratio = env_float_or_default("VSPEC_LANEA_HOT_RATIO", 0.02f);
    const size_t hot_count = (size_t)(n * hot_ratio);
    const int warmup = (int)env_size_or_default("VSPEC_LANEA_WARMUP", 12U);
    const int iters = (int)env_size_or_default("VSPEC_LANEA_ITERS", 80U);

    srand(7);
    ensure_env_default("VSPEC_INT2_COMPUTE_MODE", "dequant-cublas");

    float* input = (float*)malloc(m * k * sizeof(float));
    float* weight_f = (float*)malloc(n * k * sizeof(float));
    unsigned char* w_int2 = (unsigned char*)malloc(n * packed_k2 * sizeof(unsigned char));
    float* s_int2 = (float*)malloc(n * sizeof(float));
    unsigned char* w_int4 = (unsigned char*)malloc(n * packed_k4 * sizeof(unsigned char));
    float* s_int4 = (float*)malloc(n * sizeof(float));
    float* zp_int4 = (float*)malloc(n * sizeof(float));
    float* out_int4 = (float*)malloc(count * sizeof(float));
    float* out_draft = (float*)malloc(count * sizeof(float));
    float* out_hybrid = (float*)malloc(count * sizeof(float));
    float* col_score = (float*)malloc(n * sizeof(float));
    uint32_t* hot_indices = (uint32_t*)malloc((hot_count > 0U ? hot_count : 1U) * sizeof(uint32_t));
    uint8_t* selected = (uint8_t*)malloc(n * sizeof(uint8_t));

    if (!input || !weight_f || !w_int2 || !s_int2 || !w_int4 || !s_int4 || !zp_int4 ||
        !out_int4 || !out_draft || !out_hybrid || !col_score || !hot_indices || !selected) {
        printf("[anf_phase_a_laneA_split] status=fail alloc\n");
        return 2;
    }

    for (size_t i = 0U; i < m * k; ++i) input[i] = rand_uniform();
    for (size_t i = 0U; i < n * k; ++i) weight_f[i] = rand_uniform();

    quantize_pack_int2_rowwise(weight_f, n, k, w_int2, s_int2);
    quantize_pack_int4_rowwise(weight_f, n, k, w_int4, s_int4, zp_int4);

    if (!vspec_cuda_fused_linear_int4_bridge(input, w_int4, s_int4, zp_int4, m, k, n, 1U, out_int4)) {
        printf("[anf_phase_a_laneA_split] status=fail int4 baseline bridge\n");
        return 3;
    }

    if (!vspec_cuda_fused_linear_hybrid_bridge(
            input, w_int2, s_int2, NULL, NULL, NULL, m, k, n, NULL, 0U, 1U, out_draft)) {
        printf("[anf_phase_a_laneA_split] status=fail int2 draft bridge\n");
        return 4;
    }

    for (size_t c = 0U; c < n; ++c) {
        selected[c] = 0U;
        double s = 0.0;
        for (size_t r = 0U; r < m; ++r) {
            s += fabs((double)out_draft[r * n + c]);
        }
        col_score[c] = (float)(s / (double)m);
    }

    for (size_t i = 0U; i < hot_count; ++i) {
        size_t best = 0U;
        float best_score = -1e30f;
        for (size_t c = 0U; c < n; ++c) {
            if (selected[c]) continue;
            if (col_score[c] > best_score) {
                best_score = col_score[c];
                best = c;
            }
        }
        selected[best] = 1U;
        hot_indices[i] = (uint32_t)best;
    }

    for (int i = 0; i < warmup; ++i) {
        float h2d = 0.0f;
        float k_ms = 0.0f;
        float d2h = 0.0f;
        float total = 0.0f;
        (void)vspec_cuda_fused_linear_hybrid_profile_bridge(
            input, w_int2, s_int2, w_int4, s_int4, zp_int4,
            m, k, n, hot_indices, hot_count, 1U, out_hybrid,
            &h2d, &k_ms, &d2h, &total);
    }

    double int4_cpu_acc = 0.0;
    double hybrid_cpu_acc = 0.0;
    double gpu_h2d_acc = 0.0;
    double gpu_kernel_acc = 0.0;
    double gpu_d2h_acc = 0.0;
    double gpu_total_acc = 0.0;

    for (int i = 0; i < iters; ++i) {
        const double int4_begin = now_ms();
        (void)vspec_cuda_fused_linear_int4_bridge(input, w_int4, s_int4, zp_int4, m, k, n, 1U, out_int4);
        int4_cpu_acc += (now_ms() - int4_begin);

        float h2d = 0.0f;
        float k_ms = 0.0f;
        float d2h = 0.0f;
        float total = 0.0f;
        const double hybrid_begin = now_ms();
        if (!vspec_cuda_fused_linear_hybrid_profile_bridge(
                input, w_int2, s_int2, w_int4, s_int4, zp_int4,
                m, k, n, hot_indices, hot_count, 1U, out_hybrid,
                &h2d, &k_ms, &d2h, &total)) {
            printf("[anf_phase_a_laneA_split] status=fail hybrid profile bridge\n");
            return 5;
        }
        hybrid_cpu_acc += (now_ms() - hybrid_begin);
        gpu_h2d_acc += (double)h2d;
        gpu_kernel_acc += (double)k_ms;
        gpu_d2h_acc += (double)d2h;
        gpu_total_acc += (double)total;
    }

    {
        const float mse_draft = mse(out_draft, out_int4, count);
        const float mse_hybrid = mse(out_hybrid, out_int4, count);
        const float improve = (mse_draft > 1e-12f) ? (mse_draft - mse_hybrid) / mse_draft : 0.0f;
        const double int4_avg = int4_cpu_acc / (double)iters;
        const double hybrid_cpu_avg = hybrid_cpu_acc / (double)iters;
        const double h2d_avg = gpu_h2d_acc / (double)iters;
        const double kernel_avg = gpu_kernel_acc / (double)iters;
        const double d2h_avg = gpu_d2h_acc / (double)iters;
        const double gpu_total_avg = gpu_total_acc / (double)iters;
        const double cpu_overhead_avg = hybrid_cpu_avg - gpu_total_avg;
        const double speedup = (hybrid_cpu_avg > 1e-12) ? (int4_avg / hybrid_cpu_avg) : 0.0;
        const double uplift_pct = (speedup - 1.0) * 100.0;

        printf("[anf_phase_a_laneA_split] shape m=%zu k=%zu n=%zu hot_ratio=%.3f hot_count=%zu\n", m, k, n, hot_ratio, hot_count);
        printf("[anf_phase_a_laneA_split] int4_cpu_avg_ms=%.4f hybrid_cpu_avg_ms=%.4f\n", int4_avg, hybrid_cpu_avg);
        printf("[anf_phase_a_laneA_split] hybrid_speedup_vs_int4=%.4fx uplift_pct=%.2f\n", speedup, uplift_pct);
        printf("[anf_phase_a_laneA_split] hybrid_gpu_h2d_ms=%.4f hybrid_gpu_kernel_ms=%.4f hybrid_gpu_d2h_ms=%.4f hybrid_gpu_total_ms=%.4f\n",
               h2d_avg, kernel_avg, d2h_avg, gpu_total_avg);
        printf("[anf_phase_a_laneA_split] hybrid_cpu_overhead_ms=%.4f\n", cpu_overhead_avg);
        printf("[anf_phase_a_laneA_split] mse_draft_vs_int4=%.8f mse_hybrid_vs_int4=%.8f improve=%.4f\n",
               mse_draft, mse_hybrid, improve);
        printf("[anf_phase_a_laneA_split] status=pass\n");
    }

    free(input);
    free(weight_f);
    free(w_int2);
    free(s_int2);
    free(w_int4);
    free(s_int4);
    free(zp_int4);
    free(out_int4);
    free(out_draft);
    free(out_hybrid);
    free(col_score);
    free(hot_indices);
    free(selected);
    return 0;
}
