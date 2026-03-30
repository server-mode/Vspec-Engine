#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#define VSPEC_BENCH_API __declspec(dllimport)
#else
#include <time.h>
#define VSPEC_BENCH_API
#endif

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

static void set_env_local(const char* key, const char* value) {
#if defined(_WIN32)
    (void)_putenv_s(key, value ? value : "");
#else
    (void)setenv(key, value ? value : "", 1);
#endif
}

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

static float mse(const float* a, const float* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)(s / (double)(n > 0U ? n : 1U));
}

static float rms(const float* a, size_t n) {
    double s = 0.0;
    for (size_t i = 0U; i < n; ++i) {
        s += (double)a[i] * (double)a[i];
    }
    return (float)sqrt(s / (double)(n > 0U ? n : 1U));
}

static void quantize_pack_int2_rowwise(const float* w, size_t n, size_t k, unsigned char* packed, float* scales) {
    const size_t packed_k = (k + 3U) / 4U;
    (void)memset(packed, 0, n * packed_k * sizeof(unsigned char));
    for (size_t row = 0U; row < n; ++row) {
        const float* wr = w + row * k;
        float max_abs = 1e-6f;
        for (size_t i = 0U; i < k; ++i) {
            const float a = fabsf(wr[i]);
            if (a > max_abs) max_abs = a;
        }
        scales[row] = max_abs / 1.5f;
        if (scales[row] < 1e-6f) scales[row] = 1e-6f;
        for (size_t i = 0U; i < k; ++i) {
            float qf = wr[i] / scales[row] + 1.5f;
            int q = (int)floorf(qf + 0.5f);
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            packed[row * packed_k + (i >> 2U)] |= (unsigned char)((q & 0x3) << ((i & 0x3U) * 2U));
        }
    }
}

static void quantize_pack_int4_rowwise(const float* w, size_t n, size_t k, unsigned char* packed, float* scales, float* zp) {
    const size_t packed_k = (k + 1U) / 2U;
    (void)memset(packed, 0, n * packed_k * sizeof(unsigned char));
    for (size_t row = 0U; row < n; ++row) {
        const float* wr = w + row * k;
        float max_abs = 1e-6f;
        for (size_t i = 0U; i < k; ++i) {
            const float a = fabsf(wr[i]);
            if (a > max_abs) max_abs = a;
        }
        scales[row] = max_abs / 7.0f;
        if (scales[row] < 1e-6f) scales[row] = 1e-6f;
        zp[row] = 0.0f;
        for (size_t i = 0U; i < k; ++i) {
            float qf = wr[i] / scales[row] + 8.0f;
            int q = (int)floorf(qf + 0.5f);
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            if (i & 1U) {
                packed[row * packed_k + (i >> 1U)] |= (unsigned char)((q & 0xF) << 4U);
            } else {
                packed[row * packed_k + (i >> 1U)] |= (unsigned char)(q & 0xF);
            }
        }
    }
}

static size_t build_hot_indices(size_t n, size_t hot_count, size_t token_step, uint32_t* out) {
    const size_t shift = token_step % 7U;
    for (size_t i = 0U; i < hot_count; ++i) {
        out[i] = (uint32_t)((i * 13U + shift) % n);
    }
    return hot_count;
}

int main(void) {
    const size_t m = 1U;
    const size_t k = 4096U;
    const size_t n = 8192U;
    const size_t steps = 96U;
    const size_t hot_count = 96U;
    const size_t packed_k2 = (k + 3U) / 4U;
    const size_t packed_k4 = (k + 1U) / 2U;
    const size_t out_count = m * n;

    float* input = (float*)malloc(m * k * sizeof(float));
    float* weight_f = (float*)malloc(n * k * sizeof(float));
    unsigned char* w_int2 = (unsigned char*)malloc(n * packed_k2 * sizeof(unsigned char));
    float* s_int2 = (float*)malloc(n * sizeof(float));
    unsigned char* w_int4 = (unsigned char*)malloc(n * packed_k4 * sizeof(unsigned char));
    float* s_int4 = (float*)malloc(n * sizeof(float));
    float* zp_int4 = (float*)malloc(n * sizeof(float));
    uint32_t* hot_idx = (uint32_t*)malloc(hot_count * sizeof(uint32_t));
    float* out_full = (float*)malloc(out_count * sizeof(float));
    float* out_tcc = (float*)malloc(out_count * sizeof(float));
    float* out_full_all = (float*)malloc(steps * out_count * sizeof(float));
    float* input_base = (float*)malloc(m * k * sizeof(float));

    double full_ms_acc = 0.0;
    double tcc_ms_acc = 0.0;
    double mse_acc = 0.0;
    double rel_acc = 0.0;
    double skip_ratio_acc = 0.0;

    if (!input || !weight_f || !w_int2 || !s_int2 || !w_int4 || !s_int4 || !zp_int4 || !hot_idx || !out_full || !out_tcc || !out_full_all || !input_base) {
        printf("[anf_phase_b_cuda_laneB] status=fail alloc\n");
        return 2;
    }

    srand(11);
    for (size_t i = 0U; i < m * k; ++i) input[i] = rand_uniform();
    (void)memcpy(input_base, input, m * k * sizeof(float));
    for (size_t i = 0U; i < n * k; ++i) weight_f[i] = rand_uniform();
    quantize_pack_int2_rowwise(weight_f, n, k, w_int2, s_int2);
    quantize_pack_int4_rowwise(weight_f, n, k, w_int4, s_int4, zp_int4);

    set_env_local("VSPEC_INT2_COMPUTE_MODE", "dequant-cublas");
    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", "active");
    set_env_local("VSPEC_ANF_TCC_MAX_CHANGED_RATIO", "0.35");
    set_env_local("VSPEC_ANF_TCC_WARMUP", "4");

    set_env_local("VSPEC_ANF_TCC_ENABLE", "0");
    srand(123);
    for (size_t step = 0U; step < steps; ++step) {
        const size_t hc = build_hot_indices(n, hot_count, step, hot_idx);
        const double begin_full = now_ms();
        if (!vspec_cuda_fused_linear_hybrid_profile_bridge(
                input, w_int2, s_int2, w_int4, s_int4, zp_int4,
                m, k, n, hot_idx, hc, 1U, out_full,
                NULL, NULL, NULL, NULL)) {
            printf("[anf_phase_b_cuda_laneB] status=fail full path\n");
            return 3;
        }
        full_ms_acc += (now_ms() - begin_full);
        (void)memcpy(out_full_all + (step * out_count), out_full, out_count * sizeof(float));
        for (size_t i = 0U; i < m * k; ++i) {
            input[i] += 0.0002f * rand_uniform();
        }
    }

    (void)memcpy(input, input_base, m * k * sizeof(float));
    set_env_local("VSPEC_ANF_TCC_ENABLE", "1");
    srand(123);
    for (size_t step = 0U; step < steps; ++step) {
        const size_t hc = build_hot_indices(n, hot_count, step, hot_idx);
        const double begin_tcc = now_ms();
        if (!vspec_cuda_fused_linear_hybrid_profile_bridge(
                input, w_int2, s_int2, w_int4, s_int4, zp_int4,
                m, k, n, hot_idx, hc, 1U, out_tcc,
                NULL, NULL, NULL, NULL)) {
            printf("[anf_phase_b_cuda_laneB] status=fail tcc path\n");
            return 4;
        }
        tcc_ms_acc += (now_ms() - begin_tcc);

        {
            const float* full_ref = out_full_all + (step * out_count);
            const float e = mse(full_ref, out_tcc, out_count);
            const float base_rms = rms(full_ref, out_count);
            const float rel = (base_rms > 1e-9f) ? (float)sqrt((double)e) / base_rms : 0.0f;
            const float changed_ratio = (step <= 4U) ? 1.0f : (7.0f / (float)hot_count);
            const float skip_ratio = 1.0f - changed_ratio;
            mse_acc += (double)e;
            rel_acc += (double)rel;
            skip_ratio_acc += (double)skip_ratio;
        }

        for (size_t i = 0U; i < m * k; ++i) {
            input[i] += 0.0002f * rand_uniform();
        }
    }

    {
        const float full_avg = (float)(full_ms_acc / (double)steps);
        const float tcc_avg = (float)(tcc_ms_acc / (double)steps);
        const float speedup = (tcc_avg > 1e-9f) ? (full_avg / tcc_avg) : 0.0f;
        const float mse_avg = (float)(mse_acc / (double)steps);
        const float rel_avg = (float)(rel_acc / (double)steps);
        const float skip_avg = (float)(skip_ratio_acc / (double)steps);
        const int pass = (skip_avg >= 0.35f && rel_avg <= 0.05f && tcc_avg <= full_avg * 1.10f) ? 1 : 0;

        printf("[anf_phase_b_cuda_laneB] shape m=%zu k=%zu n=%zu steps=%zu hot_count=%zu\n", m, k, n, steps, hot_count);
        printf("[anf_phase_b_cuda_laneB] full_avg_ms=%.4f tcc_avg_ms=%.4f speedup=%.4fx\n", full_avg, tcc_avg, speedup);
        printf("[anf_phase_b_cuda_laneB] skip_ratio_avg=%.4f quality_mse=%.8f quality_rel=%.4f\n", skip_avg, mse_avg, rel_avg);
        printf("[anf_phase_b_cuda_laneB] status=%s\n", pass ? "pass" : "fail");
        free(input);
        free(input_base);
        free(weight_f);
        free(w_int2);
        free(s_int2);
        free(w_int4);
        free(s_int4);
        free(zp_int4);
        free(hot_idx);
        free(out_full);
        free(out_tcc);
        free(out_full_all);
        return pass ? 0 : 1;
    }
}
