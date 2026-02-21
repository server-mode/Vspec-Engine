#include <math.h>
#include <stdlib.h>

#include "vspec/model/qwen_ops.h"

void vspec_rmsnorm_f32(const float* input, const float* weight, size_t dim, float eps, float* output) {
    if (!input || !weight || !output || dim == 0U) {
        return;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double v = input[i];
        sum_sq += v * v;
    }

    const double mean = sum_sq / (double)dim;
    const float scale = 1.0f / (float)sqrt(mean + (double)eps);

    for (size_t i = 0; i < dim; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

void vspec_silu_inplace(float* data, size_t count) {
    if (!data || count == 0U) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        const float x = data[i];
        data[i] = x / (1.0f + expf(-x));
    }
}

void vspec_mul_inplace(float* data, const float* other, size_t count) {
    if (!data || !other || count == 0U) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        data[i] *= other[i];
    }
}

void vspec_linear_f32(
    const float* input,
    size_t m,
    size_t k,
    const float* weight,
    size_t n,
    const float* bias,
    float* output
) {
    if (!input || !weight || !output || m == 0U || k == 0U || n == 0U) {
        return;
    }

    for (size_t row = 0; row < m; ++row) {
        const float* in_row = input + row * k;
        float* out_row = output + row * n;
        for (size_t col = 0; col < n; ++col) {
            double acc = 0.0;
            const float* w_col = weight + col * k;
            for (size_t i = 0; i < k; ++i) {
                acc += (double)in_row[i] * (double)w_col[i];
            }
            if (bias) {
                acc += bias[col];
            }
            out_row[col] = (float)acc;
        }
    }
}

void vspec_qwen_mlp_ref(
    const float* input,
    size_t m,
    size_t k,
    const float* w1,
    const float* w2,
    const float* w3,
    float* output
) {
    if (!input || !w1 || !w2 || !w3 || !output || m == 0U || k == 0U) {
        return;
    }

    const size_t hidden = k * 4;
    float* gate = (float*)malloc(sizeof(float) * m * hidden);
    float* up = (float*)malloc(sizeof(float) * m * hidden);
    if (!gate || !up) {
        free(gate);
        free(up);
        return;
    }

    vspec_linear_f32(input, m, k, w1, hidden, NULL, gate);
    vspec_linear_f32(input, m, k, w3, hidden, NULL, up);

    vspec_silu_inplace(gate, m * hidden);
    vspec_mul_inplace(gate, up, m * hidden);

    vspec_linear_f32(gate, m, hidden, w2, k, NULL, output);

    free(gate);
    free(up);
}
