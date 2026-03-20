#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/attention/flash_stub.h"

void vspec_flash_attention_stub(const float* q, const float* k, const float* v, size_t tokens, size_t head_dim, float* out) {
    if (!q || !k || !v || !out || tokens == 0U || head_dim == 0U) {
        return;
    }

    const float inv_sqrt_dim = 1.0f / sqrtf((float)head_dim);
    float* scores = (float*)malloc(tokens * sizeof(float));
    if (!scores) {
        for (size_t i = 0U; i < head_dim; ++i) {
            out[i] = 0.0f;
        }
        return;
    }

    float max_score = -1.0e30f;
    for (size_t t = 0U; t < tokens; ++t) {
        const float* kt = k + t * head_dim;
        float acc = 0.0f;
        for (size_t d = 0U; d < head_dim; ++d) {
            acc += q[d] * kt[d];
        }
        scores[t] = acc * inv_sqrt_dim;
        if (scores[t] > max_score) {
            max_score = scores[t];
        }
    }

    float sum = 0.0f;
    for (size_t t = 0U; t < tokens; ++t) {
        scores[t] = expf(scores[t] - max_score);
        sum += scores[t];
    }
    if (sum < 1e-12f) {
        sum = 1e-12f;
    }

    for (size_t d = 0U; d < head_dim; ++d) {
        out[d] = 0.0f;
    }
    for (size_t t = 0U; t < tokens; ++t) {
        const float p = scores[t] / sum;
        const float* vt = v + t * head_dim;
        for (size_t d = 0U; d < head_dim; ++d) {
            out[d] += p * vt[d];
        }
    }

    free(scores);
}
