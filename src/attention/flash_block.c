#include <math.h>
#include <stdlib.h>

#include "vspec/attention/flash_block.h"

void vspec_flash_attention_block_ref(
    const float* q,
    const float* k,
    const float* v,
    size_t tokens,
    size_t head_dim,
    size_t block_tokens,
    float* out
) {
    if (!q || !k || !v || !out || tokens == 0U || head_dim == 0U || block_tokens == 0U) {
        return;
    }

    for (size_t d = 0; d < head_dim; ++d) {
        out[d] = 0.0f;
    }

    const float inv_sqrt = 1.0f / sqrtf((float)head_dim);

    for (size_t base = 0; base < tokens; base += block_tokens) {
        const size_t block_end = (base + block_tokens < tokens) ? (base + block_tokens) : tokens;
        const size_t block_len = block_end - base;

        float* scores = (float*)malloc(block_len * sizeof(float));
        if (!scores) {
            return;
        }

        float max_score = -1e30f;
        for (size_t t = 0; t < block_len; ++t) {
            const float* kt = k + (base + t) * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += q[d] * kt[d];
            }
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) {
                max_score = scores[t];
            }
        }

        float denom = 0.0f;
        for (size_t t = 0; t < block_len; ++t) {
            scores[t] = expf(scores[t] - max_score);
            denom += scores[t];
        }

        if (denom > 0.0f) {
            const float inv_denom = 1.0f / denom;
            for (size_t t = 0; t < block_len; ++t) {
                const float w = scores[t] * inv_denom;
                const float* vt = v + (base + t) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[d] += w * vt[d];
                }
            }
        }

        free(scores);
    }
}
