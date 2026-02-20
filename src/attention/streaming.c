#include <math.h>
#include <stddef.h>

#include "vspec/attention/streaming.h"

void vspec_attention_streaming_ref(
    const float* q,
    const float* k,
    const float* v,
    size_t tokens,
    size_t head_dim,
    size_t chunk,
    float* out
) {
    if (!q || !k || !v || !out || tokens == 0U || head_dim == 0U || chunk == 0U) {
        return;
    }

    for (size_t d = 0; d < head_dim; ++d) {
        out[d] = 0.0f;
    }

    const float inv_sqrt = 1.0f / sqrtf((float)head_dim);

    float max_score = -1e30f;
    for (size_t t = 0; t < tokens; ++t) {
        const float* kt = k + t * head_dim;
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            dot += q[d] * kt[d];
        }
        const float score = dot * inv_sqrt;
        if (score > max_score) {
            max_score = score;
        }
    }

    float denom = 0.0f;
    for (size_t base = 0; base < tokens; base += chunk) {
        const size_t end = (base + chunk < tokens) ? (base + chunk) : tokens;
        for (size_t t = base; t < end; ++t) {
            const float* kt = k + t * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += q[d] * kt[d];
            }
            const float score = expf(dot * inv_sqrt - max_score);
            denom += score;

            const float* vt = v + t * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                out[d] += score * vt[d];
            }
        }
    }

    if (denom > 0.0f) {
        const float inv = 1.0f / denom;
        for (size_t d = 0; d < head_dim; ++d) {
            out[d] *= inv;
        }
    }
}
