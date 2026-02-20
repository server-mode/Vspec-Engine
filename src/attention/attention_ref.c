#include <math.h>
#include <stdlib.h>

#include "vspec/attention/attention.h"

void vspec_attention_ref_single_query(
    const float* query,
    const VspecKVCache* cache,
    float* out
) {
    if (!query || !cache || !out || cache->current_tokens == 0U) {
        return;
    }

    const size_t heads = cache->num_heads;
    const size_t d = cache->head_dim;
    const size_t tmax = cache->current_tokens;
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);

    for (size_t h = 0; h < heads; ++h) {
        const float* qh = query + h * d;
        float* oh = out + h * d;

        for (size_t x = 0; x < d; ++x) {
            oh[x] = 0.0f;
        }

        float* scores = (float*)malloc(tmax * sizeof(float));
        if (!scores) {
            return;
        }

        float max_score = -1e30f;
        for (size_t t = 0; t < tmax; ++t) {
            const float* kh = vspec_kv_cache_key_at(cache, t, h);
            float dot = 0.0f;
            for (size_t x = 0; x < d; ++x) {
                dot += qh[x] * kh[x];
            }
            scores[t] = dot * inv_sqrt_d;
            if (scores[t] > max_score) {
                max_score = scores[t];
            }
        }

        float denom = 0.0f;
        for (size_t t = 0; t < tmax; ++t) {
            scores[t] = expf(scores[t] - max_score);
            denom += scores[t];
        }

        if (denom > 0.0f) {
            const float inv_denom = 1.0f / denom;
            for (size_t t = 0; t < tmax; ++t) {
                const float w = scores[t] * inv_denom;
                const float* vh = vspec_kv_cache_value_at(cache, t, h);
                for (size_t x = 0; x < d; ++x) {
                    oh[x] += w * vh[x];
                }
            }
        }

        free(scores);
    }
}
