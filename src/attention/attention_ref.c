#include <math.h>
#include <stdlib.h>

#include "vspec/attention/attention.h"
#include "vspec/runtime/three_bit_runtime_modules.h"

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
    Vspec3BitAttentionManager manager;
    vspec_3bit_attention_manager_default(&manager);

    for (size_t h = 0; h < heads; ++h) {
        const float* qh = query + h * d;
        float* oh = out + h * d;

        for (size_t x = 0; x < d; ++x) {
            oh[x] = 0.0f;
        }

        float* scores = (float*)malloc(tmax * sizeof(float));
        float* qh_clamped = (float*)malloc(d * sizeof(float));
        float* qh_denoised = (float*)malloc(d * sizeof(float));
        float* kh_clamped = (float*)malloc(d * sizeof(float));
        float* kh_denoised = (float*)malloc(d * sizeof(float));
        Vspec3BitAccumState* out_acc = (Vspec3BitAccumState*)malloc(d * sizeof(Vspec3BitAccumState));
        if (!scores || !qh_clamped || !qh_denoised || !kh_clamped || !kh_denoised || !out_acc) {
            free(scores);
            free(qh_clamped);
            free(qh_denoised);
            free(kh_clamped);
            free(kh_denoised);
            free(out_acc);
            return;
        }

        for (size_t x = 0; x < d; ++x) {
            vspec_3bit_accum_reset(&out_acc[x]);
        }
        vspec_3bit_dynamic_clamp_std(qh, d, manager.noise.activation_clamp_alpha, qh_clamped);
        vspec_3bit_noise_reduce_vector(&manager.noise, qh_clamped, d, qh_denoised);

        float max_score = -1e30f;
        for (size_t t = 0; t < tmax; ++t) {
            const float* kh = vspec_kv_cache_key_at(cache, t, h);
            vspec_3bit_dynamic_clamp_std(kh, d, manager.noise.activation_clamp_alpha, kh_clamped);
            vspec_3bit_noise_reduce_vector(&manager.noise, kh_clamped, d, kh_denoised);
            scores[t] = vspec_3bit_attention_qk_score(&manager, qh_denoised, kh_denoised, d, inv_sqrt_d);
            if (scores[t] > max_score) {
                max_score = scores[t];
            }
        }

        (void)max_score;
        vspec_3bit_softmax_apply(&manager.softmax, scores, tmax, scores);

        {
            for (size_t t = 0; t < tmax; ++t) {
                const float w = scores[t];
                const float* vh = vspec_kv_cache_value_at(cache, t, h);
                for (size_t x = 0; x < d; ++x) {
                    vspec_3bit_accum_add(&manager.accum, &out_acc[x], w * vh[x]);
                }
            }
        }

        for (size_t x = 0; x < d; ++x) {
            oh[x] = vspec_3bit_accum_value(&out_acc[x]);
        }

        free(scores);
        free(qh_clamped);
        free(qh_denoised);
        free(kh_clamped);
        free(kh_denoised);
        free(out_acc);
    }
}
