#include "vspec/runtime/sampling_core.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

typedef struct VspecSamplingPair {
    size_t idx;
    double prob;
} VspecSamplingPair;

static int _sampling_pair_desc_prob(const void* lhs, const void* rhs) {
    const VspecSamplingPair* a = (const VspecSamplingPair*)lhs;
    const VspecSamplingPair* b = (const VspecSamplingPair*)rhs;
    if (a->prob > b->prob) return -1;
    if (a->prob < b->prob) return 1;
    return 0;
}

static int _env_u32_or_default(const char* name, int fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    int parsed = atoi(value);
    if (parsed < 0) {
        parsed = 0;
    }
    return parsed;
}

static float _env_float_or_default(const char* name, float fallback) {
    const char* value = getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    const float parsed = (float)atof(value);
    if (!isfinite(parsed)) {
        return fallback;
    }
    return parsed;
}

int vspec_sampling_select_candidate(
    const int* token_ids,
    const float* scores,
    size_t count,
    int greedy,
    uint64_t random_bits,
    int* out_token_id
) {
    float max_score = -FLT_MAX;
    double total = 0.0;
    double acc = 0.0;
    double r = 0.0;
    size_t best_idx = 0U;

    if (!token_ids || !scores || !out_token_id || count == 0U) {
        return 0;
    }

    const float temperature = fmaxf(0.05f, _env_float_or_default("VSPEC_SAMPLING_TEMPERATURE", 1.0f));
    const int top_k_env = _env_u32_or_default("VSPEC_SAMPLING_TOP_K", 0);
    const float top_p = _env_float_or_default("VSPEC_SAMPLING_TOP_P", 1.0f);
    const float min_p = _env_float_or_default("VSPEC_SAMPLING_MIN_P", 0.0f);

    for (size_t i = 0U; i < count; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
            best_idx = i;
        }
    }

    *out_token_id = token_ids[best_idx];
    if (greedy || count == 1U) {
        return 1;
    }

    VspecSamplingPair* pairs = (VspecSamplingPair*)malloc(count * sizeof(VspecSamplingPair));
    if (!pairs) {
        return 1;
    }

    const double inv_temp = 1.0 / (double)temperature;
    for (size_t i = 0U; i < count; ++i) {
        const double scaled = ((double)scores[i] - (double)max_score) * inv_temp;
        const double weight = exp(scaled);
        if (isfinite(weight) && weight > 0.0) {
            total += weight;
        }
        pairs[i].idx = i;
        pairs[i].prob = (isfinite(weight) && weight > 0.0) ? weight : 0.0;
    }
    if (!(total > 0.0) || !isfinite(total)) {
        free(pairs);
        return 1;
    }

    for (size_t i = 0U; i < count; ++i) {
        pairs[i].prob /= total;
    }

    qsort(pairs, count, sizeof(VspecSamplingPair), _sampling_pair_desc_prob);

    size_t keep = count;
    if (top_k_env > 0 && (size_t)top_k_env < keep) {
        keep = (size_t)top_k_env;
    }

    if (top_p < 1.0f) {
        double cumulative = 0.0;
        size_t keep_top_p = 0U;
        for (size_t i = 0U; i < keep; ++i) {
            cumulative += pairs[i].prob;
            keep_top_p = i + 1U;
            if (cumulative >= (double)fmaxf(top_p, 0.01f)) {
                break;
            }
        }
        if (keep_top_p > 0U && keep_top_p < keep) {
            keep = keep_top_p;
        }
    }

    if (min_p > 0.0f && keep > 1U) {
        const double p_ref = pairs[0].prob;
        size_t keep_min_p = 1U;
        for (size_t i = 1U; i < keep; ++i) {
            if (pairs[i].prob >= p_ref * (double)min_p) {
                keep_min_p = i + 1U;
            }
        }
        keep = keep_min_p;
    }

    if (keep == 0U) {
        keep = 1U;
    }

    double kept_total = 0.0;
    for (size_t i = 0U; i < keep; ++i) {
        kept_total += pairs[i].prob;
    }
    if (!(kept_total > 0.0) || !isfinite(kept_total)) {
        free(pairs);
        return 1;
    }

    r = ((double)(random_bits & 0x1fffffffffffffULL)) / (double)0x20000000000000ULL;
    for (size_t i = 0U; i < keep; ++i) {
        acc += pairs[i].prob / kept_total;
        if (r <= acc) {
            *out_token_id = token_ids[pairs[i].idx];
            free(pairs);
            return 1;
        }
    }

    *out_token_id = token_ids[pairs[0].idx];
    free(pairs);

    return 1;
}