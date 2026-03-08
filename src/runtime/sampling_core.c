#include "vspec/runtime/sampling_core.h"

#include <float.h>
#include <math.h>

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

    for (size_t i = 0U; i < count; ++i) {
        const double weight = exp((double)scores[i] - (double)max_score);
        if (isfinite(weight) && weight > 0.0) {
            total += weight;
        }
    }
    if (!(total > 0.0) || !isfinite(total)) {
        return 1;
    }

    r = ((double)(random_bits & 0x1fffffffffffffULL)) / (double)0x20000000000000ULL;
    for (size_t i = 0U; i < count; ++i) {
        const double weight = exp((double)scores[i] - (double)max_score);
        if (!isfinite(weight) || weight <= 0.0) {
            continue;
        }
        acc += weight / total;
        if (r <= acc) {
            *out_token_id = token_ids[i];
            return 1;
        }
    }

    return 1;
}