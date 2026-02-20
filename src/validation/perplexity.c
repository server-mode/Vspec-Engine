#include <math.h>

#include "vspec/validation/perplexity.h"

float vspec_perplexity_from_nll(const float* nll, size_t count) {
    if (!nll || count == 0U) {
        return 0.0f;
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += nll[i];
    }

    const double mean = sum / (double)count;
    return (float)exp(mean);
}

float vspec_perplexity_from_logits(const float* logits, size_t vocab, size_t count) {
    if (!logits || vocab == 0U || count == 0U) {
        return 0.0f;
    }

    double total_nll = 0.0;
    for (size_t t = 0; t < count; ++t) {
        const float* row = logits + t * vocab;
        float max_logit = row[0];
        for (size_t i = 1; i < vocab; ++i) {
            if (row[i] > max_logit) max_logit = row[i];
        }

        double denom = 0.0;
        for (size_t i = 0; i < vocab; ++i) {
            denom += exp((double)(row[i] - max_logit));
        }

        const double log_sum_exp = (double)max_logit + log(denom);
        total_nll += (float)(log_sum_exp - row[0]);
    }

    const double mean_nll = total_nll / (double)count;
    return (float)exp(mean_nll);
}
