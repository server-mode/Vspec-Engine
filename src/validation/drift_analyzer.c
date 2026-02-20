#include <math.h>

#include "vspec/validation/drift_analyzer.h"

VspecDriftStats vspec_drift_analyze(const float* baseline, const float* test, size_t count) {
    VspecDriftStats stats = {0.0f, 0.0f, 0.0f};
    if (!baseline || !test || count == 0U) {
        return stats;
    }

    double sum_abs = 0.0;
    double sum_rel = 0.0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        const float b = baseline[i];
        const float t = test[i];
        const float diff = fabsf(t - b);
        sum_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
        }

        const float denom = fabsf(b) > 1e-6f ? fabsf(b) : 1.0f;
        sum_rel += diff / denom;
    }

    stats.mean_abs = (float)(sum_abs / (double)count);
    stats.max_abs = max_abs;
    stats.mean_rel = (float)(sum_rel / (double)count);
    return stats;
}
