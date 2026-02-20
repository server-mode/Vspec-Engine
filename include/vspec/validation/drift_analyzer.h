#ifndef VSPEC_VALIDATION_DRIFT_ANALYZER_H
#define VSPEC_VALIDATION_DRIFT_ANALYZER_H

#include <stddef.h>

typedef struct VspecDriftStats {
    float mean_abs;
    float max_abs;
    float mean_rel;
} VspecDriftStats;

VspecDriftStats vspec_drift_analyze(const float* baseline, const float* test, size_t count);

#endif
