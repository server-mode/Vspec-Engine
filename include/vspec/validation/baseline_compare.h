#ifndef VSPEC_VALIDATION_BASELINE_COMPARE_H
#define VSPEC_VALIDATION_BASELINE_COMPARE_H

#include <stddef.h>
#include "vspec/validation/drift_analyzer.h"

typedef struct VspecBaselineCompare {
    VspecDriftStats drift;
    float perplexity_baseline;
    float perplexity_test;
} VspecBaselineCompare;

VspecBaselineCompare vspec_baseline_compare(
    const float* baseline_logits,
    const float* test_logits,
    size_t vocab,
    size_t count
);

#endif
