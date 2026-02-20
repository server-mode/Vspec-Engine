#include <stdio.h>
#include <stdlib.h>

#include "vspec/validation/perplexity.h"
#include "vspec/validation/drift_analyzer.h"
#include "vspec/validation/quant_sensitivity.h"
#include "vspec/validation/baseline_compare.h"

int main(void) {
    const size_t vocab = 4;
    const size_t steps = 3;

    float baseline[12] = {
        1.0f, 0.5f, -0.2f, 0.1f,
        0.9f, 0.3f, -0.1f, 0.0f,
        1.2f, 0.4f, -0.3f, 0.2f
    };

    float test[12] = {
        0.9f, 0.4f, -0.1f, 0.1f,
        0.8f, 0.2f,  0.0f, 0.0f,
        1.1f, 0.3f, -0.2f, 0.1f
    };

    VspecBaselineCompare comp = vspec_baseline_compare(baseline, test, vocab, steps);

    printf("perplexity baseline=%.4f test=%.4f\n", comp.perplexity_baseline, comp.perplexity_test);
    printf("drift mean_abs=%.6f max_abs=%.6f mean_rel=%.6f\n", comp.drift.mean_abs, comp.drift.max_abs, comp.drift.mean_rel);

    VspecQuantSensitivity qs = vspec_quant_sensitivity(baseline, test, vocab * steps, 4);
    printf("quant sensitivity bits=%u mean_abs=%.6f max_abs=%.6f\n", (unsigned)qs.bits, qs.mean_abs, qs.max_abs);

    float nll[3] = {1.2f, 1.1f, 1.0f};
    printf("perplexity from nll=%.4f\n", vspec_perplexity_from_nll(nll, 3));

    return 0;
}
