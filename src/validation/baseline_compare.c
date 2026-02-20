#include "vspec/validation/baseline_compare.h"
#include "vspec/validation/perplexity.h"

VspecBaselineCompare vspec_baseline_compare(
    const float* baseline_logits,
    const float* test_logits,
    size_t vocab,
    size_t count
) {
    VspecBaselineCompare out;
    out.perplexity_baseline = vspec_perplexity_from_logits(baseline_logits, vocab, count);
    out.perplexity_test = vspec_perplexity_from_logits(test_logits, vocab, count);
    out.drift = vspec_drift_analyze(baseline_logits, test_logits, vocab * count);
    return out;
}
