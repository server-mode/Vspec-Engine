#include "vspec/validation/quant_sensitivity.h"
#include "vspec/validation/drift_analyzer.h"

VspecQuantSensitivity vspec_quant_sensitivity(const float* baseline, const float* quantized, size_t count, uint8_t bits) {
    VspecQuantSensitivity out;
    out.bits = bits;

    VspecDriftStats st = vspec_drift_analyze(baseline, quantized, count);
    out.mean_abs = st.mean_abs;
    out.max_abs = st.max_abs;
    return out;
}
