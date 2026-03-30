#include <stdio.h>
#include <stdlib.h>

#include "vspec/runtime/runtime.h"

#if defined(_WIN32)
static void set_env_local(const char* key, const char* value) {
    (void)_putenv_s(key, value ? value : "");
}
#else
static void set_env_local(const char* key, const char* value) {
    (void)setenv(key, value ? value : "", 1);
}
#endif

int main(void) {
    VspecRuntimeBehaviorReport report;

    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", "active");
    set_env_local("VSPEC_ANF_CASCADE_LIMIT", "3");
    set_env_local("VSPEC_ANF_CASCADE_COOLDOWN", "1");

    vspec_runtime_init_default();

    if (!vspec_runtime_anf_available()) {
        printf("[anf_phase_c_cascade_smoke] anf_available=0 (compile-time disabled)\n");
        printf("[anf_phase_c_cascade_smoke] status=pass\n");
        return 0;
    }

    for (size_t i = 0U; i < 12U; ++i) {
        vspec_runtime_behavior_observe_quality(0.18f, 0.08f, 0.06f);
    }

    for (size_t i = 0U; i < 18U; ++i) {
        vspec_runtime_behavior_observe_quality(1.85f, 0.92f, 0.76f);
    }

    for (size_t i = 0U; i < 24U; ++i) {
        vspec_runtime_behavior_observe_quality(0.12f, 0.05f, 0.03f);
    }

    vspec_runtime_behavior_report(&report);

    printf("[anf_phase_c_cascade_smoke] wave_last=%.4f wave_avg=%.4f contam_last=%.4f contam_avg=%.4f\n",
        report.anf_error_wave_last,
        report.anf_error_wave_avg,
        report.anf_contamination_last,
        report.anf_contamination_avg);
    printf("[anf_phase_c_cascade_smoke] depth=%u depth_max=%u escalations=%u deescalations=%u anomaly=%d\n",
        (unsigned)report.anf_cascade_depth,
        (unsigned)report.anf_cascade_depth_max,
        (unsigned)report.anf_cascade_escalations,
        (unsigned)report.anf_cascade_deescalations,
        report.anf_cascade_anomaly);

    if (report.anf_cascade_escalations == 0U) {
        printf("[anf_phase_c_cascade_smoke] status=fail (no escalation)\n");
        return 1;
    }
    if (report.anf_cascade_deescalations == 0U) {
        printf("[anf_phase_c_cascade_smoke] status=fail (no deescalation)\n");
        return 1;
    }
    if (report.anf_cascade_depth_max > 3U) {
        printf("[anf_phase_c_cascade_smoke] status=fail (depth exceeds limit)\n");
        return 1;
    }

    printf("[anf_phase_c_cascade_smoke] status=pass\n");
    return 0;
}
