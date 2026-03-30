#include <stdio.h>
#include <stdint.h>

#include "vspec/runtime/runtime.h"

#if defined(_WIN32)
#include <stdlib.h>
static void vspec_set_env(const char* key, const char* value) {
    (void)_putenv_s(key, value);
}
#else
#include <stdlib.h>
static void vspec_set_env(const char* key, const char* value) {
    (void)setenv(key, value, 1);
}
#endif

int main(void) {
    const float token0[16] = {
        0.10f, 0.20f, 0.30f, 0.40f,
        1.30f, 0.50f, 1.20f, 0.60f,
        0.10f, 0.20f, 0.30f, 0.40f,
        1.10f, 0.50f, 0.60f, 0.70f
    };
    const float token1[16] = {
        0.10f, 0.20f, 0.30f, 0.40f,
        1.29f, 0.50f, 1.20f, 0.60f,
        0.10f, 0.20f, 0.30f, 0.40f,
        1.09f, 0.50f, 0.60f, 0.70f
    };
    const float token2[16] = {
        0.10f, 0.20f, 0.30f, 0.40f,
        1.31f, 0.50f, 1.20f, 0.60f,
        0.10f, 0.20f, 0.30f, 0.40f,
        1.08f, 0.50f, 0.60f, 0.70f
    };

    vspec_set_env("VSPEC_ENABLE_ANF", "1");
    vspec_set_env("VSPEC_ANF_MODE", "shadow");
    vspec_set_env("VSPEC_ANF_TCC_ENABLE", "1");
    vspec_set_env("VSPEC_ANF_MAX_HOT_RATIO", "0.25");
    vspec_set_env("VSPEC_ANF_MIN_HOT_NEURONS", "2");
    vspec_set_env("VSPEC_ANF_MAX_HOT_NEURONS", "8");
    vspec_set_env("VSPEC_ANF_ACTIVATION_THRESHOLD", "0.85");

    vspec_runtime_init_default();

    if (!vspec_runtime_anf_available()) {
        printf("[anf_phase_b_tcc_smoke] anf_available=0 (compile-time disabled)\n");
        printf("[anf_phase_b_tcc_smoke] status=pass\n");
        return 0;
    }

    for (size_t i = 0U; i < 16U; ++i) {
        vspec_runtime_anf_observe_token_activations(token0, 16U);
        vspec_runtime_anf_observe_token_activations(token1, 16U);
        vspec_runtime_anf_observe_token_activations(token2, 16U);
    }

    {
        VspecRuntimeBehaviorReport report;
        vspec_runtime_behavior_report(&report);

        printf("[anf_phase_b_tcc_smoke] mode=%d cache_updates=%u skip_last=%.4f skip_avg=%.4f changed_last=%.4f confidence=%.4f\n",
            report.anf_mode,
            (unsigned)report.anf_cache_updates,
            report.anf_skip_ratio_last,
            report.anf_skip_ratio_avg,
            report.anf_changed_ratio_last,
            report.anf_pattern_confidence);

        if (report.anf_cache_updates == 0U) {
            printf("[anf_phase_b_tcc_smoke] status=fail (cache did not update)\n");
            return 1;
        }
        if (report.anf_skip_ratio_avg < 0.0f || report.anf_skip_ratio_avg > 1.0f) {
            printf("[anf_phase_b_tcc_smoke] status=fail (invalid skip ratio)\n");
            return 1;
        }
    }

    printf("[anf_phase_b_tcc_smoke] status=pass\n");
    return 0;
}
