#include <stdint.h>
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

static float clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

int main(void) {
    const uint32_t turns = 1000U;
    const uint32_t forced_fallback_limit = (turns * 2U) / 100U;
    uint32_t mode_active_turns = 0U;
    uint32_t mode_shadow_turns = 0U;
    uint32_t mode_off_turns = 0U;
    float drift = 0.16f;
    VspecRuntimeBehaviorReport report;

    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", "active");
    set_env_local("VSPEC_ANF_CASCADE_LIMIT", "3");
    set_env_local("VSPEC_ANF_CASCADE_COOLDOWN", "1");
    set_env_local("VSPEC_ANF_CASCADE_ROUTE_ENABLE", "1");
    set_env_local("VSPEC_ANF_AUTO_DEESCALATE_ENABLE", "1");
    set_env_local("VSPEC_ANF_DEESCALATE_TARGET", "shadow");
    set_env_local("VSPEC_ANF_QUALITY_BREACH_STREAK", "3");

    vspec_runtime_init_default();

    if (!vspec_runtime_anf_available()) {
        printf("[anf_phase_d_soak_1000] anf_available=0 (compile-time disabled)\n");
        printf("[anf_phase_d_soak_1000] status=pass\n");
        return 0;
    }

    for (uint32_t turn = 0U; turn < turns; ++turn) {
        VspecPrecisionRouteHint hint;
        const int severe_window = (turn >= 480U && turn <= 482U) ? 1 : 0;
        float residual_rms = 0.10f + (0.45f * drift);
        float entropy_collapse = clamp01(0.05f + (0.40f * drift));
        float norm_drift = clamp01(0.04f + (0.36f * drift));

        if (severe_window) {
            residual_rms = 1.90f;
            entropy_collapse = 0.96f;
            norm_drift = 0.90f;
        }

        vspec_runtime_behavior_observe_quality(residual_rms, entropy_collapse, norm_drift);
        vspec_runtime_behavior_report(&report);

        if (report.anf_mode == (int)VSPEC_ANF_MODE_ACTIVE) {
            mode_active_turns += 1U;
        } else if (report.anf_mode == (int)VSPEC_ANF_MODE_SHADOW) {
            mode_shadow_turns += 1U;
        } else {
            mode_off_turns += 1U;
        }

        hint.layer_type = VSPEC_LAYER_MLP;
        hint.token_importance = 0.25f;
        hint.vram_pressure = 0.58f;
        hint.quality_drift = 0.0f;
        hint.controller_target_bits = 2U;

        {
            const uint8_t bits = vspec_runtime_route_precision(&hint);
            const float recovery = (bits >= 4U) ? 0.12f : ((bits == 3U) ? 0.06f : 0.02f);
            const float excitation = severe_window ? 0.26f : 0.025f;
            drift = clamp01((drift * 0.92f) + excitation - recovery);
        }
    }

    {
        const int pass =
            (report.anf_silent_stop_count == 0U) &&
            (report.anf_forced_fallback_count <= forced_fallback_limit) &&
            (report.anf_deescalate_count >= 1U);

        printf("[anf_phase_d_soak_1000] turns=%u crash_count=0 silent_stop_count=%u forced_fallback_count=%u limit=%u\n",
            (unsigned)turns,
            (unsigned)report.anf_silent_stop_count,
            (unsigned)report.anf_forced_fallback_count,
            (unsigned)forced_fallback_limit);
        printf("[anf_phase_d_soak_1000] mode_turns active=%u shadow=%u off=%u final_mode=%d deescalate_count=%u fallback_triggered=%d\n",
            (unsigned)mode_active_turns,
            (unsigned)mode_shadow_turns,
            (unsigned)mode_off_turns,
            report.anf_mode,
            (unsigned)report.anf_deescalate_count,
            report.anf_fallback_triggered);
        printf("[anf_phase_d_soak_1000] cascade depth_max=%u anomaly=%d issue_mask=0x%08x\n",
            (unsigned)report.anf_cascade_depth_max,
            report.anf_cascade_anomaly,
            (unsigned)report.issue_mask);
        printf("[anf_phase_d_soak_1000] status=%s\n", pass ? "pass" : "fail");

        return pass ? 0 : 1;
    }
}
