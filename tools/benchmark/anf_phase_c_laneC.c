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

static float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

typedef struct LaneCRunStats {
    float drift_avg;
    float drift_p95_proxy;
    float wave_avg;
    float contam_avg;
    float avg_bits;
    uint32_t cascade_depth_max;
    uint32_t escalations;
    uint32_t deescalations;
    uint32_t anomalies;
} LaneCRunStats;

static int run_case(int cascade_route_enable, LaneCRunStats* out_stats) {
    const size_t steps = 240U;
    float drift = 0.72f;
    double drift_sum = 0.0;
    double wave_sum = 0.0;
    double contam_sum = 0.0;
    double bits_sum = 0.0;
    float drift_window[32] = {0.0f};
    size_t drift_window_n = 0U;
    VspecRuntimeBehaviorReport report;

    if (!out_stats) {
        return 0;
    }

    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", "active");
    set_env_local("VSPEC_ANF_CASCADE_LIMIT", "3");
    set_env_local("VSPEC_ANF_CASCADE_COOLDOWN", "1");
    set_env_local("VSPEC_ANF_CASCADE_ROUTE_ENABLE", cascade_route_enable ? "1" : "0");

    vspec_runtime_init_default();

    if (!vspec_runtime_anf_available()) {
        out_stats->drift_avg = 0.0f;
        out_stats->drift_p95_proxy = 0.0f;
        out_stats->wave_avg = 0.0f;
        out_stats->contam_avg = 0.0f;
        out_stats->avg_bits = 0.0f;
        out_stats->cascade_depth_max = 0U;
        out_stats->escalations = 0U;
        out_stats->deescalations = 0U;
        out_stats->anomalies = 0U;
        return 1;
    }

    for (size_t i = 0U; i < steps; ++i) {
        VspecPrecisionRouteHint hint;
        const float periodic_spike = ((i % 40U) < 10U) ? 0.06f : 0.0f;
        const float residual_rms = 0.18f + 1.95f * drift;
        const float entropy_collapse = clamp01(0.12f + 0.95f * drift + ((float)(i % 7U) * 0.01f));
        const float norm_drift = clamp01(0.06f + 0.92f * drift + ((float)(i % 5U) * 0.008f));

        vspec_runtime_behavior_observe_quality(residual_rms, entropy_collapse, norm_drift);
        vspec_runtime_behavior_report(&report);

        hint.layer_type = VSPEC_LAYER_MLP;
        hint.token_importance = 0.35f;
        hint.vram_pressure = 0.55f;
        hint.quality_drift = 0.0f;
        hint.controller_target_bits = 2U;

        {
            const uint8_t bits = vspec_runtime_route_precision(&hint);
            const float recovery = (bits >= 4U) ? 0.27f : ((bits == 3U) ? 0.09f : 0.00f);
            const float excitation = 0.024f + periodic_spike;
            drift = clamp01(drift * 0.90f + excitation - recovery);

            bits_sum += (double)bits;
            drift_sum += (double)drift;
            wave_sum += (double)report.anf_error_wave_avg;
            contam_sum += (double)report.anf_contamination_avg;
            drift_window[i % 32U] = drift;
            if (drift_window_n < 32U) {
                drift_window_n += 1U;
            }
        }
    }

    {
        float p95_proxy = 0.0f;
        for (size_t i = 0U; i < drift_window_n; ++i) {
            if (drift_window[i] > p95_proxy) {
                p95_proxy = drift_window[i];
            }
        }

        out_stats->drift_avg = (float)(drift_sum / (double)steps);
        out_stats->drift_p95_proxy = p95_proxy;
        out_stats->wave_avg = (float)(wave_sum / (double)steps);
        out_stats->contam_avg = (float)(contam_sum / (double)steps);
        out_stats->avg_bits = (float)(bits_sum / (double)steps);
        out_stats->cascade_depth_max = report.anf_cascade_depth_max;
        out_stats->escalations = report.anf_cascade_escalations;
        out_stats->deescalations = report.anf_cascade_deescalations;
        out_stats->anomalies = (uint32_t)(report.anf_cascade_anomaly ? 1U : 0U);
    }

    return 1;
}

int main(void) {
    LaneCRunStats no_cascade;
    LaneCRunStats cascade;

    if (!run_case(0, &no_cascade)) {
        printf("[anf_phase_c_laneC] status=fail run no-cascade\n");
        return 2;
    }
    if (!run_case(1, &cascade)) {
        printf("[anf_phase_c_laneC] status=fail run cascade\n");
        return 3;
    }

    if (!vspec_runtime_anf_available()) {
        printf("[anf_phase_c_laneC] anf_available=0 (compile-time disabled)\n");
        printf("[anf_phase_c_laneC] status=pass\n");
        return 0;
    }

    {
        const float drift_reduction =
            (no_cascade.drift_avg > 1e-6f) ?
            ((no_cascade.drift_avg - cascade.drift_avg) / no_cascade.drift_avg) : 0.0f;
        const int pass =
            (drift_reduction >= 0.30f) &&
            (cascade.cascade_depth_max <= 3U) &&
            (cascade.anomalies == 0U);

        printf("[anf_phase_c_laneC] no_cascade drift_avg=%.4f drift_p95_proxy=%.4f avg_bits=%.3f\n",
            no_cascade.drift_avg,
            no_cascade.drift_p95_proxy,
            no_cascade.avg_bits);
        printf("[anf_phase_c_laneC] cascade drift_avg=%.4f drift_p95_proxy=%.4f avg_bits=%.3f wave_avg=%.4f contam_avg=%.4f\n",
            cascade.drift_avg,
            cascade.drift_p95_proxy,
            cascade.avg_bits,
            cascade.wave_avg,
            cascade.contam_avg);
        printf("[anf_phase_c_laneC] cascade depth_max=%u escalations=%u deescalations=%u anomaly=%u\n",
            (unsigned)cascade.cascade_depth_max,
            (unsigned)cascade.escalations,
            (unsigned)cascade.deescalations,
            (unsigned)cascade.anomalies);
        printf("[anf_phase_c_laneC] drift_reduction=%.4f\n", drift_reduction);
        printf("[anf_phase_c_laneC] status=%s\n", pass ? "pass" : "fail");

        return pass ? 0 : 1;
    }
}
