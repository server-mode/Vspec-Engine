#include "vspec/runtime/precision_wave.h"

#include <string.h>

static float vspec_clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

void vspec_precision_wave_config_default(VspecPrecisionWaveConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->cascade_limit = 3U;
    cfg->escalate_wave_threshold = 0.62f;
    cfg->deescalate_wave_threshold = 0.28f;
    cfg->escalate_contamination_threshold = 0.58f;
    cfg->deescalate_contamination_threshold = 0.24f;
    cfg->cooldown_updates = 2U;
}

void vspec_precision_wave_init(VspecPrecisionWave* wave, const VspecPrecisionWaveConfig* cfg) {
    VspecPrecisionWaveConfig local_cfg;
    if (!wave) {
        return;
    }
    vspec_precision_wave_config_default(&local_cfg);
    (void)memset(wave, 0, sizeof(*wave));
    wave->config = cfg ? *cfg : local_cfg;
    wave->config.escalate_wave_threshold = vspec_clamp01(wave->config.escalate_wave_threshold);
    wave->config.deescalate_wave_threshold = vspec_clamp01(wave->config.deescalate_wave_threshold);
    wave->config.escalate_contamination_threshold = vspec_clamp01(wave->config.escalate_contamination_threshold);
    wave->config.deescalate_contamination_threshold = vspec_clamp01(wave->config.deescalate_contamination_threshold);

    if (wave->config.deescalate_wave_threshold > wave->config.escalate_wave_threshold) {
        wave->config.deescalate_wave_threshold = wave->config.escalate_wave_threshold;
    }
    if (wave->config.deescalate_contamination_threshold > wave->config.escalate_contamination_threshold) {
        wave->config.deescalate_contamination_threshold = wave->config.escalate_contamination_threshold;
    }
}

void vspec_precision_wave_reset(VspecPrecisionWave* wave) {
    if (!wave) {
        return;
    }
    wave->updates = 0U;
    wave->cascade_depth = 0U;
    wave->cascade_depth_max = 0U;
    wave->escalations = 0U;
    wave->deescalations = 0U;
    wave->cooldown_left = 0U;
    wave->anomaly = 0;
}

void vspec_precision_wave_observe(
    VspecPrecisionWave* wave,
    float wave_score,
    float contamination_score
) {
    int escalate;
    int deescalate;

    if (!wave) {
        return;
    }

    wave_score = vspec_clamp01(wave_score);
    contamination_score = vspec_clamp01(contamination_score);
    wave->updates += 1U;
    wave->anomaly = 0;

    if (wave->config.cascade_limit == 0U) {
        wave->cascade_depth = 0U;
        wave->cooldown_left = 0U;
        return;
    }

    escalate =
        (wave_score >= wave->config.escalate_wave_threshold) ||
        (contamination_score >= wave->config.escalate_contamination_threshold);
    deescalate =
        (wave_score <= wave->config.deescalate_wave_threshold) &&
        (contamination_score <= wave->config.deescalate_contamination_threshold);

    if (wave->cooldown_left > 0U) {
        wave->cooldown_left -= 1U;
        deescalate = 0;
    }

    if (escalate) {
        if (wave->cascade_depth < wave->config.cascade_limit) {
            wave->cascade_depth += 1U;
            wave->escalations += 1U;
            wave->cooldown_left = wave->config.cooldown_updates;
            if (wave->cascade_depth > wave->cascade_depth_max) {
                wave->cascade_depth_max = wave->cascade_depth;
            }
        } else {
            wave->anomaly = 1;
        }
        return;
    }

    if (deescalate && wave->cascade_depth > 0U) {
        wave->cascade_depth -= 1U;
        wave->deescalations += 1U;
    }
}

void vspec_precision_wave_report(const VspecPrecisionWave* wave, VspecPrecisionWaveReport* report) {
    if (!report) {
        return;
    }
    if (!wave) {
        (void)memset(report, 0, sizeof(*report));
        return;
    }

    report->updates = wave->updates;
    report->cascade_depth = wave->cascade_depth;
    report->cascade_depth_max = wave->cascade_depth_max;
    report->escalations = wave->escalations;
    report->deescalations = wave->deescalations;
    report->anomaly = wave->anomaly;
}
