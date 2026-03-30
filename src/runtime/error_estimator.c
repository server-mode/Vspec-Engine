#include "vspec/runtime/error_estimator.h"

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

void vspec_error_estimator_config_default(VspecErrorEstimatorConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->ema_decay = 0.85f;
    cfg->contamination_drift_weight = 0.50f;
    cfg->contamination_entropy_weight = 0.25f;
    cfg->contamination_residual_weight = 0.25f;
}

void vspec_error_estimator_init(VspecErrorEstimator* estimator, const VspecErrorEstimatorConfig* cfg) {
    VspecErrorEstimatorConfig local_cfg;
    if (!estimator) {
        return;
    }
    vspec_error_estimator_config_default(&local_cfg);
    (void)memset(estimator, 0, sizeof(*estimator));
    estimator->config = cfg ? *cfg : local_cfg;
    estimator->config.ema_decay = vspec_clamp01(estimator->config.ema_decay);
    if (estimator->config.ema_decay < 0.01f) {
        estimator->config.ema_decay = 0.01f;
    }
    estimator->config.contamination_drift_weight = vspec_clamp01(estimator->config.contamination_drift_weight);
    estimator->config.contamination_entropy_weight = vspec_clamp01(estimator->config.contamination_entropy_weight);
    estimator->config.contamination_residual_weight = vspec_clamp01(estimator->config.contamination_residual_weight);
}

void vspec_error_estimator_reset(VspecErrorEstimator* estimator) {
    if (!estimator) {
        return;
    }
    estimator->updates = 0U;
    estimator->wave_score_last = 0.0f;
    estimator->wave_score_ema = 0.0f;
    estimator->contamination_last = 0.0f;
    estimator->contamination_ema = 0.0f;
}

void vspec_error_estimator_observe(
    VspecErrorEstimator* estimator,
    float residual_rms,
    float attention_entropy_collapse,
    float activation_norm_drift
) {
    float residual_norm;
    float wave_score;
    float contamination;
    float alpha;
    float total_w;

    if (!estimator) {
        return;
    }

    if (residual_rms < 0.0f) {
        residual_rms = 0.0f;
    }
    if (attention_entropy_collapse < 0.0f) {
        attention_entropy_collapse = 0.0f;
    }
    if (attention_entropy_collapse > 1.0f) {
        attention_entropy_collapse = 1.0f;
    }
    if (activation_norm_drift < 0.0f) {
        activation_norm_drift = 0.0f;
    }

    residual_norm = residual_rms / 2.0f;
    if (residual_norm > 1.0f) {
        residual_norm = 1.0f;
    }

    wave_score = 0.50f * residual_norm +
                 0.30f * attention_entropy_collapse +
                 0.20f * ((activation_norm_drift > 1.0f) ? 1.0f : activation_norm_drift);
    wave_score = vspec_clamp01(wave_score);

    total_w = estimator->config.contamination_drift_weight +
              estimator->config.contamination_entropy_weight +
              estimator->config.contamination_residual_weight;
    if (total_w < 1e-6f) {
        total_w = 1.0f;
    }

    contamination =
        (estimator->config.contamination_drift_weight * ((activation_norm_drift > 1.0f) ? 1.0f : activation_norm_drift)) +
        (estimator->config.contamination_entropy_weight * attention_entropy_collapse) +
        (estimator->config.contamination_residual_weight * residual_norm);
    contamination = vspec_clamp01(contamination / total_w);

    estimator->updates += 1U;
    estimator->wave_score_last = wave_score;
    estimator->contamination_last = contamination;

    alpha = estimator->config.ema_decay;
    if (estimator->updates == 1U) {
        estimator->wave_score_ema = wave_score;
        estimator->contamination_ema = contamination;
    } else {
        estimator->wave_score_ema = alpha * estimator->wave_score_ema + (1.0f - alpha) * wave_score;
        estimator->contamination_ema = alpha * estimator->contamination_ema + (1.0f - alpha) * contamination;
    }
}

void vspec_error_estimator_report(const VspecErrorEstimator* estimator, VspecErrorEstimatorReport* report) {
    if (!report) {
        return;
    }
    if (!estimator) {
        (void)memset(report, 0, sizeof(*report));
        return;
    }

    report->updates = estimator->updates;
    report->wave_score_last = estimator->wave_score_last;
    report->wave_score_ema = estimator->wave_score_ema;
    report->contamination_last = estimator->contamination_last;
    report->contamination_ema = estimator->contamination_ema;
}
