#ifndef VSPEC_RUNTIME_ERROR_ESTIMATOR_H
#define VSPEC_RUNTIME_ERROR_ESTIMATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VspecErrorEstimatorConfig {
    float ema_decay;
    float contamination_drift_weight;
    float contamination_entropy_weight;
    float contamination_residual_weight;
} VspecErrorEstimatorConfig;

typedef struct VspecErrorEstimator {
    VspecErrorEstimatorConfig config;
    uint32_t updates;
    float wave_score_last;
    float wave_score_ema;
    float contamination_last;
    float contamination_ema;
} VspecErrorEstimator;

typedef struct VspecErrorEstimatorReport {
    uint32_t updates;
    float wave_score_last;
    float wave_score_ema;
    float contamination_last;
    float contamination_ema;
} VspecErrorEstimatorReport;

void vspec_error_estimator_config_default(VspecErrorEstimatorConfig* cfg);
void vspec_error_estimator_init(VspecErrorEstimator* estimator, const VspecErrorEstimatorConfig* cfg);
void vspec_error_estimator_reset(VspecErrorEstimator* estimator);
void vspec_error_estimator_observe(
    VspecErrorEstimator* estimator,
    float residual_rms,
    float attention_entropy_collapse,
    float activation_norm_drift
);
void vspec_error_estimator_report(const VspecErrorEstimator* estimator, VspecErrorEstimatorReport* report);

#ifdef __cplusplus
}
#endif

#endif
