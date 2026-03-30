#ifndef VSPEC_RUNTIME_PRECISION_WAVE_H
#define VSPEC_RUNTIME_PRECISION_WAVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VspecPrecisionWaveConfig {
    uint32_t cascade_limit;
    float escalate_wave_threshold;
    float deescalate_wave_threshold;
    float escalate_contamination_threshold;
    float deescalate_contamination_threshold;
    uint32_t cooldown_updates;
} VspecPrecisionWaveConfig;

typedef struct VspecPrecisionWave {
    VspecPrecisionWaveConfig config;
    uint32_t updates;
    uint32_t cascade_depth;
    uint32_t cascade_depth_max;
    uint32_t escalations;
    uint32_t deescalations;
    uint32_t cooldown_left;
    int anomaly;
} VspecPrecisionWave;

typedef struct VspecPrecisionWaveReport {
    uint32_t updates;
    uint32_t cascade_depth;
    uint32_t cascade_depth_max;
    uint32_t escalations;
    uint32_t deescalations;
    int anomaly;
} VspecPrecisionWaveReport;

void vspec_precision_wave_config_default(VspecPrecisionWaveConfig* cfg);
void vspec_precision_wave_init(VspecPrecisionWave* wave, const VspecPrecisionWaveConfig* cfg);
void vspec_precision_wave_reset(VspecPrecisionWave* wave);
void vspec_precision_wave_observe(
    VspecPrecisionWave* wave,
    float wave_score,
    float contamination_score
);
void vspec_precision_wave_report(const VspecPrecisionWave* wave, VspecPrecisionWaveReport* report);

#ifdef __cplusplus
}
#endif

#endif
