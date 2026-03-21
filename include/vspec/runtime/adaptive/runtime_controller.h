#ifndef VSPEC_RUNTIME_ADAPTIVE_RUNTIME_CONTROLLER_H
#define VSPEC_RUNTIME_ADAPTIVE_RUNTIME_CONTROLLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecRuntimeAdaptiveTelemetry {
    float token_entropy;
    float attention_entropy_collapse;
    float latency_ms;
    float vram_pressure;
    float gpu_utilization;
    float quality_drift;
} VspecRuntimeAdaptiveTelemetry;

typedef struct VspecRuntimeAdaptiveDecision {
    uint8_t target_bits;
    uint8_t enable_skip_compute;
    uint8_t reduce_attention_depth;
    uint8_t enable_kv_compression;
    float confidence;
} VspecRuntimeAdaptiveDecision;

typedef struct VspecRuntimeControllerConfig {
    float pressure_high;
    float pressure_critical;
    float latency_budget_ms;
    float entropy_low;
    float entropy_high;
    uint8_t min_bits;
    uint8_t max_bits;
} VspecRuntimeControllerConfig;

typedef struct VspecRuntimeController {
    VspecRuntimeControllerConfig cfg;
    VspecRuntimeAdaptiveTelemetry last;
    VspecRuntimeAdaptiveDecision last_decision;
    uint64_t ticks;
} VspecRuntimeController;

void vspec_runtime_controller_config_default(VspecRuntimeControllerConfig* cfg);
void vspec_runtime_controller_init(VspecRuntimeController* ctrl, const VspecRuntimeControllerConfig* cfg);
void vspec_runtime_controller_observe(VspecRuntimeController* ctrl, const VspecRuntimeAdaptiveTelemetry* telemetry);
VspecRuntimeAdaptiveDecision vspec_runtime_controller_decide(VspecRuntimeController* ctrl);

#endif
