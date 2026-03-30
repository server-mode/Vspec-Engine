#ifndef VSPEC_RUNTIME_BEHAVIOR_MONITOR_H
#define VSPEC_RUNTIME_BEHAVIOR_MONITOR_H

#include <stdint.h>

#include "vspec/runtime/hw_performance_manager.h"

typedef enum VspecRuntimeBehaviorIssue {
    VSPEC_RUNTIME_ISSUE_NONE = 0,
    VSPEC_RUNTIME_ISSUE_GPU_UNDER_TARGET = 1 << 0,
    VSPEC_RUNTIME_ISSUE_VRAM_OVER_TARGET = 1 << 1,
    VSPEC_RUNTIME_ISSUE_BITS_OVER_TARGET = 1 << 2,
    VSPEC_RUNTIME_ISSUE_INTEGRITY_FAIL = 1 << 3,
    VSPEC_RUNTIME_ISSUE_CPU_FALLBACK = 1 << 4,
    VSPEC_RUNTIME_ISSUE_QUALITY_DRIFT = 1 << 5,
    VSPEC_RUNTIME_ISSUE_CASCADE_ANOMALY = 1 << 6,
    VSPEC_RUNTIME_ISSUE_ANF_FORCED_FALLBACK = 1 << 7,
} VspecRuntimeBehaviorIssue;

typedef enum VspecRuntimeBehaviorSeverity {
    VSPEC_RUNTIME_SEVERITY_NONE = 0,
    VSPEC_RUNTIME_SEVERITY_LOW = 1,
    VSPEC_RUNTIME_SEVERITY_MEDIUM = 2,
    VSPEC_RUNTIME_SEVERITY_HIGH = 3,
    VSPEC_RUNTIME_SEVERITY_CRITICAL = 4,
} VspecRuntimeBehaviorSeverity;

typedef struct VspecRuntimeBehaviorSnapshot {
    float observed_gpu_utilization;
    float observed_vram_utilization;
    float observed_effective_bits;
    float workload_scale;
    float residual_rms;
    float attention_entropy_collapse;
    float activation_norm_drift;
    int integrity_pass;
    int using_gpu_backend;
} VspecRuntimeBehaviorSnapshot;

typedef struct VspecRuntimeBehaviorMonitor {
    float target_gpu_utilization;
    float max_vram_utilization;
    float max_effective_bits;

    VspecRuntimeBehaviorSnapshot latest;

    uint32_t total_updates;
    uint32_t breach_updates;
    uint32_t issue_mask;
    VspecRuntimeBehaviorSeverity severity;
} VspecRuntimeBehaviorMonitor;

typedef struct VspecRuntimeBehaviorReport {
    uint32_t total_updates;
    uint32_t breach_updates;
    uint32_t issue_mask;
    VspecRuntimeBehaviorSeverity severity;

    float target_gpu_utilization;
    float max_vram_utilization;
    float max_effective_bits;

    float observed_gpu_utilization;
    float observed_vram_utilization;
    float observed_effective_bits;
    float workload_scale;
    float residual_rms;
    float attention_entropy_collapse;
    float activation_norm_drift;
    int integrity_pass;
    int using_gpu_backend;

    int anf_available;
    int anf_mode;
    float anf_hot_ratio;
    uint32_t anf_hot_neurons;
    uint32_t anf_tokens_observed;
    float anf_hot_ratio_avg;
    float anf_hot_ratio_p95;
    uint32_t anf_cache_updates;
    float anf_skip_ratio_last;
    float anf_skip_ratio_avg;
    float anf_changed_ratio_last;
    float anf_pattern_confidence;
    float anf_error_wave_last;
    float anf_error_wave_avg;
    float anf_contamination_last;
    float anf_contamination_avg;
    uint32_t anf_cascade_depth;
    uint32_t anf_cascade_depth_max;
    uint32_t anf_cascade_escalations;
    uint32_t anf_cascade_deescalations;
    int anf_cascade_anomaly;
    uint32_t anf_quality_guard_fail_streak;
    uint32_t anf_deescalate_count;
    uint32_t anf_forced_fallback_count;
    uint32_t anf_silent_stop_count;
    int anf_fallback_triggered;
} VspecRuntimeBehaviorReport;

void vspec_runtime_behavior_monitor_init(
    VspecRuntimeBehaviorMonitor* monitor,
    const VspecRuntimeHwConfig* hw_config,
    int using_gpu_backend
);

void vspec_runtime_behavior_monitor_update(
    VspecRuntimeBehaviorMonitor* monitor,
    const VspecRuntimeBehaviorSnapshot* snapshot
);

void vspec_runtime_behavior_monitor_report(
    const VspecRuntimeBehaviorMonitor* monitor,
    VspecRuntimeBehaviorReport* report
);

#endif