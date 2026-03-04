#include "vspec/runtime/runtime_behavior_monitor.h"

#include <string.h>

static float vspec_clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}
    static VspecRuntimeBehaviorSeverity vspec_downgrade_severity(
        VspecRuntimeBehaviorSeverity severity,
        uint32_t levels
    ) {
        while (levels > 0U && severity > VSPEC_RUNTIME_SEVERITY_NONE) {
            severity = (VspecRuntimeBehaviorSeverity)((int)severity - 1);
            --levels;
        }
        return severity;
    }

static VspecRuntimeBehaviorSeverity vspec_merge_severity(
    VspecRuntimeBehaviorSeverity current,
    VspecRuntimeBehaviorSeverity next
) {
    return (next > current) ? next : current;
}

void vspec_runtime_behavior_monitor_init(
    VspecRuntimeBehaviorMonitor* monitor,
    const VspecRuntimeHwConfig* hw_config,
    int using_gpu_backend
) {
    if (!monitor) {
        return;
    }

    (void)memset(monitor, 0, sizeof(*monitor));

    if (hw_config) {
        monitor->target_gpu_utilization = vspec_clamp01(hw_config->target_gpu_utilization);
        monitor->max_vram_utilization = vspec_clamp01(hw_config->max_vram_utilization);
        monitor->max_effective_bits = (float)hw_config->lowbit_target_bits;
    } else {
        monitor->target_gpu_utilization = 0.95f;
        monitor->max_vram_utilization = 0.95f;
        monitor->max_effective_bits = 3.0f;
    }

    monitor->latest.integrity_pass = 1;
    monitor->latest.using_gpu_backend = using_gpu_backend ? 1 : 0;
    monitor->latest.workload_scale = 1.0f;
    monitor->latest.residual_rms = 0.0f;
    monitor->latest.attention_entropy_collapse = 0.0f;
    monitor->latest.activation_norm_drift = 0.0f;

    if (!monitor->latest.using_gpu_backend) {
        monitor->issue_mask |= VSPEC_RUNTIME_ISSUE_CPU_FALLBACK;
        monitor->severity = VSPEC_RUNTIME_SEVERITY_HIGH;
    }
}

void vspec_runtime_behavior_monitor_update(
    VspecRuntimeBehaviorMonitor* monitor,
    const VspecRuntimeBehaviorSnapshot* snapshot
) {
    if (!monitor || !snapshot) {
        return;
    }

    monitor->total_updates += 1U;
    monitor->latest = *snapshot;
    monitor->latest.observed_gpu_utilization = vspec_clamp01(monitor->latest.observed_gpu_utilization);
    monitor->latest.observed_vram_utilization = vspec_clamp01(monitor->latest.observed_vram_utilization);
    monitor->latest.workload_scale = vspec_clamp01(monitor->latest.workload_scale);
    monitor->latest.attention_entropy_collapse = vspec_clamp01(monitor->latest.attention_entropy_collapse);
    if (monitor->latest.residual_rms < 0.0f) {
        monitor->latest.residual_rms = 0.0f;
    }
    if (monitor->latest.activation_norm_drift < 0.0f) {
        monitor->latest.activation_norm_drift = 0.0f;
    }

    uint32_t issues_this_update = 0U;
    VspecRuntimeBehaviorSeverity severity_this_update = VSPEC_RUNTIME_SEVERITY_NONE;

    if (!monitor->latest.using_gpu_backend) {
        issues_this_update |= VSPEC_RUNTIME_ISSUE_CPU_FALLBACK;
        severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_HIGH);
    }

    if (monitor->latest.using_gpu_backend) {
            const float workload_factor = 0.35f + (0.65f * monitor->latest.workload_scale);
            const float gpu_gap = (monitor->target_gpu_utilization - monitor->latest.observed_gpu_utilization) * workload_factor;
        if (gpu_gap > 0.0f) {
            issues_this_update |= VSPEC_RUNTIME_ISSUE_GPU_UNDER_TARGET;
            if (gpu_gap >= 0.25f) {
                severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_CRITICAL);
            } else if (gpu_gap >= 0.12f) {
                severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_HIGH);
            } else if (gpu_gap >= 0.06f) {
                severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_MEDIUM);
            } else {
                severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_LOW);
            }

                if (monitor->latest.workload_scale < 0.20f) {
                    severity_this_update = vspec_downgrade_severity(severity_this_update, 2U);
                } else if (monitor->latest.workload_scale < 0.40f) {
                    severity_this_update = vspec_downgrade_severity(severity_this_update, 1U);
                }
        }
    }

    if (monitor->latest.observed_vram_utilization > monitor->max_vram_utilization) {
        const float over = monitor->latest.observed_vram_utilization - monitor->max_vram_utilization;
        issues_this_update |= VSPEC_RUNTIME_ISSUE_VRAM_OVER_TARGET;
        if (over >= 0.08f) {
            severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_CRITICAL);
        } else if (over >= 0.04f) {
            severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_HIGH);
        } else {
            severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_MEDIUM);
        }
    }

    if (monitor->latest.observed_effective_bits > monitor->max_effective_bits) {
        const float over_bits = monitor->latest.observed_effective_bits - monitor->max_effective_bits;
        issues_this_update |= VSPEC_RUNTIME_ISSUE_BITS_OVER_TARGET;
        if (over_bits >= 1.0f) {
            severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_CRITICAL);
        } else {
            severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_HIGH);
        }
    }

    if (!monitor->latest.integrity_pass) {
        issues_this_update |= VSPEC_RUNTIME_ISSUE_INTEGRITY_FAIL;
        severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_HIGH);
    }

    if (monitor->latest.residual_rms >= 1.35f ||
        monitor->latest.activation_norm_drift >= 0.30f ||
        monitor->latest.attention_entropy_collapse >= 0.65f) {
        issues_this_update |= VSPEC_RUNTIME_ISSUE_QUALITY_DRIFT;
        severity_this_update = vspec_merge_severity(severity_this_update, VSPEC_RUNTIME_SEVERITY_HIGH);
    }

    if (issues_this_update != 0U) {
        monitor->breach_updates += 1U;
        monitor->issue_mask |= issues_this_update;
    }

    monitor->severity = vspec_merge_severity(monitor->severity, severity_this_update);
}

void vspec_runtime_behavior_monitor_report(
    const VspecRuntimeBehaviorMonitor* monitor,
    VspecRuntimeBehaviorReport* report
) {
    if (!monitor || !report) {
        return;
    }

    report->total_updates = monitor->total_updates;
    report->breach_updates = monitor->breach_updates;
    report->issue_mask = monitor->issue_mask;
    report->severity = monitor->severity;

    report->target_gpu_utilization = monitor->target_gpu_utilization;
    report->max_vram_utilization = monitor->max_vram_utilization;
    report->max_effective_bits = monitor->max_effective_bits;

    report->observed_gpu_utilization = monitor->latest.observed_gpu_utilization;
    report->observed_vram_utilization = monitor->latest.observed_vram_utilization;
    report->observed_effective_bits = monitor->latest.observed_effective_bits;
    report->residual_rms = monitor->latest.residual_rms;
    report->attention_entropy_collapse = monitor->latest.attention_entropy_collapse;
    report->activation_norm_drift = monitor->latest.activation_norm_drift;
    report->integrity_pass = monitor->latest.integrity_pass;
    report->using_gpu_backend = monitor->latest.using_gpu_backend;
    report->workload_scale = monitor->latest.workload_scale;
}