#include "vspec/runtime/adaptive/runtime_controller.h"

#include <string.h>

static uint8_t clamp_bits(uint8_t bits, uint8_t min_bits, uint8_t max_bits) {
    if (bits < min_bits) {
        return min_bits;
    }
    if (bits > max_bits) {
        return max_bits;
    }
    return bits;
}

void vspec_runtime_controller_config_default(VspecRuntimeControllerConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->pressure_high = 0.82f;
    cfg->pressure_critical = 0.93f;
    cfg->latency_budget_ms = 35.0f;
    cfg->entropy_low = 2.2f;
    cfg->entropy_high = 5.8f;
    cfg->min_bits = 2U;
    cfg->max_bits = 4U;
}

void vspec_runtime_controller_init(VspecRuntimeController* ctrl, const VspecRuntimeControllerConfig* cfg) {
    if (!ctrl) {
        return;
    }
    (void)memset(ctrl, 0, sizeof(*ctrl));
    if (cfg) {
        ctrl->cfg = *cfg;
    } else {
        vspec_runtime_controller_config_default(&ctrl->cfg);
    }
    ctrl->last_decision.target_bits = ctrl->cfg.max_bits;
    ctrl->last_decision.confidence = 1.0f;
}

void vspec_runtime_controller_observe(VspecRuntimeController* ctrl, const VspecRuntimeAdaptiveTelemetry* telemetry) {
    if (!ctrl || !telemetry) {
        return;
    }
    ctrl->last = *telemetry;
    ctrl->ticks += 1U;
}

VspecRuntimeAdaptiveDecision vspec_runtime_controller_decide(VspecRuntimeController* ctrl) {
    VspecRuntimeAdaptiveDecision out;
    (void)memset(&out, 0, sizeof(out));

    if (!ctrl) {
        out.target_bits = 4U;
        out.confidence = 0.0f;
        return out;
    }

    const VspecRuntimeAdaptiveTelemetry* t = &ctrl->last;
    out.target_bits = ctrl->cfg.max_bits;
    out.confidence = 0.8f;

    if (t->vram_pressure >= ctrl->cfg.pressure_critical) {
        out.target_bits = ctrl->cfg.min_bits;
        out.enable_skip_compute = 1U;
        out.reduce_attention_depth = 1U;
        out.enable_kv_compression = 1U;
        out.confidence = 0.95f;
    } else if (t->vram_pressure >= ctrl->cfg.pressure_high) {
        out.target_bits = (uint8_t)(ctrl->cfg.max_bits > ctrl->cfg.min_bits ? (ctrl->cfg.max_bits - 1U) : ctrl->cfg.max_bits);
        out.enable_kv_compression = 1U;
        out.confidence = 0.88f;
    }

    if (t->latency_ms > ctrl->cfg.latency_budget_ms) {
        out.reduce_attention_depth = 1U;
        if (out.target_bits > ctrl->cfg.min_bits) {
            out.target_bits -= 1U;
        }
    }

    if (t->token_entropy < ctrl->cfg.entropy_low && out.target_bits > ctrl->cfg.min_bits) {
        out.target_bits -= 1U;
        out.enable_skip_compute = 1U;
    }

    if (t->token_entropy > ctrl->cfg.entropy_high || t->quality_drift > 0.30f) {
        out.target_bits = ctrl->cfg.max_bits;
        out.enable_skip_compute = 0U;
        out.reduce_attention_depth = 0U;
        out.enable_kv_compression = 0U;
        out.confidence = 0.92f;
    }

    out.target_bits = clamp_bits(out.target_bits, ctrl->cfg.min_bits, ctrl->cfg.max_bits);
    ctrl->last_decision = out;
    return out;
}
