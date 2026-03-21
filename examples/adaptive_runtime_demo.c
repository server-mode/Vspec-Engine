#include <stdio.h>

#include "vspec/runtime/runtime.h"

int main(void) {
    vspec_runtime_init_default();

    VspecRuntimeAdaptiveTelemetry t;
    t.token_entropy = 1.9f;
    t.attention_entropy_collapse = 0.15f;
    t.latency_ms = 42.0f;
    t.vram_pressure = 0.91f;
    t.gpu_utilization = 0.77f;
    t.quality_drift = 0.10f;

    vspec_runtime_adaptive_observe(&t);
    VspecRuntimeAdaptiveDecision decision = vspec_runtime_adaptive_decide();

    VspecTokenScheduleDecision token_decision = vspec_runtime_schedule_token("instruction", t.token_entropy);

    VspecPrecisionRouteHint route_hint;
    route_hint.layer_type = VSPEC_LAYER_MLP;
    route_hint.token_importance = token_decision.importance;
    route_hint.vram_pressure = t.vram_pressure;
    route_hint.quality_drift = t.quality_drift;
    route_hint.controller_target_bits = decision.target_bits;

    unsigned int bits = (unsigned int)vspec_runtime_route_precision(&route_hint);

    VspecMemoryPolicyInput mem_in;
    mem_in.vram_pressure = t.vram_pressure;
    mem_in.token_importance = token_decision.importance;
    mem_in.active_tokens = 128U;
    mem_in.kv_bytes = 64U * 1024U * 1024U;

    VspecKvPolicyAction kv_action = vspec_runtime_memory_decide(&mem_in);

    printf("adaptive target_bits=%u skip=%u depth_reduce=%u kv_comp=%u confidence=%.2f\n",
        (unsigned int)decision.target_bits,
        (unsigned int)decision.enable_skip_compute,
        (unsigned int)decision.reduce_attention_depth,
        (unsigned int)decision.enable_kv_compression,
        (double)decision.confidence);
    printf("token tier=%u importance=%.3f depth_hint=%u precision_hint=%u\n",
        (unsigned int)token_decision.tier,
        (double)token_decision.importance,
        (unsigned int)token_decision.attention_depth_hint,
        (unsigned int)token_decision.precision_hint_bits);
    printf("precision routed bits=%u kv_action=%u\n", bits, (unsigned int)kv_action);

    return 0;
}
