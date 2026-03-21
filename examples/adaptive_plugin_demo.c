#include <stdio.h>

#include "vspec/runtime/runtime.h"

static void on_controller(const VspecRuntimeAdaptiveTelemetry* telemetry, const VspecRuntimeAdaptiveDecision* decision) {
    printf("[plugin] controller entropy=%.2f pressure=%.2f -> bits=%u\n",
        (double)telemetry->token_entropy,
        (double)telemetry->vram_pressure,
        (unsigned int)decision->target_bits);
}

static void on_token(const char* token_text, const VspecTokenScheduleDecision* decision) {
    printf("[plugin] token=%s tier=%u importance=%.3f\n",
        token_text ? token_text : "",
        (unsigned int)decision->tier,
        (double)decision->importance);
}

int main(void) {
    vspec_runtime_init_default();

    VspecRuntimePluginHooks hooks;
    hooks.on_controller_decision = on_controller;
    hooks.on_token_scheduled = on_token;

    if (!vspec_plugin_register("demo.plugin", &hooks)) {
        printf("plugin register failed\n");
        return 1;
    }

    VspecRuntimeAdaptiveTelemetry t;
    t.token_entropy = 2.4f;
    t.attention_entropy_collapse = 0.2f;
    t.latency_ms = 24.0f;
    t.vram_pressure = 0.85f;
    t.gpu_utilization = 0.72f;
    t.quality_drift = 0.12f;

    vspec_runtime_adaptive_observe(&t);
    (void)vspec_runtime_adaptive_decide();
    (void)vspec_runtime_schedule_token("Compute", t.token_entropy);

    printf("plugin_count=%zu\n", vspec_plugin_count());
    (void)vspec_plugin_unregister("demo.plugin");
    printf("plugin_count_after=%zu\n", vspec_plugin_count());
    return 0;
}
