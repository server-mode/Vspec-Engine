#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vspec/runtime/runtime.h"

typedef struct AdaptiveProfileCase {
    const char* name;
    uint8_t bit_cap;
    float downgrade_trigger;
    float cache_compression_trigger;
    float pressure_bias;
    float latency_bias_ms;
} AdaptiveProfileCase;

static float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

static void run_profile(const AdaptiveProfileCase* profile, uint32_t token_count) {
    double total_latency_ms = 0.0;
    double total_vram_pressure = 0.0;
    double total_quality_drift = 0.0;
    double total_bits = 0.0;
    uint32_t skip_compute_count = 0U;
    uint32_t kv_compress_count = 0U;

    const float base_vram_mb = 12288.0f;

    vspec_runtime_set_model_precision_profile(
        7U,
        profile->bit_cap,
        1,
        profile->downgrade_trigger,
        profile->cache_compression_trigger
    );

    for (uint32_t step = 0U; step < token_count; ++step) {
        const float t = (float)step / (float)(token_count > 0U ? token_count : 1U);
        const float entropy = 1.25f + 0.55f * sinf(7.0f * t);
        const float collapse = clamp01(0.15f + 0.35f * fabsf(cosf(9.0f * t)));
        const float pressure = clamp01(profile->pressure_bias + 0.20f * fabsf(sinf(5.0f * t)));
        const float latency_ms = 3.5f + profile->latency_bias_ms + 2.0f * pressure + 0.8f * collapse;

        VspecRuntimeAdaptiveTelemetry telemetry;
        telemetry.token_entropy = entropy;
        telemetry.attention_entropy_collapse = collapse;
        telemetry.latency_ms = latency_ms;
        telemetry.vram_pressure = pressure;
        telemetry.gpu_utilization = clamp01(0.70f + 0.20f * sinf(3.0f * t));
        telemetry.quality_drift = clamp01(0.20f + 0.25f * collapse);

        vspec_runtime_adaptive_observe(&telemetry);
        VspecRuntimeAdaptiveDecision decision = vspec_runtime_adaptive_decide();
        VspecTokenScheduleDecision token_decision = vspec_runtime_schedule_token("token", telemetry.token_entropy);

        VspecPrecisionRouteHint route_hint;
        route_hint.layer_type = VSPEC_LAYER_MLP;
        route_hint.token_importance = token_decision.importance;
        route_hint.vram_pressure = telemetry.vram_pressure;
        route_hint.quality_drift = telemetry.quality_drift;
        route_hint.controller_target_bits = decision.target_bits;
        uint8_t routed_bits = vspec_runtime_route_precision(&route_hint);

        VspecMemoryPolicyInput mem_in;
        mem_in.vram_pressure = telemetry.vram_pressure;
        mem_in.token_importance = token_decision.importance;
        mem_in.active_tokens = (size_t)(step + 1U);
        mem_in.kv_bytes = (size_t)((step + 1U) * 16384U);
        VspecKvPolicyAction kv_action = vspec_runtime_memory_decide(&mem_in);

        if (decision.enable_skip_compute) {
            skip_compute_count += 1U;
        }
        if (decision.enable_kv_compression || kv_action == VSPEC_KV_POLICY_COMPRESS) {
            kv_compress_count += 1U;
        }

        {
            float bit_factor = (float)routed_bits / 8.0f;
            float effective_latency_ms = telemetry.latency_ms * (0.85f + 0.30f * bit_factor);
            float estimated_vram_mb = base_vram_mb * pressure * (0.70f + 0.35f * bit_factor);
            float quality_penalty = clamp01(
                telemetry.quality_drift
                + (float)(8U - (uint32_t)routed_bits) * 0.035f
                + (kv_action == VSPEC_KV_POLICY_RECOMPUTE ? 0.05f : 0.0f)
            );

            total_latency_ms += (double)effective_latency_ms;
            total_vram_pressure += (double)(estimated_vram_mb / base_vram_mb);
            total_quality_drift += (double)quality_penalty;
            total_bits += (double)routed_bits;
        }
    }

    if (token_count == 0U) {
        token_count = 1U;
    }

    printf(
        "[adaptive_bench] profile=%s tokens=%u avg_latency_ms=%.3f avg_vram_pressure=%.3f avg_quality_drift=%.3f avg_bits=%.2f skip_rate=%.3f kv_compress_rate=%.3f\n",
        profile->name,
        (unsigned)token_count,
        total_latency_ms / (double)token_count,
        total_vram_pressure / (double)token_count,
        total_quality_drift / (double)token_count,
        total_bits / (double)token_count,
        (double)skip_compute_count / (double)token_count,
        (double)kv_compress_count / (double)token_count
    );
}

int main(void) {
    const AdaptiveProfileCase profiles[] = {
        {"quality_guarded", 8U, 0.92f, 0.95f, 0.38f, 0.40f},
        {"balanced", 4U, 0.78f, 0.82f, 0.58f, 0.20f},
        {"latency_first", 3U, 0.64f, 0.72f, 0.74f, 0.05f},
    };

    const uint32_t token_count = 768U;
    vspec_runtime_init_default();

    printf("[adaptive_bench] benchmark=quality_vs_latency_vs_vram tokens=%u\n", (unsigned)token_count);
    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); ++i) {
        run_profile(&profiles[i], token_count);
    }
    return 0;
}
