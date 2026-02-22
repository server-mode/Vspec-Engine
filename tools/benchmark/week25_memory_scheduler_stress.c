#include <stdio.h>
#include <string.h>

#include <stdint.h>

#include "vspec/memory/memory_metrics.h"
#include "vspec/memory/vram_scheduler.h"
#include "vspec/runtime/mixed_bit_policy.h"
#include "vspec/runtime/mixed_bit_runtime.h"

typedef struct Week25ProfileConfig {
    const char* name;
    size_t total_vram_bytes;
    size_t weight_bytes;
    size_t activation_bytes;
    size_t kv_token_bytes_q4;
    size_t kv_max_tokens;
    size_t evict_tokens;
    float pressure_high;
    float pressure_critical;
    uint8_t downshift_step;
    float frag_increment;
} Week25ProfileConfig;

typedef struct Week25ProfileResult {
    int oom;
    size_t peak_used;
    size_t tokens_kept;
    size_t evictions;
    size_t downshift_events;
} Week25ProfileResult;

static int run_profile(const Week25ProfileConfig* cfg, Week25ProfileResult* out) {
    if (!cfg || !out) {
        return 0;
    }

    VspecMixedBitPolicy policy;
    vspec_mixed_bit_policy_default(&policy);
    policy.attention_bits = 4;
    policy.mlp_bits = 4;
    policy.embed_bits = 8;
    policy.min_bits = 2;
    policy.max_bits = 8;
    policy.downshift_step = cfg->downshift_step;
    policy.pressure_high = cfg->pressure_high;
    policy.pressure_critical = cfg->pressure_critical;

    VspecVramBudget budget;
    vspec_vram_budget_init(&budget, cfg->total_vram_bytes);

    VspecMemoryMetrics metrics;
    vspec_memory_metrics_reset(&metrics);
    metrics.weight_bytes = cfg->weight_bytes;
    metrics.activation_bytes = cfg->activation_bytes;

    if (!vspec_vram_try_reserve(&budget, metrics.weight_bytes + metrics.activation_bytes)) {
        memset(out, 0, sizeof(*out));
        out->oom = 1;
        return 1;
    }

    VspecMixedBitRuntime runtime;
    vspec_mixed_bit_runtime_init(&runtime);

    int oom = 0;
    size_t evictions = 0U;
    size_t downshift_events = 0U;
    size_t tokens_kept = 0U;
    size_t peak_used = budget.used_bytes;
    float kv_fragmentation = 0.05f;

    for (size_t token = 0U; token < cfg->kv_max_tokens; ++token) {
        float pressure = (budget.total_bytes > 0U)
            ? ((float)budget.used_bytes / (float)budget.total_bytes)
            : 0.0f;

        VspecMixedBitPressureProfile profile;
        profile.vram_pressure = pressure;
        profile.kv_pressure = (float)tokens_kept / (float)cfg->kv_max_tokens;
        profile.kv_fragmentation = kv_fragmentation;
        profile.kv_active_tokens = tokens_kept;
        profile.kv_max_tokens = cfg->kv_max_tokens;

        uint8_t bits = vspec_mixed_bit_select_bits_realtime(
            &runtime,
            &policy,
            (uint32_t)(token % 64U),
            VSPEC_LAYER_ATTENTION,
            NULL,
            0U,
            &metrics,
            &budget,
            &profile
        );

        if (bits < 4U) {
            downshift_events += 1U;
        }

        size_t kv_bytes = cfg->kv_token_bytes_q4;
        if (bits > 0U) {
            kv_bytes = (cfg->kv_token_bytes_q4 * (size_t)bits + 3U) / 4U;
        }

        if (!vspec_vram_try_reserve(&budget, kv_bytes)) {
            size_t releasable = cfg->evict_tokens * cfg->kv_token_bytes_q4;
            vspec_vram_release(&budget, releasable);
            if (tokens_kept > cfg->evict_tokens) {
                tokens_kept -= cfg->evict_tokens;
            } else {
                tokens_kept = 0U;
            }
            evictions += 1U;
            kv_fragmentation *= 0.7f;

            if (!vspec_vram_try_reserve(&budget, kv_bytes)) {
                oom = 1;
                break;
            }
        }

        tokens_kept += 1U;
        metrics.kv_bytes = budget.used_bytes - (metrics.weight_bytes + metrics.activation_bytes);

        if (budget.used_bytes > peak_used) {
            peak_used = budget.used_bytes;
        }

        if ((token % 48U) == 0U && token > 0U) {
            kv_fragmentation += cfg->frag_increment;
            if (kv_fragmentation > 0.60f) {
                kv_fragmentation = 0.60f;
            }
        }
    }

    out->oom = oom;
    out->peak_used = peak_used;
    out->tokens_kept = tokens_kept;
    out->evictions = evictions;
    out->downshift_events = downshift_events;
    return 1;
}

int main(void) {
    const Week25ProfileConfig profiles[] = {
        {"balanced",   (size_t)2048U * 1024U * 1024U, (size_t)680U * 1024U * 1024U, (size_t)80U * 1024U * 1024U, 420U * 1024U, 4096U, 96U,  0.80f, 0.92f, 1U, 0.008f},
        {"aggressive", (size_t)1536U * 1024U * 1024U, (size_t)700U * 1024U * 1024U, (size_t)80U * 1024U * 1024U, 480U * 1024U, 4096U, 128U, 0.78f, 0.90f, 1U, 0.010f},
        {"extreme",    (size_t)1280U * 1024U * 1024U, (size_t)710U * 1024U * 1024U, (size_t)96U * 1024U * 1024U, 520U * 1024U, 4096U, 160U, 0.74f, 0.86f, 2U, 0.012f},
    };

    const size_t profile_count = sizeof(profiles) / sizeof(profiles[0]);
    size_t pass_count = 0U;

    printf("{\n");
    printf("  \"week\": 25,\n");
    printf("  \"profiles\": [\n");

    for (size_t i = 0U; i < profile_count; ++i) {
        Week25ProfileResult result;
        memset(&result, 0, sizeof(result));
        (void)run_profile(&profiles[i], &result);

        int kpi_no_oom = (result.oom == 0);
        int kpi_downshift = (result.downshift_events > 0U);
        int pass = (kpi_no_oom && kpi_downshift);
        if (pass) {
            pass_count += 1U;
        }

        float peak_pressure = (profiles[i].total_vram_bytes > 0U)
            ? ((float)result.peak_used / (float)profiles[i].total_vram_bytes)
            : 0.0f;

        printf("    {\n");
        printf("      \"name\": \"%s\",\n", profiles[i].name);
        printf("      \"budget_total_bytes\": %zu,\n", profiles[i].total_vram_bytes);
        printf("      \"peak_used_bytes\": %zu,\n", result.peak_used);
        printf("      \"peak_pressure\": %.6f,\n", peak_pressure);
        printf("      \"tokens_kept\": %zu,\n", result.tokens_kept);
        printf("      \"evictions\": %zu,\n", result.evictions);
        printf("      \"downshift_events\": %zu,\n", result.downshift_events);
        printf("      \"kpi_no_oom\": %s,\n", kpi_no_oom ? "true" : "false");
        printf("      \"kpi_downshift_works\": %s,\n", kpi_downshift ? "true" : "false");
        printf("      \"kpi_week25_pass\": %s\n", pass ? "true" : "false");
        if (i + 1U < profile_count) {
            printf("    },\n");
        } else {
            printf("    }\n");
        }
    }

    printf("  ],\n");
    printf("  \"profiles_passed\": %zu,\n", pass_count);
    printf("  \"profiles_total\": %zu,\n", profile_count);
    printf("  \"kpi_week25_pass\": %s\n", (pass_count == profile_count) ? "true" : "false");
    printf("}\n");

    return (pass_count == profile_count) ? 0 : 2;
}
