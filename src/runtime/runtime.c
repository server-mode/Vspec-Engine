#include "vspec/runtime/runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VspecRuntimeHwState g_hw_state;
static VspecLanguageStructureGuard g_language_guard;
static int g_language_guard_enabled = 0;
static VspecRuntimeBehaviorMonitor g_behavior_monitor;
static VspecRuntimeUltimateState g_ultimate_state;
static VspecRuntimeController g_runtime_controller;
static VspecTokenScheduler g_token_scheduler;
static VspecPrecisionRouter g_precision_router;
static VspecMemoryPolicy g_memory_policy;

static void vspec_runtime_enable_language_guard_auto(void) {
    const char* disable = getenv("VSPEC_LANGUAGE_GUARD_DISABLE");
    if (disable && (disable[0] == '1' || disable[0] == 'y' || disable[0] == 'Y' || disable[0] == 't' || disable[0] == 'T')) {
        g_language_guard_enabled = 0;
        return;
    }

    const char* strictness_env = getenv("VSPEC_LANGUAGE_GUARD_STRICTNESS");
    float strictness = 0.55f;
    if (strictness_env && strictness_env[0] != '\0') {
        strictness = (float)atof(strictness_env);
        if (strictness < 0.0f) {
            strictness = 0.0f;
        }
        if (strictness > 1.0f) {
            strictness = 1.0f;
        }
    }
    vspec_runtime_language_guard_init(NULL, strictness);
}

static void vspec_runtime_apply_hw_env_hints(const VspecRuntimeHwConfig* cfg) {
    if (!cfg || !cfg->enable_lowbit_boost) {
        return;
    }

    char bits_value[8] = {0};
    char batch_value[16] = {0};
    char stream_value[16] = {0};
    char gpu_util_value[16] = {0};
    char vram_util_value[16] = {0};
    char ultimate_value[8] = {0};
    char outlier_value[8] = {0};
    char qlora_value[8] = {0};
    char tensorcore_value[8] = {0};
    char outlier_th_value[16] = {0};
    char quality_bias_value[16] = {0};
    char qlora_rank_value[16] = {0};
    char precision_downgrade_trigger_value[16] = {0};
    char cache_compression_trigger_value[16] = {0};
    char per_model_adaptive_bit_cap_value[8] = {0};
    char runtime_3bit_value[8] = {0};
    char atn_qk_bits_value[8] = {0};
    char atn_proj_bits_value[8] = {0};
    char mlp_bits_value[8] = {0};
    char lm_head_bits_value[8] = {0};
    (void)snprintf(bits_value, sizeof(bits_value), "%u", 4U);
    (void)snprintf(batch_value, sizeof(batch_value), "%u", (unsigned)cfg->dispatch_batch_hint);
    (void)snprintf(stream_value, sizeof(stream_value), "%u", (unsigned)cfg->stream_count_hint);
    (void)snprintf(gpu_util_value, sizeof(gpu_util_value), "%.2f", cfg->target_gpu_utilization);
    (void)snprintf(vram_util_value, sizeof(vram_util_value), "%.2f", cfg->max_vram_utilization);
    (void)snprintf(ultimate_value, sizeof(ultimate_value), "%d", cfg->enable_ultimate_mode ? 1 : 0);
    (void)snprintf(outlier_value, sizeof(outlier_value), "%d", cfg->enable_outlier_aware ? 1 : 0);
    (void)snprintf(qlora_value, sizeof(qlora_value), "%d", cfg->enable_qlora_adapter ? 1 : 0);
    (void)snprintf(tensorcore_value, sizeof(tensorcore_value), "%d", cfg->prefer_tensor_core ? 1 : 0);
    (void)snprintf(outlier_th_value, sizeof(outlier_th_value), "%.3f", cfg->outlier_threshold);
    (void)snprintf(quality_bias_value, sizeof(quality_bias_value), "%.3f", cfg->quality_bias);
    (void)snprintf(qlora_rank_value, sizeof(qlora_rank_value), "%u", (unsigned)cfg->qlora_rank);
    (void)snprintf(precision_downgrade_trigger_value, sizeof(precision_downgrade_trigger_value), "%.3f", cfg->precision_downgrade_trigger);
    (void)snprintf(cache_compression_trigger_value, sizeof(cache_compression_trigger_value), "%.3f", cfg->cache_compression_trigger);
    (void)snprintf(per_model_adaptive_bit_cap_value, sizeof(per_model_adaptive_bit_cap_value), "%u", (unsigned)cfg->per_model_adaptive_bit_cap);
    (void)snprintf(runtime_3bit_value, sizeof(runtime_3bit_value), "%d", (cfg->lowbit_target_bits == 3U) ? 1 : 0);
    (void)snprintf(atn_qk_bits_value, sizeof(atn_qk_bits_value), "%u", 4U);
    (void)snprintf(atn_proj_bits_value, sizeof(atn_proj_bits_value), "%u", (unsigned)((cfg->lowbit_target_bits == 0U) ? 4U : 4U));
    (void)snprintf(mlp_bits_value, sizeof(mlp_bits_value), "%u", (unsigned)((cfg->max_vram_utilization >= cfg->precision_downgrade_trigger) ? 3U : 4U));
    (void)snprintf(lm_head_bits_value, sizeof(lm_head_bits_value), "%u", 4U);

#if defined(_WIN32)
    (void)_putenv_s("VSPEC_FUSED_BITS", bits_value);
    (void)_putenv_s("VSPEC_DISABLE_FUSED_ATTN", "0");
    (void)_putenv_s("VSPEC_GPU_BATCH_HINT", batch_value);
    (void)_putenv_s("VSPEC_GPU_STREAMS", stream_value);
    (void)_putenv_s("VSPEC_TARGET_GPU_UTIL", gpu_util_value);
    (void)_putenv_s("VSPEC_MAX_VRAM_UTIL", vram_util_value);
    (void)_putenv_s("VSPEC_ULTIMATE_ENABLE", ultimate_value);
    (void)_putenv_s("VSPEC_ULTIMATE_OUTLIER_AWARE", outlier_value);
    (void)_putenv_s("VSPEC_ULTIMATE_QLORA", qlora_value);
    (void)_putenv_s("VSPEC_ULTIMATE_TENSORCORE", tensorcore_value);
    (void)_putenv_s("VSPEC_ULTIMATE_OUTLIER_TH", outlier_th_value);
    (void)_putenv_s("VSPEC_ULTIMATE_QUALITY_BIAS", quality_bias_value);
    (void)_putenv_s("VSPEC_ULTIMATE_QLORA_RANK", qlora_rank_value);
    (void)_putenv_s("VSPEC_PRECISION_DOWNGRADE_TRIGGER", precision_downgrade_trigger_value);
    (void)_putenv_s("VSPEC_CACHE_COMPRESSION_TRIGGER", cache_compression_trigger_value);
    (void)_putenv_s("VSPEC_PER_MODEL_ADAPTIVE_BIT_CAP", per_model_adaptive_bit_cap_value);
    (void)_putenv_s("VSPEC_3BIT_RUNTIME_MODULE", runtime_3bit_value);
    (void)_putenv_s("VSPEC_3BIT_ATTN_QK_BITS", atn_qk_bits_value);
    (void)_putenv_s("VSPEC_3BIT_ATTN_PROJ_BITS", atn_proj_bits_value);
    (void)_putenv_s("VSPEC_3BIT_MLP_BITS", mlp_bits_value);
    (void)_putenv_s("VSPEC_3BIT_LM_HEAD_BITS", lm_head_bits_value);
#else
    (void)setenv("VSPEC_FUSED_BITS", bits_value, 1);
    (void)setenv("VSPEC_DISABLE_FUSED_ATTN", "0", 1);
    (void)setenv("VSPEC_GPU_BATCH_HINT", batch_value, 1);
    (void)setenv("VSPEC_GPU_STREAMS", stream_value, 1);
    (void)setenv("VSPEC_TARGET_GPU_UTIL", gpu_util_value, 1);
    (void)setenv("VSPEC_MAX_VRAM_UTIL", vram_util_value, 1);
    (void)setenv("VSPEC_ULTIMATE_ENABLE", ultimate_value, 1);
    (void)setenv("VSPEC_ULTIMATE_OUTLIER_AWARE", outlier_value, 1);
    (void)setenv("VSPEC_ULTIMATE_QLORA", qlora_value, 1);
    (void)setenv("VSPEC_ULTIMATE_TENSORCORE", tensorcore_value, 1);
    (void)setenv("VSPEC_ULTIMATE_OUTLIER_TH", outlier_th_value, 1);
    (void)setenv("VSPEC_ULTIMATE_QUALITY_BIAS", quality_bias_value, 1);
    (void)setenv("VSPEC_ULTIMATE_QLORA_RANK", qlora_rank_value, 1);
    (void)setenv("VSPEC_PRECISION_DOWNGRADE_TRIGGER", precision_downgrade_trigger_value, 1);
    (void)setenv("VSPEC_CACHE_COMPRESSION_TRIGGER", cache_compression_trigger_value, 1);
    (void)setenv("VSPEC_PER_MODEL_ADAPTIVE_BIT_CAP", per_model_adaptive_bit_cap_value, 1);
    (void)setenv("VSPEC_3BIT_RUNTIME_MODULE", runtime_3bit_value, 1);
    (void)setenv("VSPEC_3BIT_ATTN_QK_BITS", atn_qk_bits_value, 1);
    (void)setenv("VSPEC_3BIT_ATTN_PROJ_BITS", atn_proj_bits_value, 1);
    (void)setenv("VSPEC_3BIT_MLP_BITS", mlp_bits_value, 1);
    (void)setenv("VSPEC_3BIT_LM_HEAD_BITS", lm_head_bits_value, 1);
#endif
}

void vspec_runtime_init_default(void) {
    vspec_runtime_init_with_hw_config("config/runtime_hardware.conf");
}

void vspec_runtime_init_with_hw_config(const char* config_path) {
    vspec_runtime_hw_config_default(&g_hw_state.config);
    vspec_adaptive_precision_reset();
    g_hw_state.config_loaded_from_file = vspec_runtime_hw_config_load_file(config_path, &g_hw_state.config);
    vspec_runtime_apply_hw_env_hints(&g_hw_state.config);
    {
        VspecRuntimeControllerConfig ctrl_cfg;
        vspec_runtime_controller_config_default(&ctrl_cfg);
        ctrl_cfg.max_bits = (uint8_t)((g_hw_state.config.lowbit_target_bits >= 2U && g_hw_state.config.lowbit_target_bits <= 4U)
            ? g_hw_state.config.lowbit_target_bits
            : 4U);
        ctrl_cfg.min_bits = (ctrl_cfg.max_bits > 2U) ? (uint8_t)(ctrl_cfg.max_bits - 2U) : 2U;
        ctrl_cfg.latency_budget_ms = 30.0f + (float)(g_hw_state.config.dispatch_batch_hint * 0.5f);
        vspec_runtime_controller_init(&g_runtime_controller, &ctrl_cfg);
    }
    {
        VspecTokenSchedulerConfig sched_cfg;
        vspec_token_scheduler_config_default(&sched_cfg);
        sched_cfg.max_attention_depth = (unsigned int)(64U + g_hw_state.config.dispatch_batch_hint * 2U);
        if (sched_cfg.max_attention_depth < sched_cfg.base_attention_depth) {
            sched_cfg.max_attention_depth = sched_cfg.base_attention_depth;
        }
        vspec_token_scheduler_init(&g_token_scheduler, &sched_cfg);
    }
    {
        VspecPrecisionRouterConfig pr_cfg;
        vspec_precision_router_config_default(&pr_cfg);
        pr_cfg.max_bits = (uint8_t)((g_hw_state.config.lowbit_target_bits >= 2U && g_hw_state.config.lowbit_target_bits <= 4U)
            ? g_hw_state.config.lowbit_target_bits
            : 4U);
        pr_cfg.min_bits = (pr_cfg.max_bits > 2U) ? (uint8_t)(pr_cfg.max_bits - 2U) : 2U;
        pr_cfg.pressure_guard = g_hw_state.config.max_vram_utilization;
        vspec_precision_router_init(&g_precision_router, &pr_cfg);
    }
    {
        VspecMemoryPolicyConfig mem_cfg;
        vspec_memory_policy_config_default(&mem_cfg);
        mem_cfg.pressure_compress = g_hw_state.config.precision_downgrade_trigger;
        mem_cfg.pressure_recompute = g_hw_state.config.cache_compression_trigger;
        vspec_memory_policy_init(&g_memory_policy, &mem_cfg);
    }
    vspec_runtime_ultimate_init(&g_ultimate_state, &g_hw_state.config);
    vspec_qlora_adapter_clear();
    g_hw_state.active_backend_name = "cpu";
    vspec_runtime_enable_language_guard_auto();

    VspecBackend backend = vspec_make_cpu_backend();
    if (vspec_runtime_hw_pick_backend(&g_hw_state.config, &backend)) {
        vspec_set_backend(backend);
        g_hw_state.active_backend_name = backend.name ? backend.name : "cpu";
        vspec_runtime_behavior_monitor_init(
            &g_behavior_monitor,
            &g_hw_state.config,
            (backend.name && strcmp(backend.name, "cpu") != 0) ? 1 : 0
        );
        return;
    }

    vspec_set_backend(vspec_make_cpu_backend());
    g_hw_state.active_backend_name = "cpu";
    vspec_runtime_behavior_monitor_init(&g_behavior_monitor, &g_hw_state.config, 0);
}

const VspecRuntimeHwState* vspec_runtime_get_hw_state(void) {
    return &g_hw_state;
}

void vspec_linear_forward(VspecKernelContext* ctx) {
    const VspecBackend* backend = vspec_get_backend();
    if (!backend || !backend->launch_linear) {
        return;
    }
    backend->launch_linear(ctx);
}

void vspec_attention_forward(VspecKernelContext* ctx) {
    const VspecBackend* backend = vspec_get_backend();
    if (!backend || !backend->launch_attention) {
        return;
    }
    backend->launch_attention(ctx);
}

void vspec_runtime_language_guard_init(const char* prompt_text, float strictness) {
    VspecLanguageStructureGuardConfig cfg;
    vspec_language_structure_guard_config_default(&cfg);
    cfg.strictness = strictness;
    vspec_language_structure_guard_init(&g_language_guard, &cfg, prompt_text);
    g_language_guard_enabled = 1;
}

int vspec_runtime_language_guard_allow(const char* token_text) {
    if (!g_language_guard_enabled) {
        return 1;
    }
    return vspec_language_structure_guard_allow_text(&g_language_guard, token_text);
}

float vspec_runtime_language_guard_compensate(const char* token_text) {
    if (!g_language_guard_enabled) {
        return 0.0f;
    }
    return vspec_language_structure_guard_token_compensation(&g_language_guard, token_text);
}

void vspec_runtime_language_guard_observe(const char* token_text) {
    if (!g_language_guard_enabled) {
        return;
    }
    vspec_language_structure_guard_observe_text(&g_language_guard, token_text);
}

void vspec_runtime_language_guard_report(VspecLanguageStructureGuardReport* report) {
    if (!report) {
        return;
    }
    if (!g_language_guard_enabled) {
        (void)memset(report, 0, sizeof(*report));
        report->integrity_pass = 1;
        vspec_runtime_behavior_set_integrity_pass(1);
        return;
    }
    vspec_language_structure_guard_report(&g_language_guard, report);
    vspec_runtime_behavior_set_integrity_pass(report->integrity_pass);
}

void vspec_runtime_behavior_observe(
    float observed_gpu_utilization,
    float observed_vram_utilization,
    float observed_effective_bits
) {
    VspecRuntimeBehaviorSnapshot snapshot = g_behavior_monitor.latest;
    snapshot.observed_gpu_utilization = observed_gpu_utilization;
    snapshot.observed_vram_utilization = observed_vram_utilization;
    snapshot.observed_effective_bits = observed_effective_bits;
    snapshot.using_gpu_backend =
        (g_hw_state.active_backend_name && strcmp(g_hw_state.active_backend_name, "cpu") != 0) ? 1 : 0;

    if (!g_language_guard_enabled) {
        snapshot.integrity_pass = 1;
    }

    vspec_runtime_behavior_monitor_update(&g_behavior_monitor, &snapshot);
}

void vspec_runtime_behavior_observe_quality(
    float residual_rms,
    float attention_entropy_collapse,
    float activation_norm_drift
) {
    VspecRuntimeBehaviorSnapshot snapshot = g_behavior_monitor.latest;
    snapshot.residual_rms = residual_rms;
    snapshot.attention_entropy_collapse = attention_entropy_collapse;
    snapshot.activation_norm_drift = activation_norm_drift;
    snapshot.using_gpu_backend =
        (g_hw_state.active_backend_name && strcmp(g_hw_state.active_backend_name, "cpu") != 0) ? 1 : 0;

    if (!g_language_guard_enabled) {
        snapshot.integrity_pass = 1;
    }

    vspec_runtime_behavior_monitor_update(&g_behavior_monitor, &snapshot);
}

void vspec_runtime_behavior_set_workload_scale(float workload_scale) {
    if (workload_scale < 0.0f) {
        workload_scale = 0.0f;
    }
    if (workload_scale > 1.0f) {
        workload_scale = 1.0f;
    }
    g_behavior_monitor.latest.workload_scale = workload_scale;
}

void vspec_runtime_behavior_set_integrity_pass(int integrity_pass) {
    g_behavior_monitor.latest.integrity_pass = integrity_pass ? 1 : 0;
}

void vspec_runtime_behavior_report(VspecRuntimeBehaviorReport* report) {
    vspec_runtime_behavior_monitor_report(&g_behavior_monitor, report);
}

void vspec_runtime_set_model_precision_profile(
    uint16_t model_id,
    uint8_t bit_cap,
    int storage_heavy_mode,
    float precision_downgrade_trigger,
    float cache_compression_trigger
) {
    VspecAdaptiveModelProfile profile;
    profile.model_id = model_id;
    profile.bit_cap = bit_cap;
    profile.storage_heavy_mode = storage_heavy_mode ? 1 : 0;
    profile.precision_downgrade_trigger = precision_downgrade_trigger;
    profile.cache_compression_trigger = cache_compression_trigger;
    vspec_adaptive_precision_set_profile(&profile);
}

VspecQuantType vspec_runtime_ultimate_recommend_quant_for_input(
    const float* input,
    size_t count
) {
    return vspec_runtime_ultimate_recommend_quant(&g_ultimate_state, input, count);
}

void vspec_runtime_get_ultimate_report(VspecRuntimeUltimateReport* report) {
    vspec_runtime_ultimate_report(&g_ultimate_state, report);
}

int vspec_runtime_qlora_load_file(const char* path) {
    return vspec_qlora_adapter_load_file(path);
}

int vspec_runtime_qlora_load_manifest_json(const char* manifest_path) {
    return vspec_qlora_adapter_load_manifest_json(manifest_path);
}

void vspec_runtime_qlora_clear(void) {
    vspec_qlora_adapter_clear();
}

void vspec_runtime_adaptive_observe(const VspecRuntimeAdaptiveTelemetry* telemetry) {
    vspec_runtime_controller_observe(&g_runtime_controller, telemetry);
}

VspecRuntimeAdaptiveDecision vspec_runtime_adaptive_decide(void) {
    VspecRuntimeAdaptiveDecision d = vspec_runtime_controller_decide(&g_runtime_controller);
    vspec_plugin_emit_controller_decision(&g_runtime_controller.last, &d);
    return d;
}

VspecTokenScheduleDecision vspec_runtime_schedule_token(const char* token_text, float entropy_hint) {
    VspecTokenScheduleDecision d = vspec_token_scheduler_schedule_token(&g_token_scheduler, token_text, entropy_hint);
    vspec_plugin_emit_token_scheduled(token_text, &d);
    return d;
}

uint8_t vspec_runtime_route_precision(const VspecPrecisionRouteHint* hint) {
    return vspec_precision_router_select_bits(&g_precision_router, hint);
}

VspecKvPolicyAction vspec_runtime_memory_decide(const VspecMemoryPolicyInput* input) {
    return vspec_memory_policy_decide(&g_memory_policy, input);
}
