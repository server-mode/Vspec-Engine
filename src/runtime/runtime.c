#include "vspec/runtime/runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VSPEC_ANF_HOT_RATIO_WINDOW 512U

static VspecRuntimeHwState g_hw_state;
static VspecLanguageStructureGuard g_language_guard;
static int g_language_guard_enabled = 0;
static VspecRuntimeOutputGuard g_output_guard;
static int g_output_guard_enabled = 0;
static VspecRuntimeBehaviorMonitor g_behavior_monitor;
static VspecRuntimeUltimateState g_ultimate_state;
static VspecRuntimeController g_runtime_controller;
static VspecTokenScheduler g_token_scheduler;
static VspecPrecisionRouter g_precision_router;
static VspecMemoryPolicy g_memory_policy;
static VspecNeuronRouter g_neuron_router;
static VspecAnfPatternCache g_anf_pattern_cache;
static VspecErrorEstimator g_anf_error_estimator;
static VspecPrecisionWave g_anf_precision_wave;
static int g_anf_tcc_enabled = 0;
static int g_runtime_initialized = 0;

typedef struct VspecAnfTokenTelemetry {
    uint32_t tokens_observed;
    double hot_ratio_sum;
    float hot_ratio_window[VSPEC_ANF_HOT_RATIO_WINDOW];
    uint32_t hot_ratio_window_count;
    uint32_t hot_ratio_window_cursor;
    uint32_t last_hot_neurons;
    float last_hot_ratio;
} VspecAnfTokenTelemetry;

static VspecAnfTokenTelemetry g_anf_token_telemetry;

typedef struct VspecAnfSafetyState {
    uint32_t quality_guard_fail_streak;
    uint32_t deescalate_count;
    uint32_t forced_fallback_count;
    uint32_t silent_stop_count;
    int fallback_triggered;
} VspecAnfSafetyState;

static VspecAnfSafetyState g_anf_safety_state;

static int vspec_compare_float_asc(const void* lhs, const void* rhs) {
    const float a = *(const float*)lhs;
    const float b = *(const float*)rhs;
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}

static float vspec_anf_p95_hot_ratio(void) {
    float sample[VSPEC_ANF_HOT_RATIO_WINDOW];
    uint32_t n = g_anf_token_telemetry.hot_ratio_window_count;
    if (n == 0U) {
        return 0.0f;
    }
    if (n > VSPEC_ANF_HOT_RATIO_WINDOW) {
        n = VSPEC_ANF_HOT_RATIO_WINDOW;
    }
    (void)memcpy(sample, g_anf_token_telemetry.hot_ratio_window, (size_t)n * sizeof(float));
    qsort(sample, n, sizeof(float), vspec_compare_float_asc);
    {
        const uint32_t idx = (uint32_t)((((double)n - 1.0) * 0.95) + 0.5);
        return sample[idx];
    }
}

static size_t vspec_anf_shadow_fast_hot_count(
    const float* activations,
    size_t count,
    const VspecNeuronRouterConfig* cfg,
    size_t out_capacity
) {
    size_t above_threshold = 0U;
    size_t target;
    size_t min_required;

    if (!activations || !cfg || count == 0U || out_capacity == 0U) {
        return 0U;
    }

    target = (size_t)ceilf((float)count * cfg->max_hot_ratio);
    if (target < (size_t)cfg->min_hot_neurons) {
        target = (size_t)cfg->min_hot_neurons;
    }
    if (target > (size_t)cfg->max_hot_neurons) {
        target = (size_t)cfg->max_hot_neurons;
    }
    if (target > count) {
        target = count;
    }
    if (target > out_capacity) {
        target = out_capacity;
    }
    if (target == 0U) {
        return 0U;
    }

    for (size_t i = 0U; i < count; ++i) {
        if (fabsf(activations[i]) >= cfg->activation_threshold) {
            above_threshold += 1U;
        }
    }

    if (above_threshold > target) {
        above_threshold = target;
    }

    min_required = (size_t)cfg->min_hot_neurons;
    if (min_required > target) {
        min_required = target;
    }
    if (above_threshold < min_required) {
        above_threshold = min_required;
    }

    return above_threshold;
}

#if defined(VSPEC_ENABLE_ANF) && VSPEC_ENABLE_ANF
#define VSPEC_RUNTIME_ANF_ENABLED 1
#else
#define VSPEC_RUNTIME_ANF_ENABLED 0
#endif

static const char* vspec_anf_mode_to_env(VspecAnfMode mode) {
    switch (mode) {
        case VSPEC_ANF_MODE_SHADOW:
            return "shadow";
        case VSPEC_ANF_MODE_ACTIVE:
            return "active";
        case VSPEC_ANF_MODE_OFF:
        default:
            return "off";
    }
}

static int vspec_is_true_env(const char* v) {
    if (!v || v[0] == '\0') {
        return 0;
    }
    return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static int vspec_is_false_env(const char* v) {
    if (!v || v[0] == '\0') {
        return 0;
    }
    return (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}

static int vspec_runtime_anf_tcc_enabled_from_env(void) {
    const char* v = getenv("VSPEC_ANF_TCC_ENABLE");
    if (!v || v[0] == '\0') {
        /* Default ON to avoid telemetry-only TCC deployments. */
        return vspec_is_true_env(getenv("VSPEC_ENABLE_ANF")) ? 1 : 0;
    }
    return vspec_is_false_env(v) ? 0 : 1;
}

static uint32_t vspec_env_u32_clamped(const char* name, uint32_t def, uint32_t lo, uint32_t hi) {
    const char* v = getenv(name);
    uint32_t out = def;
    if (v && v[0] != '\0') {
        unsigned long parsed = strtoul(v, NULL, 10);
        if (parsed > 0UL) {
            out = (uint32_t)parsed;
        }
    }
    if (out < lo) out = lo;
    if (out > hi) out = hi;
    return out;
}

static float vspec_env_f32_clamped(const char* name, float def, float lo, float hi) {
    const char* v = getenv(name);
    float out = def;
    if (v && v[0] != '\0') {
        out = (float)atof(v);
    }
    if (out < lo) out = lo;
    if (out > hi) out = hi;
    return out;
}

static VspecAnfMode vspec_runtime_anf_deescalate_target_from_env(void) {
    const char* target_env = getenv("VSPEC_ANF_DEESCALATE_TARGET");
    if (!target_env || target_env[0] == '\0') {
        return VSPEC_ANF_MODE_SHADOW;
    }
    if (strcmp(target_env, "off") == 0) {
        return VSPEC_ANF_MODE_OFF;
    }
    if (strcmp(target_env, "shadow") == 0) {
        return VSPEC_ANF_MODE_SHADOW;
    }
    return VSPEC_ANF_MODE_SHADOW;
}

static void vspec_runtime_anf_apply_safety_policy(void) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    return;
#else
    const int policy_enabled =
        (!getenv("VSPEC_ANF_AUTO_DEESCALATE_ENABLE") || getenv("VSPEC_ANF_AUTO_DEESCALATE_ENABLE")[0] == '\0')
        ? 1
        : vspec_is_true_env(getenv("VSPEC_ANF_AUTO_DEESCALATE_ENABLE"));
    const uint32_t streak_threshold = vspec_env_u32_clamped("VSPEC_ANF_QUALITY_BREACH_STREAK", 3U, 1U, 32U);
    const int severe_quality_breach =
        (g_behavior_monitor.latest.residual_rms >= 1.35f) ||
        (g_behavior_monitor.latest.activation_norm_drift >= 0.30f) ||
        (g_behavior_monitor.latest.attention_entropy_collapse >= 0.65f) ||
        (g_anf_precision_wave.anomaly != 0);

    g_anf_safety_state.fallback_triggered = 0;
    if (severe_quality_breach) {
        g_anf_safety_state.quality_guard_fail_streak += 1U;
    } else {
        g_anf_safety_state.quality_guard_fail_streak = 0U;
    }

    if (!policy_enabled || g_neuron_router.config.mode != VSPEC_ANF_MODE_ACTIVE) {
        return;
    }

    if (g_anf_safety_state.quality_guard_fail_streak >= streak_threshold) {
        const VspecAnfMode target_mode = vspec_runtime_anf_deescalate_target_from_env();
        vspec_neuron_router_set_mode(&g_neuron_router, target_mode);
        g_hw_state.config.anf_mode = target_mode;
        g_anf_safety_state.deescalate_count += 1U;
        g_anf_safety_state.forced_fallback_count += 1U;
        g_anf_safety_state.fallback_triggered = 1;
        g_anf_safety_state.quality_guard_fail_streak = 0U;

#if defined(_WIN32)
        (void)_putenv_s("VSPEC_ANF_MODE", vspec_anf_mode_to_env(target_mode));
#else
        (void)setenv("VSPEC_ANF_MODE", vspec_anf_mode_to_env(target_mode), 1);
#endif
    }
#endif
}

static void vspec_runtime_apply_anf_env_overrides(VspecRuntimeHwConfig* cfg) {
    const char* mode_env;
    const char* hot_ratio_env;
    const char* min_hot_env;
    const char* max_hot_env;
    const char* threshold_env;

    if (!cfg) {
        return;
    }

    mode_env = getenv("VSPEC_ANF_MODE");
    if (mode_env && mode_env[0] != '\0') {
        if (strcmp(mode_env, "off") == 0) {
            cfg->anf_mode = VSPEC_ANF_MODE_OFF;
        } else if (strcmp(mode_env, "shadow") == 0) {
            cfg->anf_mode = VSPEC_ANF_MODE_SHADOW;
        } else if (strcmp(mode_env, "active") == 0) {
            cfg->anf_mode = VSPEC_ANF_MODE_ACTIVE;
        }
    }

    hot_ratio_env = getenv("VSPEC_ANF_MAX_HOT_RATIO");
    if (hot_ratio_env && hot_ratio_env[0] != '\0') {
        float v = (float)atof(hot_ratio_env);
        if (v < 0.01f) v = 0.01f;
        if (v > 0.50f) v = 0.50f;
        cfg->anf_max_hot_ratio = v;
    }

    min_hot_env = getenv("VSPEC_ANF_MIN_HOT_NEURONS");
    if (min_hot_env && min_hot_env[0] != '\0') {
        unsigned long v = strtoul(min_hot_env, NULL, 10);
        if (v > 65536UL) v = 65536UL;
        cfg->anf_min_hot_neurons = (uint32_t)v;
    }

    max_hot_env = getenv("VSPEC_ANF_MAX_HOT_NEURONS");
    if (max_hot_env && max_hot_env[0] != '\0') {
        unsigned long v = strtoul(max_hot_env, NULL, 10);
        if (v < 1UL) v = 1UL;
        if (v > 65536UL) v = 65536UL;
        cfg->anf_max_hot_neurons = (uint32_t)v;
    }

    threshold_env = getenv("VSPEC_ANF_ACTIVATION_THRESHOLD");
    if (threshold_env && threshold_env[0] != '\0') {
        float v = (float)atof(threshold_env);
        if (v < 0.0f) v = 0.0f;
        if (v > 8.0f) v = 8.0f;
        cfg->anf_activation_threshold = v;
    }

    if (cfg->anf_min_hot_neurons > cfg->anf_max_hot_neurons) {
        cfg->anf_min_hot_neurons = cfg->anf_max_hot_neurons;
    }

    if (vspec_is_true_env(getenv("VSPEC_CHAT_PROTOTYPE")) && vspec_is_true_env(getenv("VSPEC_ENABLE_ANF"))) {
        /* Prototype safety profile: keep hot routing selective by default. */
        if (!hot_ratio_env || hot_ratio_env[0] == '\0') {
            if (cfg->anf_max_hot_ratio > 0.10f) {
                cfg->anf_max_hot_ratio = 0.10f;
            }
        }
        if (!threshold_env || threshold_env[0] == '\0') {
            if (cfg->anf_activation_threshold < 1.10f) {
                cfg->anf_activation_threshold = 1.10f;
            }
        }
    }

    if (!vspec_is_true_env(getenv("VSPEC_ENABLE_ANF"))) {
        cfg->anf_mode = VSPEC_ANF_MODE_OFF;
    }
}

static void vspec_runtime_anf_autotune_hot_profile(float hot_ratio) {
    const float target_hi = vspec_env_f32_clamped("VSPEC_ANF_TARGET_HOT_RATIO_HI", 0.15f, 0.05f, 0.50f);
    const float target_lo = vspec_env_f32_clamped("VSPEC_ANF_TARGET_HOT_RATIO_LO", 0.05f, 0.01f, target_hi);
    const float relax_threshold = target_lo * 0.85f;

    if (!g_runtime_initialized || g_neuron_router.config.mode != VSPEC_ANF_MODE_ACTIVE) {
        return;
    }

    if (g_anf_token_telemetry.tokens_observed < 32U) {
        return;
    }

    if (hot_ratio > target_hi) {
        g_neuron_router.config.activation_threshold += 0.03f;
        if (g_neuron_router.config.activation_threshold > 2.50f) {
            g_neuron_router.config.activation_threshold = 2.50f;
        }
        if (g_neuron_router.config.max_hot_ratio > target_hi) {
            g_neuron_router.config.max_hot_ratio -= 0.01f;
            if (g_neuron_router.config.max_hot_ratio < target_hi) {
                g_neuron_router.config.max_hot_ratio = target_hi;
            }
        }
    } else if (hot_ratio < relax_threshold) {
        g_neuron_router.config.activation_threshold -= 0.01f;
        if (g_neuron_router.config.activation_threshold < 0.60f) {
            g_neuron_router.config.activation_threshold = 0.60f;
        }
    }
}

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

static void vspec_runtime_enable_output_guard_auto(void) {
    const char* disable = getenv("VSPEC_OUTPUT_GUARD_DISABLE");
    if (disable && (disable[0] == '1' || disable[0] == 'y' || disable[0] == 'Y' || disable[0] == 't' || disable[0] == 'T')) {
        g_output_guard_enabled = 0;
        return;
    }

    {
        const char* strictness_env = getenv("VSPEC_OUTPUT_GUARD_STRICTNESS");
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
        vspec_runtime_output_guard_init(strictness);
    }
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
    char anf_mode_value[16] = {0};
    char anf_hot_ratio_value[16] = {0};
    char anf_min_hot_value[16] = {0};
    char anf_max_hot_value[16] = {0};
    char anf_activation_threshold_value[16] = {0};
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
    (void)snprintf(anf_mode_value, sizeof(anf_mode_value), "%s", vspec_anf_mode_to_env(cfg->anf_mode));
    (void)snprintf(anf_hot_ratio_value, sizeof(anf_hot_ratio_value), "%.3f", cfg->anf_max_hot_ratio);
    (void)snprintf(anf_min_hot_value, sizeof(anf_min_hot_value), "%u", (unsigned)cfg->anf_min_hot_neurons);
    (void)snprintf(anf_max_hot_value, sizeof(anf_max_hot_value), "%u", (unsigned)cfg->anf_max_hot_neurons);
    (void)snprintf(anf_activation_threshold_value, sizeof(anf_activation_threshold_value), "%.3f", cfg->anf_activation_threshold);

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
    (void)_putenv_s("VSPEC_ANF_MODE", anf_mode_value);
    (void)_putenv_s("VSPEC_ANF_MAX_HOT_RATIO", anf_hot_ratio_value);
    (void)_putenv_s("VSPEC_ANF_MIN_HOT_NEURONS", anf_min_hot_value);
    (void)_putenv_s("VSPEC_ANF_MAX_HOT_NEURONS", anf_max_hot_value);
    (void)_putenv_s("VSPEC_ANF_ACTIVATION_THRESHOLD", anf_activation_threshold_value);
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
    (void)setenv("VSPEC_ANF_MODE", anf_mode_value, 1);
    (void)setenv("VSPEC_ANF_MAX_HOT_RATIO", anf_hot_ratio_value, 1);
    (void)setenv("VSPEC_ANF_MIN_HOT_NEURONS", anf_min_hot_value, 1);
    (void)setenv("VSPEC_ANF_MAX_HOT_NEURONS", anf_max_hot_value, 1);
    (void)setenv("VSPEC_ANF_ACTIVATION_THRESHOLD", anf_activation_threshold_value, 1);
#endif
}

void vspec_runtime_init_default(void) {
    vspec_runtime_init_with_hw_config("config/runtime_hardware.conf");
}

void vspec_runtime_init_with_hw_config(const char* config_path) {
    vspec_runtime_hw_config_default(&g_hw_state.config);
    vspec_adaptive_precision_reset();
    g_hw_state.config_loaded_from_file = vspec_runtime_hw_config_load_file(config_path, &g_hw_state.config);
    vspec_runtime_apply_anf_env_overrides(&g_hw_state.config);
#if !VSPEC_RUNTIME_ANF_ENABLED
    g_hw_state.config.anf_mode = VSPEC_ANF_MODE_OFF;
#endif
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
    {
        VspecNeuronRouterConfig anf_cfg;
        vspec_neuron_router_config_default(&anf_cfg);
        anf_cfg.mode = g_hw_state.config.anf_mode;
        anf_cfg.max_hot_ratio = g_hw_state.config.anf_max_hot_ratio;
        anf_cfg.min_hot_neurons = g_hw_state.config.anf_min_hot_neurons;
        anf_cfg.max_hot_neurons = g_hw_state.config.anf_max_hot_neurons;
        anf_cfg.activation_threshold = g_hw_state.config.anf_activation_threshold;
        vspec_neuron_router_init(&g_neuron_router, &anf_cfg);
    }
    {
        VspecAnfPatternCacheConfig pattern_cfg;
        vspec_anf_pattern_cache_config_default(&pattern_cfg);
        g_anf_tcc_enabled = vspec_runtime_anf_tcc_enabled_from_env();
        vspec_anf_pattern_cache_init(&g_anf_pattern_cache, &pattern_cfg);
    }
    {
        VspecErrorEstimatorConfig estimator_cfg;
        VspecPrecisionWaveConfig wave_cfg;
        vspec_error_estimator_config_default(&estimator_cfg);
        vspec_precision_wave_config_default(&wave_cfg);

        estimator_cfg.ema_decay = vspec_env_f32_clamped("VSPEC_ANF_ERROR_EMA_DECAY", estimator_cfg.ema_decay, 0.01f, 0.99f);
        wave_cfg.cascade_limit = vspec_env_u32_clamped("VSPEC_ANF_CASCADE_LIMIT", wave_cfg.cascade_limit, 0U, 16U);
        wave_cfg.escalate_wave_threshold = vspec_env_f32_clamped("VSPEC_ANF_CASCADE_ESC_WAVE", wave_cfg.escalate_wave_threshold, 0.01f, 1.00f);
        wave_cfg.deescalate_wave_threshold = vspec_env_f32_clamped("VSPEC_ANF_CASCADE_DESC_WAVE", wave_cfg.deescalate_wave_threshold, 0.00f, 1.00f);
        wave_cfg.escalate_contamination_threshold = vspec_env_f32_clamped("VSPEC_ANF_CASCADE_ESC_CONTAM", wave_cfg.escalate_contamination_threshold, 0.01f, 1.00f);
        wave_cfg.deescalate_contamination_threshold = vspec_env_f32_clamped("VSPEC_ANF_CASCADE_DESC_CONTAM", wave_cfg.deescalate_contamination_threshold, 0.00f, 1.00f);
        wave_cfg.cooldown_updates = vspec_env_u32_clamped("VSPEC_ANF_CASCADE_COOLDOWN", wave_cfg.cooldown_updates, 0U, 64U);

        vspec_error_estimator_init(&g_anf_error_estimator, &estimator_cfg);
        vspec_precision_wave_init(&g_anf_precision_wave, &wave_cfg);
    }
    (void)memset(&g_anf_token_telemetry, 0, sizeof(g_anf_token_telemetry));
    (void)memset(&g_anf_safety_state, 0, sizeof(g_anf_safety_state));
    vspec_runtime_ultimate_init(&g_ultimate_state, &g_hw_state.config);
    vspec_qlora_adapter_clear();
    g_hw_state.active_backend_name = "cpu";
    vspec_runtime_enable_language_guard_auto();
    vspec_runtime_enable_output_guard_auto();

    VspecBackend backend = vspec_make_cpu_backend();
    if (vspec_runtime_hw_pick_backend(&g_hw_state.config, &backend)) {
        vspec_set_backend(backend);
        g_hw_state.active_backend_name = backend.name ? backend.name : "cpu";
        vspec_runtime_behavior_monitor_init(
            &g_behavior_monitor,
            &g_hw_state.config,
            (backend.name && strcmp(backend.name, "cpu") != 0) ? 1 : 0
        );
        g_runtime_initialized = 1;
        return;
    }

    vspec_set_backend(vspec_make_cpu_backend());
    g_hw_state.active_backend_name = "cpu";
    vspec_runtime_behavior_monitor_init(&g_behavior_monitor, &g_hw_state.config, 0);
    g_runtime_initialized = 1;
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
    int allow = 1;
    if (g_language_guard_enabled) {
        allow = vspec_language_structure_guard_allow_text(&g_language_guard, token_text);
    }
    if (allow && g_output_guard_enabled) {
        allow = vspec_output_guard_allow(&g_output_guard, token_text);
    }
    return allow;
}

float vspec_runtime_language_guard_compensate(const char* token_text) {
    float score = 0.0f;
    if (g_language_guard_enabled) {
        score += vspec_language_structure_guard_token_compensation(&g_language_guard, token_text);
    }
    if (g_output_guard_enabled) {
        score += vspec_output_guard_score_adjustment(&g_output_guard, token_text);
    }
    return score;
}

void vspec_runtime_language_guard_observe(const char* token_text) {
    if (g_language_guard_enabled) {
        vspec_language_structure_guard_observe_text(&g_language_guard, token_text);
    }
    if (g_output_guard_enabled) {
        vspec_output_guard_observe(&g_output_guard, token_text);
    }
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

    if (g_runtime_initialized && g_neuron_router.config.mode != VSPEC_ANF_MODE_OFF) {
        VspecErrorEstimatorReport estimator_report;
        vspec_error_estimator_observe(
            &g_anf_error_estimator,
            residual_rms,
            attention_entropy_collapse,
            activation_norm_drift
        );
        vspec_error_estimator_report(&g_anf_error_estimator, &estimator_report);
        vspec_precision_wave_observe(
            &g_anf_precision_wave,
            estimator_report.wave_score_ema,
            estimator_report.contamination_ema
        );
        vspec_runtime_anf_apply_safety_policy();
    }
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
    if (!report) {
        return;
    }

    report->anf_available = vspec_runtime_anf_available();
    if (report->anf_available) {
        VspecNeuronRouterReport anf_report;
        VspecAnfPatternCacheReport pattern_report;
        VspecErrorEstimatorReport estimator_report;
        VspecPrecisionWaveReport wave_report;
        vspec_runtime_anf_router_report(&anf_report);
        vspec_runtime_anf_pattern_cache_report(&pattern_report);
        vspec_runtime_anf_error_estimator_report(&estimator_report);
        vspec_runtime_anf_precision_wave_report(&wave_report);
        report->anf_mode = (int)anf_report.mode;
        report->anf_hot_ratio = anf_report.hot_ratio;
        report->anf_hot_neurons = (uint32_t)anf_report.hot_neurons;
        report->anf_tokens_observed = g_anf_token_telemetry.tokens_observed;
        report->anf_hot_ratio_avg = (g_anf_token_telemetry.tokens_observed > 0U)
            ? (float)(g_anf_token_telemetry.hot_ratio_sum / (double)g_anf_token_telemetry.tokens_observed)
            : 0.0f;
        report->anf_hot_ratio_p95 = vspec_anf_p95_hot_ratio();
        report->anf_cache_updates = pattern_report.updates;
        report->anf_skip_ratio_last = pattern_report.last_skip_ratio;
        report->anf_skip_ratio_avg = pattern_report.skip_ratio_ema;
        report->anf_changed_ratio_last = pattern_report.last_changed_ratio;
        report->anf_pattern_confidence = pattern_report.confidence;
        report->anf_error_wave_last = estimator_report.wave_score_last;
        report->anf_error_wave_avg = estimator_report.wave_score_ema;
        report->anf_contamination_last = estimator_report.contamination_last;
        report->anf_contamination_avg = estimator_report.contamination_ema;
        report->anf_cascade_depth = wave_report.cascade_depth;
        report->anf_cascade_depth_max = wave_report.cascade_depth_max;
        report->anf_cascade_escalations = wave_report.escalations;
        report->anf_cascade_deescalations = wave_report.deescalations;
        report->anf_cascade_anomaly = wave_report.anomaly;
        report->anf_quality_guard_fail_streak = g_anf_safety_state.quality_guard_fail_streak;
        report->anf_deescalate_count = g_anf_safety_state.deescalate_count;
        report->anf_forced_fallback_count = g_anf_safety_state.forced_fallback_count;
        report->anf_silent_stop_count = g_anf_safety_state.silent_stop_count;
        report->anf_fallback_triggered = g_anf_safety_state.fallback_triggered;
        if (g_anf_safety_state.forced_fallback_count > 0U) {
            report->issue_mask |= VSPEC_RUNTIME_ISSUE_ANF_FORCED_FALLBACK;
        }
        if (wave_report.anomaly) {
            report->issue_mask |= VSPEC_RUNTIME_ISSUE_QUALITY_DRIFT;
            report->issue_mask |= VSPEC_RUNTIME_ISSUE_CASCADE_ANOMALY;
        }
    } else {
        report->anf_mode = (int)VSPEC_ANF_MODE_OFF;
        report->anf_hot_ratio = 0.0f;
        report->anf_hot_neurons = 0U;
        report->anf_tokens_observed = 0U;
        report->anf_hot_ratio_avg = 0.0f;
        report->anf_hot_ratio_p95 = 0.0f;
        report->anf_cache_updates = 0U;
        report->anf_skip_ratio_last = 0.0f;
        report->anf_skip_ratio_avg = 0.0f;
        report->anf_changed_ratio_last = 0.0f;
        report->anf_pattern_confidence = 0.0f;
        report->anf_error_wave_last = 0.0f;
        report->anf_error_wave_avg = 0.0f;
        report->anf_contamination_last = 0.0f;
        report->anf_contamination_avg = 0.0f;
        report->anf_cascade_depth = 0U;
        report->anf_cascade_depth_max = 0U;
        report->anf_cascade_escalations = 0U;
        report->anf_cascade_deescalations = 0U;
        report->anf_cascade_anomaly = 0;
        report->anf_quality_guard_fail_streak = 0U;
        report->anf_deescalate_count = 0U;
        report->anf_forced_fallback_count = 0U;
        report->anf_silent_stop_count = 0U;
        report->anf_fallback_triggered = 0;
    }
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

void vspec_runtime_output_guard_init(float strictness) {
    VspecRuntimeOutputGuardConfig cfg;
    vspec_output_guard_config_default(&cfg);
    cfg.strictness = strictness;
    if (cfg.strictness < 0.0f) {
        cfg.strictness = 0.0f;
    }
    if (cfg.strictness > 1.0f) {
        cfg.strictness = 1.0f;
    }
    vspec_output_guard_init(&g_output_guard, &cfg);
    g_output_guard_enabled = 1;
}

int vspec_runtime_output_guard_allow(const char* text_fragment) {
    if (!g_output_guard_enabled) {
        return 1;
    }
    return vspec_output_guard_allow(&g_output_guard, text_fragment);
}

float vspec_runtime_output_guard_score_adjustment(const char* text_fragment) {
    if (!g_output_guard_enabled) {
        return 0.0f;
    }
    return vspec_output_guard_score_adjustment(&g_output_guard, text_fragment);
}

void vspec_runtime_output_guard_observe(const char* text_fragment) {
    if (!g_output_guard_enabled) {
        return;
    }
    vspec_output_guard_observe(&g_output_guard, text_fragment);
}

void vspec_runtime_output_guard_report(VspecRuntimeOutputGuardReport* report) {
    if (!report) {
        return;
    }
    if (!g_output_guard_enabled) {
        (void)memset(report, 0, sizeof(*report));
        report->integrity_pass = 1;
        return;
    }
    vspec_output_guard_report(&g_output_guard, report);
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
    uint8_t bits = vspec_precision_router_select_bits(&g_precision_router, hint);
#if !VSPEC_RUNTIME_ANF_ENABLED
    return bits;
#else
    if (!hint || !g_runtime_initialized || g_neuron_router.config.mode == VSPEC_ANF_MODE_OFF) {
        return bits;
    }

    {
        const char* route_env = getenv("VSPEC_ANF_CASCADE_ROUTE_ENABLE");
        const int route_enable = (!route_env || route_env[0] == '\0') ? 1 : vspec_is_true_env(route_env);
        const uint32_t depth = g_anf_precision_wave.cascade_depth;
        const int anomaly = g_anf_precision_wave.anomaly;
        uint8_t floor_bits = 3U;

        if (!route_enable) {
            return bits;
        }

        if (anomaly || depth >= 1U) {
            floor_bits = 4U;
        }

        if (bits < floor_bits) {
            bits = floor_bits;
        }
        if (bits > 4U) {
            bits = 4U;
        }
    }

    return bits;
#endif
}

VspecKvPolicyAction vspec_runtime_memory_decide(const VspecMemoryPolicyInput* input) {
    return vspec_memory_policy_decide(&g_memory_policy, input);
}

void vspec_runtime_anf_router_configure(const VspecNeuronRouterConfig* cfg) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    (void)cfg;
    return;
#else
    vspec_neuron_router_init(&g_neuron_router, cfg);
#endif
}

size_t vspec_runtime_anf_select_hot_neurons(
    const float* activations,
    size_t count,
    uint32_t* out_indices,
    size_t out_capacity
) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    (void)activations;
    (void)count;
    (void)out_indices;
    (void)out_capacity;
    return 0U;
#else
    return vspec_neuron_router_select_hot(&g_neuron_router, activations, count, out_indices, out_capacity);
#endif
}

void vspec_runtime_anf_router_report(VspecNeuronRouterReport* report) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    if (!report) {
        return;
    }
    (void)memset(report, 0, sizeof(*report));
    report->mode = VSPEC_ANF_MODE_OFF;
#else
    vspec_neuron_router_report(&g_neuron_router, report);
#endif
}

int vspec_runtime_anf_available(void) {
    return VSPEC_RUNTIME_ANF_ENABLED;
}

void vspec_runtime_anf_observe_token_activations(const float* activations, size_t count) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    (void)activations;
    (void)count;
    return;
#else
    uint32_t hot_indices_stack[2048];
    uint32_t* hot_indices = hot_indices_stack;
    size_t cap = 0U;
    size_t hot_count = 0U;
    int have_hot_indices = 0;
    int use_heap = 0;

    if (!g_runtime_initialized || !activations || count == 0U) {
        return;
    }
    if (g_neuron_router.config.mode == VSPEC_ANF_MODE_OFF) {
        return;
    }

    cap = count;
    if (cap > (size_t)g_neuron_router.config.max_hot_neurons) {
        cap = (size_t)g_neuron_router.config.max_hot_neurons;
    }
    if (cap == 0U) {
        return;
    }

    if ((g_neuron_router.config.mode != VSPEC_ANF_MODE_SHADOW || g_anf_tcc_enabled) &&
        cap > (sizeof(hot_indices_stack) / sizeof(hot_indices_stack[0]))) {
        hot_indices = (uint32_t*)malloc(cap * sizeof(uint32_t));
        if (!hot_indices) {
            return;
        }
        use_heap = 1;
    }

    if (g_neuron_router.config.mode == VSPEC_ANF_MODE_SHADOW) {
        if (g_anf_tcc_enabled) {
            hot_count = vspec_runtime_anf_select_hot_neurons(activations, count, hot_indices, cap);
            have_hot_indices = 1;
        } else {
            hot_count = vspec_anf_shadow_fast_hot_count(activations, count, &g_neuron_router.config, cap);
            g_neuron_router.last_input_neurons = count;
            g_neuron_router.last_hot_neurons = hot_count;
            g_neuron_router.last_hot_ratio = (count > 0U) ? ((float)hot_count / (float)count) : 0.0f;
        }
    } else {
        hot_count = vspec_runtime_anf_select_hot_neurons(activations, count, hot_indices, cap);
        have_hot_indices = 1;
    }

    if (g_anf_tcc_enabled && have_hot_indices) {
        vspec_anf_pattern_cache_update(&g_anf_pattern_cache, hot_indices, hot_count, count);
    }

    if (use_heap) {
        free(hot_indices);
    }

    {
        const float hot_ratio = (count > 0U) ? ((float)hot_count / (float)count) : 0.0f;
        const uint32_t window_pos = g_anf_token_telemetry.hot_ratio_window_cursor % VSPEC_ANF_HOT_RATIO_WINDOW;
        g_anf_token_telemetry.tokens_observed += 1U;
        g_anf_token_telemetry.hot_ratio_sum += (double)hot_ratio;
        g_anf_token_telemetry.last_hot_neurons = (uint32_t)hot_count;
        g_anf_token_telemetry.last_hot_ratio = hot_ratio;
        g_anf_token_telemetry.hot_ratio_window[window_pos] = hot_ratio;
        if (g_anf_token_telemetry.hot_ratio_window_count < VSPEC_ANF_HOT_RATIO_WINDOW) {
            g_anf_token_telemetry.hot_ratio_window_count += 1U;
        }
        g_anf_token_telemetry.hot_ratio_window_cursor += 1U;

        /* Keep ANF selective over long runs when activation distribution drifts. */
        vspec_runtime_anf_autotune_hot_profile(hot_ratio);
    }
#endif
}

void vspec_runtime_anf_pattern_cache_report(VspecAnfPatternCacheReport* report) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    if (!report) {
        return;
    }
    (void)memset(report, 0, sizeof(*report));
#else
    vspec_anf_pattern_cache_report(&g_anf_pattern_cache, report);
#endif
}

void vspec_runtime_anf_error_estimator_report(VspecErrorEstimatorReport* report) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    if (!report) {
        return;
    }
    (void)memset(report, 0, sizeof(*report));
#else
    vspec_error_estimator_report(&g_anf_error_estimator, report);
#endif
}

void vspec_runtime_anf_precision_wave_report(VspecPrecisionWaveReport* report) {
#if !VSPEC_RUNTIME_ANF_ENABLED
    if (!report) {
        return;
    }
    (void)memset(report, 0, sizeof(*report));
#else
    vspec_precision_wave_report(&g_anf_precision_wave, report);
#endif
}
