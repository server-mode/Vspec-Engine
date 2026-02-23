#include "vspec/runtime/hw_performance_manager.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* vspec_trim(char* text) {
    if (!text) {
        return text;
    }

    while (*text && isspace((unsigned char)*text)) {
        ++text;
    }

    char* end = text + strlen(text);
    while (end > text && isspace((unsigned char)*(end - 1))) {
        --end;
    }
    *end = '\0';
    return text;
}

static int vspec_equals_ignore_case(const char* a, const char* b) {
    if (!a || !b) {
        return 0;
    }

    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        ++a;
        ++b;
    }

    return *a == '\0' && *b == '\0';
}

static float vspec_clamp_float(float value, float lo, float hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static uint32_t vspec_clamp_u32(uint32_t value, uint32_t lo, uint32_t hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static uint8_t vspec_clamp_u8(uint8_t value, uint8_t lo, uint8_t hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static void vspec_parse_tuning_mode(const char* value, VspecRuntimeHwConfig* config) {
    if (vspec_equals_ignore_case(value, "performent") ||
        vspec_equals_ignore_case(value, "performance") ||
        vspec_equals_ignore_case(value, "performant")) {
        config->tuning_mode = VSPEC_HW_TUNING_PERFORMENT;
    } else if (vspec_equals_ignore_case(value, "balanced")) {
        config->tuning_mode = VSPEC_HW_TUNING_BALANCED;
    } else if (vspec_equals_ignore_case(value, "eco")) {
        config->tuning_mode = VSPEC_HW_TUNING_ECO;
    } else if (vspec_equals_ignore_case(value, "ultimate")) {
        config->tuning_mode = VSPEC_HW_TUNING_ULTIMATE;
    }
}

static void vspec_parse_backend_preference(const char* value, VspecRuntimeHwConfig* config) {
    if (vspec_equals_ignore_case(value, "auto")) {
        config->backend_preference = VSPEC_BACKEND_PREFERENCE_AUTO;
    } else if (vspec_equals_ignore_case(value, "cuda")) {
        config->backend_preference = VSPEC_BACKEND_PREFERENCE_CUDA;
    } else if (vspec_equals_ignore_case(value, "rocm")) {
        config->backend_preference = VSPEC_BACKEND_PREFERENCE_ROCM;
    } else if (vspec_equals_ignore_case(value, "sycl")) {
        config->backend_preference = VSPEC_BACKEND_PREFERENCE_SYCL;
    } else if (vspec_equals_ignore_case(value, "cpu")) {
        config->backend_preference = VSPEC_BACKEND_PREFERENCE_CPU;
    }
}

void vspec_runtime_hw_config_default(VspecRuntimeHwConfig* config) {
    if (!config) {
        return;
    }

    config->tuning_mode = VSPEC_HW_TUNING_PERFORMENT;
    config->backend_preference = VSPEC_BACKEND_PREFERENCE_AUTO;
    config->target_gpu_utilization = 0.98f;
    config->max_vram_utilization = 0.96f;
    config->dispatch_batch_hint = 12U;
    config->stream_count_hint = 6U;
    config->lowbit_target_bits = 3U;
    config->enable_lowbit_boost = 1;

    config->enable_ultimate_mode = 0;
    config->enable_outlier_aware = 1;
    config->enable_qlora_adapter = 0;
    config->prefer_tensor_core = 1;
    config->outlier_threshold = 6.0f;
    config->quality_bias = 0.75f;
    config->qlora_rank = 16U;
}

int vspec_runtime_hw_config_load_file(const char* path, VspecRuntimeHwConfig* out_config) {
    if (!out_config) {
        return 0;
    }

    vspec_runtime_hw_config_default(out_config);

    if (!path || path[0] == '\0') {
        return 0;
    }

    FILE* file = fopen(path, "rb");
    if (!file) {
        return 0;
    }

    char line[256];
    while (fgets(line, (int)sizeof(line), file)) {
        char* text = vspec_trim(line);
        if (*text == '\0' || *text == '#' || *text == ';') {
            continue;
        }

        char* eq = strchr(text, '=');
        if (!eq) {
            continue;
        }

        *eq = '\0';
        char* key = vspec_trim(text);
        char* value = vspec_trim(eq + 1);

        if (vspec_equals_ignore_case(key, "mode")) {
            vspec_parse_tuning_mode(value, out_config);
        } else if (vspec_equals_ignore_case(key, "backend_preference")) {
            vspec_parse_backend_preference(value, out_config);
        } else if (vspec_equals_ignore_case(key, "target_gpu_utilization")) {
            out_config->target_gpu_utilization = vspec_clamp_float((float)atof(value), 0.10f, 1.00f);
        } else if (vspec_equals_ignore_case(key, "max_vram_utilization")) {
            out_config->max_vram_utilization = vspec_clamp_float((float)atof(value), 0.10f, 1.00f);
        } else if (vspec_equals_ignore_case(key, "dispatch_batch_hint")) {
            out_config->dispatch_batch_hint = vspec_clamp_u32((uint32_t)strtoul(value, NULL, 10), 1U, 128U);
        } else if (vspec_equals_ignore_case(key, "stream_count_hint")) {
            out_config->stream_count_hint = vspec_clamp_u32((uint32_t)strtoul(value, NULL, 10), 1U, 32U);
        } else if (vspec_equals_ignore_case(key, "lowbit_target_bits")) {
            out_config->lowbit_target_bits = vspec_clamp_u8((uint8_t)strtoul(value, NULL, 10), 2U, 3U);
        } else if (vspec_equals_ignore_case(key, "enable_lowbit_boost")) {
            out_config->enable_lowbit_boost =
                (vspec_equals_ignore_case(value, "1") ||
                 vspec_equals_ignore_case(value, "true") ||
                 vspec_equals_ignore_case(value, "yes")) ? 1 : 0;
        } else if (vspec_equals_ignore_case(key, "enable_ultimate_mode")) {
            out_config->enable_ultimate_mode =
                (vspec_equals_ignore_case(value, "1") ||
                 vspec_equals_ignore_case(value, "true") ||
                 vspec_equals_ignore_case(value, "yes")) ? 1 : 0;
        } else if (vspec_equals_ignore_case(key, "enable_outlier_aware")) {
            out_config->enable_outlier_aware =
                (vspec_equals_ignore_case(value, "1") ||
                 vspec_equals_ignore_case(value, "true") ||
                 vspec_equals_ignore_case(value, "yes")) ? 1 : 0;
        } else if (vspec_equals_ignore_case(key, "enable_qlora_adapter")) {
            out_config->enable_qlora_adapter =
                (vspec_equals_ignore_case(value, "1") ||
                 vspec_equals_ignore_case(value, "true") ||
                 vspec_equals_ignore_case(value, "yes")) ? 1 : 0;
        } else if (vspec_equals_ignore_case(key, "prefer_tensor_core")) {
            out_config->prefer_tensor_core =
                (vspec_equals_ignore_case(value, "1") ||
                 vspec_equals_ignore_case(value, "true") ||
                 vspec_equals_ignore_case(value, "yes")) ? 1 : 0;
        } else if (vspec_equals_ignore_case(key, "outlier_threshold")) {
            out_config->outlier_threshold = vspec_clamp_float((float)atof(value), 2.0f, 20.0f);
        } else if (vspec_equals_ignore_case(key, "quality_bias")) {
            out_config->quality_bias = vspec_clamp_float((float)atof(value), 0.0f, 1.0f);
        } else if (vspec_equals_ignore_case(key, "qlora_rank")) {
            out_config->qlora_rank = vspec_clamp_u32((uint32_t)strtoul(value, NULL, 10), 0U, 256U);
        }
    }

    fclose(file);

    if (out_config->tuning_mode == VSPEC_HW_TUNING_PERFORMENT) {
        if (out_config->dispatch_batch_hint < 8U) {
            out_config->dispatch_batch_hint = 8U;
        }
        if (out_config->stream_count_hint < 4U) {
            out_config->stream_count_hint = 4U;
        }
        out_config->lowbit_target_bits = vspec_clamp_u8(out_config->lowbit_target_bits, 2U, 3U);
        out_config->target_gpu_utilization = vspec_clamp_float(out_config->target_gpu_utilization, 0.90f, 1.00f);
    } else if (out_config->tuning_mode == VSPEC_HW_TUNING_ULTIMATE) {
        out_config->enable_ultimate_mode = 1;
        out_config->enable_outlier_aware = 1;
        out_config->prefer_tensor_core = 1;
        out_config->enable_qlora_adapter = 1;
        out_config->target_gpu_utilization = vspec_clamp_float(out_config->target_gpu_utilization, 0.85f, 1.00f);
        out_config->quality_bias = vspec_clamp_float(out_config->quality_bias, 0.60f, 1.00f);
        if (out_config->qlora_rank < 8U) {
            out_config->qlora_rank = 8U;
        }
    }

    return 1;
}

int vspec_runtime_hw_pick_backend(const VspecRuntimeHwConfig* config, VspecBackend* out_backend) {
    if (!out_backend) {
        return 0;
    }

    const VspecBackendPreference pref = config ? config->backend_preference : VSPEC_BACKEND_PREFERENCE_AUTO;
    switch (pref) {
        case VSPEC_BACKEND_PREFERENCE_CUDA:
            if (vspec_cuda_backend_available()) {
                *out_backend = vspec_make_cuda_backend();
                return 1;
            }
            break;
        case VSPEC_BACKEND_PREFERENCE_ROCM:
            if (vspec_rocm_backend_available()) {
                *out_backend = vspec_make_rocm_backend();
                return 1;
            }
            break;
        case VSPEC_BACKEND_PREFERENCE_SYCL:
            if (vspec_sycl_backend_available()) {
                *out_backend = vspec_make_sycl_backend();
                return 1;
            }
            break;
        case VSPEC_BACKEND_PREFERENCE_CPU:
            *out_backend = vspec_make_cpu_backend();
            return 1;
        case VSPEC_BACKEND_PREFERENCE_AUTO:
        default:
            break;
    }

    if (vspec_cuda_backend_available()) {
        *out_backend = vspec_make_cuda_backend();
        return 1;
    }

    if (vspec_rocm_backend_available()) {
        *out_backend = vspec_make_rocm_backend();
        return 1;
    }

    if (vspec_sycl_backend_available()) {
        *out_backend = vspec_make_sycl_backend();
        return 1;
    }

    *out_backend = vspec_make_cpu_backend();
    return 1;
}
