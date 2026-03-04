#ifndef VSPEC_RUNTIME_HW_PERFORMANCE_MANAGER_H
#define VSPEC_RUNTIME_HW_PERFORMANCE_MANAGER_H

#include <stdint.h>

#include "vspec/kernel/backend.h"

typedef enum VspecHwTuningMode {
    VSPEC_HW_TUNING_PERFORMENT = 0,
    VSPEC_HW_TUNING_BALANCED = 1,
    VSPEC_HW_TUNING_ECO = 2,
    VSPEC_HW_TUNING_ULTIMATE = 3,
} VspecHwTuningMode;

typedef enum VspecBackendPreference {
    VSPEC_BACKEND_PREFERENCE_AUTO = 0,
    VSPEC_BACKEND_PREFERENCE_CUDA = 1,
    VSPEC_BACKEND_PREFERENCE_ROCM = 2,
    VSPEC_BACKEND_PREFERENCE_SYCL = 3,
    VSPEC_BACKEND_PREFERENCE_CPU = 4,
} VspecBackendPreference;

typedef struct VspecRuntimeHwConfig {
    VspecHwTuningMode tuning_mode;
    VspecBackendPreference backend_preference;
    float target_gpu_utilization;
    float max_vram_utilization;
    uint32_t dispatch_batch_hint;
    uint32_t stream_count_hint;
    uint8_t lowbit_target_bits;
    int enable_lowbit_boost;

    int enable_ultimate_mode;
    int enable_outlier_aware;
    int enable_qlora_adapter;
    int prefer_tensor_core;
    float outlier_threshold;
    float quality_bias;
    uint32_t qlora_rank;

    float precision_downgrade_trigger;
    float cache_compression_trigger;
    uint8_t per_model_adaptive_bit_cap;
} VspecRuntimeHwConfig;

typedef struct VspecRuntimeHwState {
    VspecRuntimeHwConfig config;
    int config_loaded_from_file;
    const char* active_backend_name;
} VspecRuntimeHwState;

void vspec_runtime_hw_config_default(VspecRuntimeHwConfig* config);
int vspec_runtime_hw_config_load_file(const char* path, VspecRuntimeHwConfig* out_config);
int vspec_runtime_hw_pick_backend(const VspecRuntimeHwConfig* config, VspecBackend* out_backend);

#endif
