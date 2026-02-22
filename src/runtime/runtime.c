#include "vspec/runtime/runtime.h"

#include <stdio.h>
#include <stdlib.h>

static VspecRuntimeHwState g_hw_state;

static void vspec_runtime_apply_hw_env_hints(const VspecRuntimeHwConfig* cfg) {
    if (!cfg || !cfg->enable_lowbit_boost) {
        return;
    }

    char bits_value[8] = {0};
    (void)snprintf(bits_value, sizeof(bits_value), "%u", (unsigned)cfg->lowbit_target_bits);

#if defined(_WIN32)
    (void)_putenv_s("VSPEC_FUSED_BITS", bits_value);
    (void)_putenv_s("VSPEC_DISABLE_FUSED_ATTN", "0");
#else
    (void)setenv("VSPEC_FUSED_BITS", bits_value, 1);
    (void)setenv("VSPEC_DISABLE_FUSED_ATTN", "0", 1);
#endif
}

void vspec_runtime_init_default(void) {
    vspec_runtime_init_with_hw_config("config/runtime_hardware.conf");
}

void vspec_runtime_init_with_hw_config(const char* config_path) {
    vspec_runtime_hw_config_default(&g_hw_state.config);
    g_hw_state.config_loaded_from_file = vspec_runtime_hw_config_load_file(config_path, &g_hw_state.config);
    vspec_runtime_apply_hw_env_hints(&g_hw_state.config);
    g_hw_state.active_backend_name = "cpu";

    VspecBackend backend = vspec_make_cpu_backend();
    if (vspec_runtime_hw_pick_backend(&g_hw_state.config, &backend)) {
        vspec_set_backend(backend);
        g_hw_state.active_backend_name = backend.name ? backend.name : "cpu";
        return;
    }

    vspec_set_backend(vspec_make_cpu_backend());
    g_hw_state.active_backend_name = "cpu";
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
