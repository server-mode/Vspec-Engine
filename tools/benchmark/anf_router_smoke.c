#include <stdio.h>
#include <stdint.h>

#include "vspec/runtime/runtime.h"

int main(void) {
    vspec_runtime_init_default();

    if (!vspec_runtime_anf_available()) {
        printf("[anf_router_smoke] anf_available=0 (compile-time disabled)\n");
        printf("[anf_router_smoke] status=pass\n");
        return 0;
    }

    const float activations[16] = {
        0.05f, 0.10f, 0.15f, 0.20f,
        1.90f, 0.30f, 1.25f, 0.40f,
        0.50f, 0.60f, 0.70f, 1.10f,
        0.80f, 0.90f, 0.02f, 0.01f
    };

    VspecNeuronRouterConfig cfg;
    vspec_neuron_router_config_default(&cfg);
    cfg.mode = VSPEC_ANF_MODE_SHADOW;
    cfg.max_hot_ratio = 0.25f;
    cfg.min_hot_neurons = 2U;
    cfg.max_hot_neurons = 6U;
    cfg.activation_threshold = 0.85f;

    vspec_runtime_anf_router_configure(&cfg);

    uint32_t hot_indices[8] = {0};
    const size_t hot_count = vspec_runtime_anf_select_hot_neurons(
        activations,
        16U,
        hot_indices,
        8U
    );

    VspecNeuronRouterReport report;
    vspec_runtime_anf_router_report(&report);

    printf("[anf_router_smoke] mode=%d input=%zu hot=%zu ratio=%.3f threshold=%.3f\n",
        (int)report.mode,
        report.input_neurons,
        report.hot_neurons,
        report.hot_ratio,
        report.activation_threshold);

    printf("[anf_router_smoke] hot_indices=");
    for (size_t i = 0U; i < hot_count; ++i) {
        printf("%u", (unsigned)hot_indices[i]);
        if (i + 1U < hot_count) {
            printf(",");
        }
    }
    printf("\n");

    if (hot_count < 2U || hot_count > 6U) {
        printf("[anf_router_smoke] status=fail (hot_count out of configured range)\n");
        return 1;
    }

    printf("[anf_router_smoke] status=pass\n");
    return 0;
}
