#ifndef VSPEC_RUNTIME_NEURON_ROUTER_H
#define VSPEC_RUNTIME_NEURON_ROUTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VspecAnfMode {
    VSPEC_ANF_MODE_OFF = 0,
    VSPEC_ANF_MODE_SHADOW = 1,
    VSPEC_ANF_MODE_ACTIVE = 2,
} VspecAnfMode;

typedef struct VspecNeuronRouterConfig {
    VspecAnfMode mode;
    float max_hot_ratio;
    uint32_t min_hot_neurons;
    uint32_t max_hot_neurons;
    float activation_threshold;
} VspecNeuronRouterConfig;

typedef struct VspecNeuronRouter {
    VspecNeuronRouterConfig config;
    size_t last_input_neurons;
    size_t last_hot_neurons;
    float last_hot_ratio;
} VspecNeuronRouter;

typedef struct VspecNeuronRouterReport {
    VspecAnfMode mode;
    size_t input_neurons;
    size_t hot_neurons;
    float hot_ratio;
    float activation_threshold;
    float max_hot_ratio;
} VspecNeuronRouterReport;

void vspec_neuron_router_config_default(VspecNeuronRouterConfig* cfg);
void vspec_neuron_router_init(VspecNeuronRouter* router, const VspecNeuronRouterConfig* cfg);
void vspec_neuron_router_set_mode(VspecNeuronRouter* router, VspecAnfMode mode);
size_t vspec_neuron_router_select_hot(
    VspecNeuronRouter* router,
    const float* activations,
    size_t count,
    uint32_t* out_indices,
    size_t out_capacity
);
void vspec_neuron_router_report(const VspecNeuronRouter* router, VspecNeuronRouterReport* report);

#ifdef __cplusplus
}
#endif

#endif
