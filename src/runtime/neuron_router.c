#include "vspec/runtime/neuron_router.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

void vspec_neuron_router_config_default(VspecNeuronRouterConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->mode = VSPEC_ANF_MODE_OFF;
    cfg->max_hot_ratio = 0.15f;
    cfg->min_hot_neurons = 16U;
    cfg->max_hot_neurons = 1024U;
    cfg->activation_threshold = 0.80f;
}

void vspec_neuron_router_init(VspecNeuronRouter* router, const VspecNeuronRouterConfig* cfg) {
    if (!router) {
        return;
    }
    (void)memset(router, 0, sizeof(*router));
    if (cfg) {
        router->config = *cfg;
    } else {
        vspec_neuron_router_config_default(&router->config);
    }
    router->config.max_hot_ratio = vspec_clamp_float(router->config.max_hot_ratio, 0.01f, 0.50f);
    router->config.max_hot_neurons = vspec_clamp_u32(router->config.max_hot_neurons, 1U, 65536U);
    if (router->config.min_hot_neurons > router->config.max_hot_neurons) {
        router->config.min_hot_neurons = router->config.max_hot_neurons;
    }
    if (router->config.activation_threshold < 0.0f) {
        router->config.activation_threshold = 0.0f;
    }
}

void vspec_neuron_router_set_mode(VspecNeuronRouter* router, VspecAnfMode mode) {
    if (!router) {
        return;
    }
    if (mode < VSPEC_ANF_MODE_OFF || mode > VSPEC_ANF_MODE_ACTIVE) {
        mode = VSPEC_ANF_MODE_OFF;
    }
    router->config.mode = mode;
}

static void vspec_topk_insert(
    float* top_values,
    uint32_t* top_indices,
    size_t k,
    float value,
    uint32_t index
) {
    if (k == 0U) {
        return;
    }

    size_t pos = k;
    for (size_t i = 0U; i < k; ++i) {
        if (value > top_values[i]) {
            pos = i;
            break;
        }
    }

    if (pos == k) {
        return;
    }

    for (size_t i = k - 1U; i > pos; --i) {
        top_values[i] = top_values[i - 1U];
        top_indices[i] = top_indices[i - 1U];
    }
    top_values[pos] = value;
    top_indices[pos] = index;
}

size_t vspec_neuron_router_select_hot(
    VspecNeuronRouter* router,
    const float* activations,
    size_t count,
    uint32_t* out_indices,
    size_t out_capacity
) {
    if (!router || !activations || !out_indices || count == 0U || out_capacity == 0U) {
        return 0U;
    }

    router->last_input_neurons = count;
    router->last_hot_neurons = 0U;
    router->last_hot_ratio = 0.0f;

    if (router->config.mode == VSPEC_ANF_MODE_OFF) {
        return 0U;
    }

    size_t target = (size_t)ceilf((float)count * router->config.max_hot_ratio);
    if (target < (size_t)router->config.min_hot_neurons) {
        target = (size_t)router->config.min_hot_neurons;
    }
    if (target > (size_t)router->config.max_hot_neurons) {
        target = (size_t)router->config.max_hot_neurons;
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

    float top_values_stack[256];
    uint32_t top_indices_stack[256];
    float* top_values = top_values_stack;
    uint32_t* top_indices = top_indices_stack;
    int use_heap = 0;

    if (target > 256U) {
        top_values = (float*)malloc(target * sizeof(float));
        top_indices = (uint32_t*)malloc(target * sizeof(uint32_t));
        if (!top_values || !top_indices) {
            free(top_values);
            free(top_indices);
            target = 256U;
            top_values = top_values_stack;
            top_indices = top_indices_stack;
        } else {
            use_heap = 1;
        }
    }

    for (size_t i = 0U; i < target; ++i) {
        top_values[i] = -1.0f;
        top_indices[i] = 0U;
    }

    for (size_t i = 0U; i < count; ++i) {
        float magnitude = fabsf(activations[i]);
        vspec_topk_insert(top_values, top_indices, target, magnitude, (uint32_t)i);
    }

    size_t selected = 0U;
    for (size_t i = 0U; i < target; ++i) {
        if (top_values[i] < 0.0f) {
            continue;
        }
        if (top_values[i] >= router->config.activation_threshold) {
            out_indices[selected++] = top_indices[i];
        }
    }

    size_t min_required = (size_t)router->config.min_hot_neurons;
    if (min_required > target) {
        min_required = target;
    }

    for (size_t i = 0U; selected < min_required && i < target; ++i) {
        uint32_t candidate = top_indices[i];
        int exists = 0;
        for (size_t j = 0U; j < selected; ++j) {
            if (out_indices[j] == candidate) {
                exists = 1;
                break;
            }
        }
        if (!exists) {
            out_indices[selected++] = candidate;
        }
    }

    router->last_hot_neurons = selected;
    router->last_hot_ratio = (count > 0U) ? ((float)selected / (float)count) : 0.0f;
    if (use_heap) {
        free(top_values);
        free(top_indices);
    }
    return selected;
}

void vspec_neuron_router_report(const VspecNeuronRouter* router, VspecNeuronRouterReport* report) {
    if (!router || !report) {
        return;
    }
    report->mode = router->config.mode;
    report->input_neurons = router->last_input_neurons;
    report->hot_neurons = router->last_hot_neurons;
    report->hot_ratio = router->last_hot_ratio;
    report->activation_threshold = router->config.activation_threshold;
    report->max_hot_ratio = router->config.max_hot_ratio;
}
