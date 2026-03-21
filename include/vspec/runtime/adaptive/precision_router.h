#ifndef VSPEC_RUNTIME_ADAPTIVE_PRECISION_ROUTER_H
#define VSPEC_RUNTIME_ADAPTIVE_PRECISION_ROUTER_H

#include <stdint.h>

#include "vspec/runtime/mixed_bit_policy.h"

typedef struct VspecPrecisionRouteHint {
    VspecLayerType layer_type;
    float token_importance;
    float vram_pressure;
    float quality_drift;
    uint8_t controller_target_bits;
} VspecPrecisionRouteHint;

typedef struct VspecPrecisionRouterConfig {
    uint8_t min_bits;
    uint8_t max_bits;
    float quality_drift_guard;
    float pressure_guard;
} VspecPrecisionRouterConfig;

typedef struct VspecPrecisionRouter {
    VspecPrecisionRouterConfig cfg;
} VspecPrecisionRouter;

void vspec_precision_router_config_default(VspecPrecisionRouterConfig* cfg);
void vspec_precision_router_init(VspecPrecisionRouter* router, const VspecPrecisionRouterConfig* cfg);
uint8_t vspec_precision_router_select_bits(const VspecPrecisionRouter* router, const VspecPrecisionRouteHint* hint);

#endif
