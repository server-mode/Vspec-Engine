#include "vspec/runtime/adaptive/precision_router.h"

#include <stddef.h>

static uint8_t clamp_bits(uint8_t bits, uint8_t min_bits, uint8_t max_bits) {
    if (bits < min_bits) {
        return min_bits;
    }
    if (bits > max_bits) {
        return max_bits;
    }
    return bits;
}

void vspec_precision_router_config_default(VspecPrecisionRouterConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->min_bits = 2U;
    cfg->max_bits = 4U;
    cfg->quality_drift_guard = 0.30f;
    cfg->pressure_guard = 0.90f;
}

void vspec_precision_router_init(VspecPrecisionRouter* router, const VspecPrecisionRouterConfig* cfg) {
    if (!router) {
        return;
    }
    if (cfg) {
        router->cfg = *cfg;
    } else {
        vspec_precision_router_config_default(&router->cfg);
    }
}

uint8_t vspec_precision_router_select_bits(const VspecPrecisionRouter* router, const VspecPrecisionRouteHint* hint) {
    if (!router || !hint) {
        return 4U;
    }

    uint8_t bits = hint->controller_target_bits;

    if (hint->layer_type == VSPEC_LAYER_ATTENTION_QK ||
        hint->layer_type == VSPEC_LAYER_ATTENTION_PROJ ||
        hint->layer_type == VSPEC_LAYER_LM_HEAD) {
        bits = 4U;
    }

    if (hint->token_importance >= 0.75f && bits < 4U) {
        bits += 1U;
    }

    if (hint->vram_pressure >= router->cfg.pressure_guard && bits > router->cfg.min_bits) {
        bits -= 1U;
    }

    if (hint->quality_drift >= router->cfg.quality_drift_guard) {
        bits = router->cfg.max_bits;
    }

    return clamp_bits(bits, router->cfg.min_bits, router->cfg.max_bits);
}
