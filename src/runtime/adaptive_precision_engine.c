#include "vspec/runtime/adaptive_precision_engine.h"

#include <string.h>

#define VSPEC_ADAPTIVE_MAX_MODELS 64U

typedef struct VspecAdaptiveSlot {
    int used;
    VspecAdaptiveModelProfile profile;
} VspecAdaptiveSlot;

static VspecAdaptiveSlot g_slots[VSPEC_ADAPTIVE_MAX_MODELS];

void vspec_adaptive_precision_reset(void) {
    (void)memset(g_slots, 0, sizeof(g_slots));
}

void vspec_adaptive_precision_set_profile(const VspecAdaptiveModelProfile* profile) {
    if (!profile) {
        return;
    }

    int free_idx = -1;
    for (size_t i = 0U; i < VSPEC_ADAPTIVE_MAX_MODELS; ++i) {
        if (g_slots[i].used) {
            if (g_slots[i].profile.model_id == profile->model_id) {
                g_slots[i].profile = *profile;
                return;
            }
        } else if (free_idx < 0) {
            free_idx = (int)i;
        }
    }

    if (free_idx >= 0) {
        g_slots[free_idx].used = 1;
        g_slots[free_idx].profile = *profile;
    }
}

int vspec_adaptive_precision_get_profile(uint16_t model_id, VspecAdaptiveModelProfile* out_profile) {
    if (!out_profile) {
        return 0;
    }

    for (size_t i = 0U; i < VSPEC_ADAPTIVE_MAX_MODELS; ++i) {
        if (g_slots[i].used && g_slots[i].profile.model_id == model_id) {
            *out_profile = g_slots[i].profile;
            return 1;
        }
    }

    return 0;
}

uint8_t vspec_adaptive_precision_resolve_bit_cap(uint16_t model_id, uint8_t fallback_cap) {
    VspecAdaptiveModelProfile profile;
    if (vspec_adaptive_precision_get_profile(model_id, &profile)) {
        uint8_t cap = profile.bit_cap;
        if (cap < 2U) {
            cap = 2U;
        }
        if (cap > 4U) {
            cap = 4U;
        }
        return cap;
    }

    if (fallback_cap < 2U) {
        fallback_cap = 2U;
    }
    if (fallback_cap > 4U) {
        fallback_cap = 4U;
    }
    return fallback_cap;
}

float vspec_adaptive_precision_resolve_precision_downgrade_trigger(uint16_t model_id, float fallback_trigger) {
    VspecAdaptiveModelProfile profile;
    if (vspec_adaptive_precision_get_profile(model_id, &profile)) {
        const float trigger = profile.precision_downgrade_trigger;
        if (trigger > 0.0f) {
            return trigger;
        }
    }
    return fallback_trigger;
}

int vspec_adaptive_precision_should_compress_kv(uint16_t model_id, float pressure, float fallback_trigger) {
    VspecAdaptiveModelProfile profile;
    if (vspec_adaptive_precision_get_profile(model_id, &profile)) {
        float trigger = profile.cache_compression_trigger;
        if (trigger <= 0.0f) {
            trigger = fallback_trigger;
        }
        if (profile.storage_heavy_mode) {
            trigger -= 0.05f;
            if (trigger < 0.50f) {
                trigger = 0.50f;
            }
        }
        return pressure >= trigger;
    }

    return pressure >= fallback_trigger;
}
