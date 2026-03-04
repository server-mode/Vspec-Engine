#ifndef VSPEC_RUNTIME_ADAPTIVE_PRECISION_ENGINE_H
#define VSPEC_RUNTIME_ADAPTIVE_PRECISION_ENGINE_H

#include <stdint.h>

typedef struct VspecAdaptiveModelProfile {
    uint16_t model_id;
    uint8_t bit_cap;
    int storage_heavy_mode;
    float precision_downgrade_trigger;
    float cache_compression_trigger;
} VspecAdaptiveModelProfile;

void vspec_adaptive_precision_reset(void);
void vspec_adaptive_precision_set_profile(const VspecAdaptiveModelProfile* profile);
int vspec_adaptive_precision_get_profile(uint16_t model_id, VspecAdaptiveModelProfile* out_profile);
uint8_t vspec_adaptive_precision_resolve_bit_cap(uint16_t model_id, uint8_t fallback_cap);
float vspec_adaptive_precision_resolve_precision_downgrade_trigger(uint16_t model_id, float fallback_trigger);
int vspec_adaptive_precision_should_compress_kv(uint16_t model_id, float pressure, float fallback_trigger);

#endif