#ifndef VSPEC_VALIDATION_QUANT_SENSITIVITY_H
#define VSPEC_VALIDATION_QUANT_SENSITIVITY_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecQuantSensitivity {
    uint8_t bits;
    float mean_abs;
    float max_abs;
} VspecQuantSensitivity;

VspecQuantSensitivity vspec_quant_sensitivity(const float* baseline, const float* quantized, size_t count, uint8_t bits);

#endif
