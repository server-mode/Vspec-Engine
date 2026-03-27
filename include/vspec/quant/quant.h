#ifndef VSPEC_QUANT_QUANT_H
#define VSPEC_QUANT_QUANT_H

#include <stddef.h>

#include "vspec/quant/formats/special_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VspecQuantType {
    VSPEC_QUANT_NONE = 0,
    VSPEC_QUANT_INT2 = 2,
    VSPEC_QUANT_INT3 = 3,
    VSPEC_QUANT_INT4 = 4
} VspecQuantType;

typedef struct VspecQuantMeta {
    unsigned int schema_version;
    VspecQuantType type;
    const float* scales;
    size_t scale_count;
    VspecSpecialQuantFormat special_format;
    const void* special_data;
} VspecQuantMeta;

void vspec_quant_meta_init(VspecQuantMeta* meta);

#ifdef __cplusplus
}
#endif

#endif
