#ifndef VSPEC_QUANT_QUANT_H
#define VSPEC_QUANT_QUANT_H

#include <stddef.h>

typedef enum VspecQuantType {
    VSPEC_QUANT_NONE = 0,
    VSPEC_QUANT_INT4 = 4
} VspecQuantType;

typedef struct VspecQuantMeta {
    unsigned int schema_version;
    VspecQuantType type;
    const float* scales;
    size_t scale_count;
} VspecQuantMeta;

void vspec_quant_meta_init(VspecQuantMeta* meta);

#endif
