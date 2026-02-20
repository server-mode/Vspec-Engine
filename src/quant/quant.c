#include "vspec/quant/quant.h"
#include "vspec/version.h"

void vspec_quant_meta_init(VspecQuantMeta* meta) {
    if (!meta) {
        return;
    }
    meta->schema_version = VSPEC_QUANT_SCHEMA_VERSION;
    meta->type = VSPEC_QUANT_NONE;
    meta->scales = 0;
    meta->scale_count = 0;
}
