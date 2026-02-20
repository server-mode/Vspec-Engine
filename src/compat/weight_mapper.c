#include <string.h>

#include "vspec/compat/weight_mapper.h"

int vspec_weight_map_identity(const VspecCompatModel* in, VspecCompatModel* out) {
    if (!in || !out) {
        return 0;
    }

    *out = *in;
    return 1;
}
