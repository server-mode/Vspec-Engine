#ifndef VSPEC_COMPAT_WEIGHT_MAPPER_H
#define VSPEC_COMPAT_WEIGHT_MAPPER_H

#include <stddef.h>

#include "vspec/compat/types.h"

int vspec_weight_map_identity(const VspecCompatModel* in, VspecCompatModel* out);
int vspec_weight_canonical_name(const char* raw_name, char* out_name, size_t out_name_size);

#endif
