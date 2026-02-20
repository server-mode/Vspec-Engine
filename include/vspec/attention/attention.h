#ifndef VSPEC_ATTENTION_ATTENTION_H
#define VSPEC_ATTENTION_ATTENTION_H

#include <stddef.h>

#include "vspec/attention/kv_cache.h"

void vspec_attention_ref_single_query(
    const float* query,
    const VspecKVCache* cache,
    float* out
);

#endif
