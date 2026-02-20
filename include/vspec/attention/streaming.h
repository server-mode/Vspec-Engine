#ifndef VSPEC_ATTENTION_STREAMING_H
#define VSPEC_ATTENTION_STREAMING_H

#include <stddef.h>

void vspec_attention_streaming_ref(
    const float* q,
    const float* k,
    const float* v,
    size_t tokens,
    size_t head_dim,
    size_t chunk,
    float* out
);

#endif
