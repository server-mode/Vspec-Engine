#ifndef VSPEC_ATTENTION_ROTARY_H
#define VSPEC_ATTENTION_ROTARY_H

#include <stddef.h>

void vspec_rotary_apply(float* q, float* k, size_t head_dim, size_t base_pos);

#endif
