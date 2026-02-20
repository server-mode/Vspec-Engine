#ifndef VSPEC_ATTENTION_FLASH_BLOCK_H
#define VSPEC_ATTENTION_FLASH_BLOCK_H

#include <stddef.h>

void vspec_flash_attention_block_ref(
    const float* q,
    const float* k,
    const float* v,
    size_t tokens,
    size_t head_dim,
    size_t block_tokens,
    float* out
);

#endif
