#ifndef VSPEC_ATTENTION_FLASH_STUB_H
#define VSPEC_ATTENTION_FLASH_STUB_H

#include <stddef.h>

void vspec_flash_attention_stub(const float* q, const float* k, const float* v, size_t tokens, size_t head_dim, float* out);

#endif
