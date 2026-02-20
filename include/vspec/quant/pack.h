#ifndef VSPEC_QUANT_PACK_H
#define VSPEC_QUANT_PACK_H

#include <stddef.h>
#include <stdint.h>

size_t vspec_quant_packed_bytes(size_t elements, uint8_t bits);
int8_t vspec_quant_clip_signed(int8_t value, uint8_t bits);

void vspec_quant_pack_signed(const int8_t* src, size_t elements, uint8_t bits, uint8_t* dst);
int8_t vspec_quant_get_signed(const uint8_t* packed, size_t index, uint8_t bits);
void vspec_quant_unpack_signed(const uint8_t* packed, size_t elements, uint8_t bits, int8_t* dst);

#endif
