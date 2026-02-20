#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "vspec/quant/pack.h"

static int valid_bits(uint8_t bits) {
    return bits >= 2U && bits <= 4U;
}

size_t vspec_quant_packed_bytes(size_t elements, uint8_t bits) {
    if (!valid_bits(bits) || elements == 0U) {
        return 0U;
    }
    return ((elements * (size_t)bits) + 7U) / 8U;
}

int8_t vspec_quant_clip_signed(int8_t value, uint8_t bits) {
    if (!valid_bits(bits)) {
        return 0;
    }

    const int8_t minv = (int8_t)(-(1 << (bits - 1U)));
    const int8_t maxv = (int8_t)((1 << (bits - 1U)) - 1);
    if (value < minv) return minv;
    if (value > maxv) return maxv;
    return value;
}

void vspec_quant_pack_signed(const int8_t* src, size_t elements, uint8_t bits, uint8_t* dst) {
    if (!src || !dst || !valid_bits(bits)) {
        return;
    }

    const size_t bytes = vspec_quant_packed_bytes(elements, bits);
    memset(dst, 0, bytes);

    const uint8_t mask = (uint8_t)((1U << bits) - 1U);

    for (size_t i = 0; i < elements; ++i) {
        const uint8_t code = ((uint8_t)vspec_quant_clip_signed(src[i], bits)) & mask;
        const size_t bit_pos = i * (size_t)bits;
        const size_t byte_idx = bit_pos / 8U;
        const uint8_t shift = (uint8_t)(bit_pos % 8U);

        dst[byte_idx] |= (uint8_t)(code << shift);
        if ((uint8_t)(8U - shift) < bits) {
            dst[byte_idx + 1U] |= (uint8_t)(code >> (8U - shift));
        }
    }
}

int8_t vspec_quant_get_signed(const uint8_t* packed, size_t index, uint8_t bits) {
    if (!packed || !valid_bits(bits)) {
        return 0;
    }

    const uint8_t mask = (uint8_t)((1U << bits) - 1U);
    const size_t bit_pos = index * (size_t)bits;
    const size_t byte_idx = bit_pos / 8U;
    const uint8_t shift = (uint8_t)(bit_pos % 8U);

    uint16_t code = (uint16_t)(packed[byte_idx] >> shift);
    if ((uint8_t)(8U - shift) < bits) {
        code |= (uint16_t)(packed[byte_idx + 1U] << (8U - shift));
    }
    code &= mask;

    const uint8_t sign_bit = (uint8_t)(1U << (bits - 1U));
    if (((uint8_t)code & sign_bit) != 0U) {
        return (int8_t)(code - (1U << bits));
    }
    return (int8_t)code;
}

void vspec_quant_unpack_signed(const uint8_t* packed, size_t elements, uint8_t bits, int8_t* dst) {
    if (!packed || !dst || !valid_bits(bits)) {
        return;
    }

    for (size_t i = 0; i < elements; ++i) {
        dst[i] = vspec_quant_get_signed(packed, i, bits);
    }
}
