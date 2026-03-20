#include "vspec/quant/formats/q4_k.h"

#include <stdlib.h>
#include <string.h>

static float vspec_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000U) << 16U;
    const uint32_t exp = (uint32_t)(h >> 10U) & 0x1FU;
    const uint32_t frac = (uint32_t)h & 0x3FFU;

    uint32_t out_bits = 0U;
    if (exp == 0U) {
        if (frac == 0U) {
            out_bits = sign;
        } else {
            uint32_t mant = frac;
            uint32_t e = 113U;
            while ((mant & 0x400U) == 0U) {
                mant <<= 1U;
                e -= 1U;
            }
            mant &= 0x3FFU;
            out_bits = sign | (e << 23U) | (mant << 13U);
        }
    } else if (exp == 31U) {
        out_bits = sign | 0x7F800000U | (frac << 13U);
    } else {
        out_bits = sign | ((exp + 112U) << 23U) | (frac << 13U);
    }

    float out = 0.0f;
    memcpy(&out, &out_bits, sizeof(out));
    return out;
}

static uint8_t vspec_q4k_nibble(const uint8_t* qs, size_t idx) {
    const uint8_t byte = qs[idx >> 1U];
    return (idx & 1U) ? (uint8_t)((byte >> 4U) & 0x0FU) : (uint8_t)(byte & 0x0FU);
}

static void vspec_q4k_decode_6bit_stream(const uint8_t* packed12, uint8_t out16[16]) {
    size_t bit_pos = 0U;
    for (size_t i = 0U; i < 16U; ++i) {
        size_t byte_pos = bit_pos >> 3U;
        size_t shift = bit_pos & 7U;
        uint32_t v = (uint32_t)packed12[byte_pos] >> shift;
        if (byte_pos + 1U < 12U) {
            v |= (uint32_t)packed12[byte_pos + 1U] << (8U - shift);
        }
        if (shift > 2U && byte_pos + 2U < 12U) {
            v |= (uint32_t)packed12[byte_pos + 2U] << (16U - shift);
        }
        out16[i] = (uint8_t)(v & 0x3FU);
        bit_pos += 6U;
    }
}

size_t vspec_q4k_blocks_for_elements(size_t elements) {
    return (elements + VSPEC_Q4_K_BLOCK_ELEMENTS - 1U) / VSPEC_Q4_K_BLOCK_ELEMENTS;
}

size_t vspec_q4k_storage_bytes(size_t elements) {
    return vspec_q4k_blocks_for_elements(elements) * sizeof(VspecQ4KBlock);
}

void vspec_q4k_dequantize_row(
    const VspecQ4KBlock* row_blocks,
    size_t elements,
    float* out
) {
    if (!row_blocks || !out || elements == 0U) {
        return;
    }

    const size_t blocks = vspec_q4k_blocks_for_elements(elements);
    size_t produced = 0U;

    for (size_t b = 0U; b < blocks; ++b) {
        const VspecQ4KBlock* blk = &row_blocks[b];
        const float d = vspec_half_to_float(blk->d);
        const float dmin = vspec_half_to_float(blk->dmin);
        uint8_t sm[16];
        vspec_q4k_decode_6bit_stream(blk->scales, sm);

        for (size_t sb = 0U; sb < VSPEC_Q4_K_SUBBLOCKS; ++sb) {
            const float sub_scale = d * ((float)sm[sb] / 63.0f);
            const float sub_min = dmin * ((float)sm[8U + sb] / 63.0f);
            for (size_t t = 0U; t < VSPEC_Q4_K_SUBBLOCK_ELEMENTS; ++t) {
                if (produced >= elements) {
                    return;
                }
                const size_t local = sb * VSPEC_Q4_K_SUBBLOCK_ELEMENTS + t;
                const float q = (float)vspec_q4k_nibble(blk->qs, local);
                out[produced++] = (sub_scale * q) - sub_min;
            }
        }
    }
}

void vspec_q4k_matmul_ref_f32(
    const float* a,
    size_t m,
    size_t k,
    const VspecQ4KBlock* b_rows,
    size_t n,
    float* c
) {
    if (!a || !b_rows || !c || m == 0U || n == 0U || k == 0U) {
        return;
    }

    float* row_tmp = (float*)malloc(k * sizeof(float));
    if (!row_tmp) {
        return;
    }

    const size_t blocks_per_row = vspec_q4k_blocks_for_elements(k);

    for (size_t j = 0U; j < n; ++j) {
        const VspecQ4KBlock* row_blocks = b_rows + j * blocks_per_row;
        vspec_q4k_dequantize_row(row_blocks, k, row_tmp);

        for (size_t i = 0U; i < m; ++i) {
            const float* a_row = a + i * k;
            float acc = 0.0f;
            for (size_t t = 0U; t < k; ++t) {
                acc += a_row[t] * row_tmp[t];
            }
            c[i * n + j] = acc;
        }
    }

    free(row_tmp);
}
