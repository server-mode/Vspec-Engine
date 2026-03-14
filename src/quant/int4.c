#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "vspec/quant/int4.h"

size_t vspec_int4_packed_bytes(size_t elements) {
    return (elements + 1U) / 2U;
}

static uint8_t encode_int4(int8_t v) {
    if (v < -8) v = -8;
    if (v > 7) v = 7;
    return (uint8_t)(v & 0x0F);
}

static int8_t decode_int4(uint8_t nibble) {
    nibble &= 0x0F;
    if (nibble & 0x08) {
        return (int8_t)(nibble - 16);
    }
    return (int8_t)nibble;
}

void vspec_int4_pack(const int8_t* src, size_t elements, uint8_t* dst) {
    if (!src || !dst) {
        return;
    }

    size_t out_i = 0;
    for (size_t i = 0; i < elements; i += 2) {
        uint8_t lo = encode_int4(src[i]);
        uint8_t hi = 0;
        if (i + 1 < elements) {
            hi = (uint8_t)(encode_int4(src[i + 1]) << 4);
        }
        dst[out_i++] = (uint8_t)(lo | hi);
    }
}

int8_t vspec_int4_get(const uint8_t* packed, size_t index) {
    const uint8_t byte = packed[index / 2U];
    const uint8_t nibble = (index % 2U == 0U) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    return decode_int4(nibble);
}

void vspec_int4_matmul_ref_f32_q4(
    const float* a,
    size_t m,
    size_t k,
    const uint8_t* b_packed,
    size_t n,
    const float* scales,
    float* c
) {
    if (!a || !b_packed || !scales || !c) {
        return;
    }

    float* zero_points = (float*)malloc(n * sizeof(float));
    if (!zero_points) {
        return;
    }
    vspec_int4_compute_zero_points(b_packed, k, n, zero_points);
    vspec_int4_matmul_ref_f32_q4_with_zero_points(a, m, k, b_packed, n, scales, zero_points, c);
    free(zero_points);
}

void vspec_int4_compute_zero_points(
    const uint8_t* b_packed,
    size_t k,
    size_t n,
    float* zero_points
) {
    if (!b_packed || !zero_points || k == 0U || n == 0U) {
        return;
    }

    const size_t b_row_packed = vspec_int4_packed_bytes(k);
    for (size_t j = 0U; j < n; ++j) {
        const uint8_t* b_row = b_packed + (j * b_row_packed);
        double sum_q = 0.0;
        for (size_t t = 0U; t < k; ++t) {
            const int8_t wq = vspec_int4_get(b_row, t);
            sum_q += (double)wq;
        }
        zero_points[j] = (float)(sum_q / (double)k);
    }
}

void vspec_int4_matmul_ref_f32_q4_with_zero_points(
    const float* a,
    size_t m,
    size_t k,
    const uint8_t* b_packed,
    size_t n,
    const float* scales,
    const float* zero_points,
    float* c
) {
    if (!a || !b_packed || !scales || !c) {
        return;
    }

    const size_t b_row_packed = vspec_int4_packed_bytes(k);

    for (size_t i = 0; i < m; ++i) {
        const float* a_row = a + (i * k);
        float* c_row = c + (i * n);

        for (size_t j = 0; j < n; ++j) {
            const uint8_t* b_row = b_packed + (j * b_row_packed);
            const float scale = scales[j];
            const float zero_point = zero_points ? zero_points[j] : 0.0f;
            float acc = 0.0f;

            size_t t = 0;
            for (; t + 1U < k; t += 2U) {
                const uint8_t byte = b_row[t >> 1U];
                const int8_t w0 = decode_int4(byte & 0x0F);
                const int8_t w1 = decode_int4((byte >> 4) & 0x0F);
                acc += a_row[t] * (((float)w0 - zero_point) * scale);
                acc += a_row[t + 1U] * (((float)w1 - zero_point) * scale);
            }

            if (t < k) {
                const int8_t wq = vspec_int4_get(b_row, t);
                acc += a_row[t] * (((float)wq - zero_point) * scale);
            }

            c_row[j] = acc;
        }
    }
}
