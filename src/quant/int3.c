#include <stddef.h>
#include <stdint.h>

#include "vspec/quant/int3.h"
#include "vspec/quant/pack.h"

size_t vspec_int3_packed_bytes(size_t elements) {
    return vspec_quant_packed_bytes(elements, 3);
}

void vspec_int3_matmul_ref_f32_q3(
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

    const size_t row_bytes = vspec_int3_packed_bytes(k);

    for (size_t i = 0; i < m; ++i) {
        const float* a_row = a + (i * k);
        float* c_row = c + (i * n);

        for (size_t j = 0; j < n; ++j) {
            const uint8_t* b_row = b_packed + (j * row_bytes);
            const float scale = scales[j];
            float acc = 0.0f;

            for (size_t t = 0; t < k; ++t) {
                const int8_t wq = vspec_quant_get_signed(b_row, t, 3);
                acc += a_row[t] * ((float)wq * scale);
            }

            c_row[j] = acc;
        }
    }
}
