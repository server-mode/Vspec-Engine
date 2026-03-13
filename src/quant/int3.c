#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "vspec/quant/int3.h"
#include "vspec/quant/pack.h"
#include "vspec/runtime/three_bit_runtime_modules.h"

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
    const int use_runtime_3bit = vspec_runtime_3bit_enabled();
    const size_t block = vspec_3bit_resolve_block_size(64U);

    float* a_row_clamped = NULL;
    if (use_runtime_3bit) {
        a_row_clamped = (float*)malloc(k * sizeof(float));
        if (!a_row_clamped) {
            return;
        }
    }

    for (size_t i = 0; i < m; ++i) {
        const float* a_row = a + (i * k);
        float* c_row = c + (i * n);

        if (use_runtime_3bit) {
            vspec_3bit_dynamic_clamp_std(a_row, k, 2.8f, a_row_clamped);
        }

        for (size_t j = 0; j < n; ++j) {
            const uint8_t* b_row = b_packed + (j * row_bytes);
            const float scale = scales[j];
            float zero_point = 0.0f;
            {
                double sum_q = 0.0;
                for (size_t t = 0; t < k; ++t) {
                    const int8_t wq = vspec_quant_get_signed(b_row, t, 3);
                    sum_q += (double)wq;
                }
                zero_point = (float)(sum_q / (double)k);
            }
            float acc = 0.0f;

            if (!use_runtime_3bit) {
                for (size_t t = 0; t < k; ++t) {
                    const int8_t wq = vspec_quant_get_signed(b_row, t, 3);
                    acc += a_row[t] * (((float)wq - zero_point) * scale);
                }
            } else {
                for (size_t base = 0; base < k; base += block) {
                    size_t end = base + block;
                    if (end > k) {
                        end = k;
                    }

                    float block_abs = 0.0f;
                    for (size_t t = base; t < end; ++t) {
                        const float av = a_row_clamped[t];
                        const int8_t wq = vspec_quant_get_signed(b_row, t, 3);
                        const float bv = ((float)wq - zero_point) * scale;
                        const float aabs = (av < 0.0f) ? -av : av;
                        const float babs = (bv < 0.0f) ? -bv : bv;
                        if (aabs > block_abs) {
                            block_abs = aabs;
                        }
                        if (babs > block_abs) {
                            block_abs = babs;
                        }
                    }

                    const float block_scale = (block_abs > 1e-6f) ? block_abs : 1e-6f;
                    float block_acc = 0.0f;
                    for (size_t t = base; t < end; ++t) {
                        const int8_t wq = vspec_quant_get_signed(b_row, t, 3);
                        const float bn = (((float)wq - zero_point) * scale) / block_scale;
                        const float an = a_row_clamped[t] / block_scale;
                        block_acc += an * bn;
                    }
                    acc += block_acc * block_scale * block_scale;
                }
            }

            c_row[j] = acc;
        }
    }

    free(a_row_clamped);
}
