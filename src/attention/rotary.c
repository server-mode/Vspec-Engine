#include <math.h>

#include "vspec/attention/rotary.h"

void vspec_rotary_apply(float* q, float* k, size_t head_dim, size_t base_pos) {
    if (!q || !k || head_dim < 2) {
        return;
    }

    for (size_t i = 0; i + 1 < head_dim; i += 2) {
        const float angle = (float)base_pos * 0.001f * (float)(i + 1);
        const float c = cosf(angle);
        const float s = sinf(angle);

        const float q0 = q[i];
        const float q1 = q[i + 1];
        const float k0 = k[i];
        const float k1 = k[i + 1];

        q[i] = q0 * c - q1 * s;
        q[i + 1] = q0 * s + q1 * c;
        k[i] = k0 * c - k1 * s;
        k[i + 1] = k0 * s + k1 * c;
    }
}
