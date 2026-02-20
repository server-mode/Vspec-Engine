#include <string.h>

#include "vspec/attention/flash_stub.h"

void vspec_flash_attention_stub(const float* q, const float* k, const float* v, size_t tokens, size_t head_dim, float* out) {
    if (!q || !k || !v || !out || tokens == 0U || head_dim == 0U) {
        return;
    }

    (void)q;
    (void)k;

    for (size_t i = 0; i < head_dim; ++i) {
        out[i] = v[i];
    }
}
