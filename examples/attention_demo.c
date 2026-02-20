#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vspec/attention/kv_cache.h"
#include "vspec/kernel/context.h"
#include "vspec/runtime/runtime.h"

int main(void) {
    const size_t heads = 2;
    const size_t head_dim = 4;
    const size_t max_tokens = 8;

    const size_t kv_elems = max_tokens * heads * head_dim;
    float* key_buf = (float*)malloc(kv_elems * sizeof(float));
    float* value_buf = (float*)malloc(kv_elems * sizeof(float));
    if (!key_buf || !value_buf) {
        free(key_buf);
        free(value_buf);
        return 1;
    }
    memset(key_buf, 0, kv_elems * sizeof(float));
    memset(value_buf, 0, kv_elems * sizeof(float));

    VspecKVCache cache;
    if (!vspec_kv_cache_init(&cache, key_buf, value_buf, max_tokens, heads, head_dim)) {
        free(key_buf);
        free(value_buf);
        return 1;
    }

    float k0[8] = {0.2f, 0.1f, 0.0f, -0.1f, 0.1f, 0.3f, -0.2f, 0.4f};
    float v0[8] = {0.5f, 0.0f, 0.1f, 0.0f, 0.2f, 0.3f, 0.1f, -0.2f};
    float k1[8] = {0.0f, 0.2f, 0.1f, 0.3f, -0.1f, 0.2f, 0.2f, 0.1f};
    float v1[8] = {0.1f, 0.2f, 0.4f, 0.2f, 0.0f, 0.1f, 0.3f, 0.2f};

    vspec_kv_cache_append(&cache, k0, v0);
    vspec_kv_cache_append(&cache, k1, v1);

    float q[8] = {0.3f, 0.1f, -0.1f, 0.0f, 0.2f, -0.2f, 0.3f, 0.1f};
    float out[8] = {0};

    VspecKernelContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.input = q;
    ctx.weight = &cache;
    ctx.output = out;
    ctx.config.m = heads;
    ctx.config.k = head_dim;

    vspec_runtime_init_default();
    vspec_attention_forward(&ctx);

    printf("tokens=%zu output:\n", cache.current_tokens);
    for (size_t h = 0; h < heads; ++h) {
        printf("head %zu: ", h);
        for (size_t d = 0; d < head_dim; ++d) {
            printf("%7.4f ", out[h * head_dim + d]);
        }
        printf("\n");
    }

    free(key_buf);
    free(value_buf);

    return 0;
}
