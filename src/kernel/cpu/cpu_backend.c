#include <stddef.h>

#include "vspec/attention/attention.h"
#include "vspec/attention/kv_cache.h"
#include "vspec/kernel/backend.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/quant.h"

static void cpu_launch_linear(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    if (ctx->qmeta.type != VSPEC_QUANT_INT4 || !ctx->qmeta.scales) {
        return;
    }

    vspec_int4_matmul_ref_f32_q4(
        (const float*)ctx->input,
        ctx->config.m,
        ctx->config.k,
        (const uint8_t*)ctx->weight,
        ctx->config.n,
        ctx->qmeta.scales,
        (float*)ctx->output
    );
}

static void cpu_launch_attention(VspecKernelContext* ctx) {
    if (!ctx || !ctx->input || !ctx->weight || !ctx->output) {
        return;
    }

    const VspecKVCache* cache = (const VspecKVCache*)ctx->weight;
    vspec_attention_ref_single_query(
        (const float*)ctx->input,
        cache,
        (float*)ctx->output
    );
}

VspecBackend vspec_make_cpu_backend(void) {
    VspecBackend backend;
    backend.name = "cpu-ref";
    backend.launch_linear = cpu_launch_linear;
    backend.launch_attention = cpu_launch_attention;
    return backend;
}
