#include <stdio.h>
#include <string.h>

#include "vspec/attention/streaming.h"
#include "vspec/attention/kv_ring.h"
#include "vspec/attention/fused_graph.h"
#include "vspec/graph/ir.h"

int main(void) {
    const size_t tokens = 4;
    const size_t head_dim = 4;
    const size_t chunk = 2;

    float q[4] = {0.2f, 0.1f, -0.1f, 0.0f};
    float k[16] = {
        0.1f, 0.2f, 0.0f, 0.1f,
        0.2f, 0.1f, 0.0f, -0.1f,
        0.0f, 0.1f, 0.2f, 0.0f,
        -0.1f, 0.0f, 0.1f, 0.2f
    };
    float v[16] = {
        0.2f, 0.1f, 0.0f, 0.3f,
        0.1f, 0.2f, 0.1f, 0.0f,
        0.3f, -0.1f, 0.0f, 0.1f,
        0.0f, 0.1f, 0.2f, 0.2f
    };

    float out[4] = {0};
    vspec_attention_streaming_ref(q, k, v, tokens, head_dim, chunk, out);
    printf("streaming out: %.3f %.3f %.3f %.3f\n", out[0], out[1], out[2], out[3]);

    float key_buf[8] = {0};
    float val_buf[8] = {0};
    VspecKVCacheRing ring;
    vspec_kv_ring_init(&ring, key_buf, val_buf, 1, 1, 4);
    vspec_kv_ring_push(&ring, q, k);
    vspec_kv_ring_evict(&ring, 1);
    const float* k0 = vspec_kv_ring_key_at(&ring, 0, 0);
    printf("kv ring k0: %.3f %.3f %.3f %.3f\n", k0[0], k0[1], k0[2], k0[3]);

    VspecGraph g;
    vspec_graph_init(&g);
    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 2, 0, 3);
    const size_t fused = vspec_attention_graph_fuse(&g);
    printf("fused nodes=%zu op=%d\n", fused, (int)g.nodes[1].op);

    return 0;
}
