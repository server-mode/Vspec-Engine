#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "vspec/compat/pytorch_loader.h"
#include "vspec/compat/safetensors_parser.h"
#include "vspec/compat/graph_rewrite.h"

static void print_model(const char* tag, const VspecCompatModel* m) {
    printf("%s tensors=%zu\n", tag, m->tensor_count);
    for (size_t i = 0; i < m->tensor_count; ++i) {
        const VspecCompatTensorInfo* t = &m->tensors[i];
        printf("  - %s dtype=%s shape=[", t->name, t->dtype);
        for (size_t d = 0; d < t->ndim; ++d) {
            printf("%zu%s", t->shape[d], (d + 1U < t->ndim) ? "," : "");
        }
        printf("] offs=[%llu,%llu]\n",
            (unsigned long long)t->data_offset_start,
            (unsigned long long)t->data_offset_end);
    }
}

int main(void) {
    const char* manifest_path = "sample_weights.vpt";
    {
        FILE* f = fopen(manifest_path, "wb");
        if (!f) return 1;
        fputs("embed.weight|F16|32000,4096\n", f);
        fputs("lm_head.weight|F16|32000,4096\n", f);
        fclose(f);
    }

    VspecCompatModel m1;
    if (!vspec_pytorch_load_manifest(manifest_path, &m1)) {
        printf("pytorch manifest parse failed\n");
        return 1;
    }
    print_model("pytorch-manifest", &m1);

    const char* st_path = "sample.safetensors";
    {
        const char* header =
            "{\"linear.weight\":{\"dtype\":\"F16\",\"shape\":[4,4],\"data_offsets\":[0,32]},"
            "\"linear.bias\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[32,40]}}";
        const uint64_t hlen = (uint64_t)strlen(header);

        uint8_t len_le[8];
        for (size_t i = 0; i < 8; ++i) {
            len_le[i] = (uint8_t)((hlen >> (8U * i)) & 0xFFU);
        }

        FILE* f = fopen(st_path, "wb");
        if (!f) return 1;
        fwrite(len_le, 1, 8, f);
        fwrite(header, 1, (size_t)hlen, f);
        uint8_t dummy[40] = {0};
        fwrite(dummy, 1, sizeof(dummy), f);
        fclose(f);
    }

    VspecCompatModel m2;
    if (!vspec_safetensors_parse_header_file(st_path, &m2)) {
        printf("safetensors parse failed\n");
        return 1;
    }
    print_model("safetensors", &m2);

    VspecGraph g;
    vspec_graph_init(&g);
    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ACT_RELU, 2, 0, 3);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 3, 4, 5);

    VspecGraphRewriteStats st = {0};
    vspec_graph_rewrite_fuse_linear_relu(&g, &st);
    vspec_graph_rewrite_compact(&g, &st);

    printf("rewrite fused=%zu removed=%zu final_nodes=%zu\n",
        st.fused_linear_relu, st.removed_invalid_nodes, g.node_count);

    for (size_t i = 0; i < g.node_count; ++i) {
        const VspecNode* n = vspec_graph_get_node(&g, i);
        printf("node[%zu] op=%d inA=%u inB=%u out=%u\n",
            i, (int)n->op, n->input_a, n->input_b, n->output);
    }

    return 0;
}
