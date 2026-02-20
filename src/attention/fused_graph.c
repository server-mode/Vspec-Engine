#include "vspec/attention/fused_graph.h"

size_t vspec_attention_graph_fuse(VspecGraph* graph) {
    if (!graph) {
        return 0U;
    }

    size_t fused = 0U;
    for (size_t i = 0; i + 1U < graph->node_count; ++i) {
        VspecNode* a = &graph->nodes[i];
        VspecNode* b = &graph->nodes[i + 1U];
        if (a->op == VSPEC_OP_LINEAR && b->op == VSPEC_OP_ATTENTION && b->input_a == a->output) {
            b->op = VSPEC_OP_ATTENTION_FUSED;
            fused += 1U;
        }
    }

    return fused;
}
