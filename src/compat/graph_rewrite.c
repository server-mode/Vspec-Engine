#include "vspec/compat/graph_rewrite.h"

void vspec_graph_rewrite_fuse_linear_relu(VspecGraph* graph, VspecGraphRewriteStats* stats) {
    if (!graph) {
        return;
    }

    if (stats) {
        stats->fused_linear_relu = 0U;
        stats->removed_invalid_nodes = 0U;
    }

    if (graph->node_count < 2U) {
        return;
    }

    for (size_t i = 0; i + 1U < graph->node_count; ++i) {
        VspecNode* a = &graph->nodes[i];
        VspecNode* b = &graph->nodes[i + 1U];

        if (a->op == VSPEC_OP_LINEAR && b->op == VSPEC_OP_ACT_RELU && b->input_a == a->output) {
            a->op = VSPEC_OP_LINEAR_RELU_FUSED;
            a->output = b->output;
            b->op = VSPEC_OP_INVALID;
            if (stats) {
                stats->fused_linear_relu += 1U;
            }
        }
    }
}

void vspec_graph_rewrite_compact(VspecGraph* graph, VspecGraphRewriteStats* stats) {
    if (!graph) {
        return;
    }

    size_t write_idx = 0U;
    size_t removed = 0U;

    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].op == VSPEC_OP_INVALID) {
            removed += 1U;
            continue;
        }

        if (write_idx != i) {
            graph->nodes[write_idx] = graph->nodes[i];
            graph->nodes[write_idx].id = (uint32_t)write_idx;
        }
        write_idx += 1U;
    }

    graph->node_count = write_idx;

    if (stats) {
        stats->removed_invalid_nodes += removed;
    }
}
