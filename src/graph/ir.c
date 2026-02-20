#include "vspec/graph/ir.h"

void vspec_graph_init(VspecGraph* graph) {
    if (!graph) {
        return;
    }
    graph->node_count = 0U;
}

int vspec_graph_add_node(VspecGraph* graph, VspecOpType op, uint32_t input_a, uint32_t input_b, uint32_t output) {
    if (!graph || graph->node_count >= VSPEC_GRAPH_MAX_NODES || op == VSPEC_OP_INVALID) {
        return 0;
    }

    VspecNode* n = &graph->nodes[graph->node_count];
    n->id = (uint32_t)graph->node_count;
    n->op = op;
    n->input_a = input_a;
    n->input_b = input_b;
    n->output = output;

    graph->node_count += 1U;
    return 1;
}

const VspecNode* vspec_graph_get_node(const VspecGraph* graph, size_t index) {
    if (!graph || index >= graph->node_count) {
        return 0;
    }
    return &graph->nodes[index];
}
