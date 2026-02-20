#include <stdio.h>

#include "vspec/graph/ir.h"

int main(void) {
    VspecGraph g;
    vspec_graph_init(&g);

    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ACT_RELU, 2, 0, 3);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 3, 4, 5);

    printf("graph nodes=%zu\n", g.node_count);
    for (size_t i = 0; i < g.node_count; ++i) {
        const VspecNode* n = vspec_graph_get_node(&g, i);
        printf("node[%zu]: id=%u op=%d inA=%u inB=%u out=%u\n",
            i, n->id, (int)n->op, n->input_a, n->input_b, n->output);
    }

    return 0;
}
