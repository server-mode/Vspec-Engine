#include <stdio.h>

#include "vspec/graph/ir.h"
#include "vspec/graph/optimizer.h"

int main(void) {
    VspecGraph g;
    vspec_graph_init(&g);
    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ACT_RELU, 2, 0, 3);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 3, 4, 5);

    const uint32_t live[] = {5};
    VspecGraphOptStats st = {0};
    vspec_graph_optimize_dead_nodes(&g, live, 1, &st);

    printf("after optimize: nodes=%zu removed=%zu\n", g.node_count, st.removed_dead_nodes);
    for (size_t i = 0; i < g.node_count; ++i) {
        const VspecNode* n = vspec_graph_get_node(&g, i);
        printf("node[%zu] op=%d out=%u\n", i, (int)n->op, n->output);
    }

    return 0;
}
