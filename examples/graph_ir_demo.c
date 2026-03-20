#include <stdio.h>

#include "vspec/graph/ir.h"

int main(void) {
    VspecGraph g;
    vspec_graph_init(&g);

    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ACT_RELU, 2, 0, 3);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 3, 1, 5);

    char err[128] = {0};
    int ok = vspec_graph_validate(&g, err, sizeof(err));
    printf("graph validate=%d err=%s\n", ok, ok ? "none" : err);

    uint32_t topo[VSPEC_GRAPH_MAX_NODES] = {0};
    size_t topo_n = vspec_graph_build_topological_order(&g, topo, VSPEC_GRAPH_MAX_NODES);
    printf("graph topo_count=%zu\n", topo_n);
    for (size_t i = 0; i < topo_n; ++i) {
        printf("topo[%zu]=%u\n", i, topo[i]);
    }

    printf("graph nodes=%zu\n", g.node_count);
    for (size_t i = 0; i < g.node_count; ++i) {
        const VspecNode* n = vspec_graph_get_node(&g, i);
        printf("node[%zu]: id=%u op=%d inA=%u inB=%u out=%u\n",
            i, n->id, (int)n->op, n->input_a, n->input_b, n->output);
    }

    return 0;
}
