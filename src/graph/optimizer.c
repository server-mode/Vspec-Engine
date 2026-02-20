#include <string.h>

#include "vspec/graph/optimizer.h"

static int output_is_live(uint32_t out, const uint32_t* live, size_t live_count) {
    for (size_t i = 0; i < live_count; ++i) {
        if (live[i] == out) {
            return 1;
        }
    }
    return 0;
}

void vspec_graph_optimize_dead_nodes(
    VspecGraph* graph,
    const uint32_t* live_outputs,
    size_t live_count,
    VspecGraphOptStats* stats
) {
    if (!graph || !live_outputs || live_count == 0U) {
        return;
    }

    if (stats) {
        stats->removed_dead_nodes = 0U;
    }

    for (size_t i = 0; i < graph->node_count; ++i) {
        VspecNode* n = &graph->nodes[i];
        if (n->op == VSPEC_OP_INVALID) {
            continue;
        }

        if (!output_is_live(n->output, live_outputs, live_count)) {
            n->op = VSPEC_OP_INVALID;
            if (stats) {
                stats->removed_dead_nodes += 1U;
            }
        }
    }

    size_t write = 0U;
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].op == VSPEC_OP_INVALID) {
            continue;
        }
        if (write != i) {
            graph->nodes[write] = graph->nodes[i];
            graph->nodes[write].id = (uint32_t)write;
        }
        write += 1U;
    }

    graph->node_count = write;
}
