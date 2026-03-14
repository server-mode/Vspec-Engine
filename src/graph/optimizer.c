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

static int find_node_by_output(const VspecGraph* graph, uint32_t output) {
    if (!graph) {
        return -1;
    }
    for (size_t i = 0U; i < graph->node_count; ++i) {
        if (graph->nodes[i].op != VSPEC_OP_INVALID && graph->nodes[i].output == output) {
            return (int)i;
        }
    }
    return -1;
}

void vspec_graph_optimize_fuse_attention(VspecGraph* graph, VspecGraphOptStats* stats) {
    if (!graph) {
        return;
    }

    for (size_t i = 0U; i + 1U < graph->node_count; ++i) {
        VspecNode* a = &graph->nodes[i];
        VspecNode* b = &graph->nodes[i + 1U];
        if (a->op == VSPEC_OP_ATTENTION && b->op == VSPEC_OP_LINEAR && b->input_a == a->output) {
            a->op = VSPEC_OP_ATTENTION_FUSED;
            a->output = b->output;
            b->op = VSPEC_OP_INVALID;
            if (stats) {
                stats->fused_attention += 1U;
            }
        }
    }
}

void vspec_graph_optimize_fuse_rmsnorm_linear(VspecGraph* graph, VspecGraphOptStats* stats) {
    if (!graph) {
        return;
    }

    for (size_t i = 0U; i + 1U < graph->node_count; ++i) {
        VspecNode* a = &graph->nodes[i];
        VspecNode* b = &graph->nodes[i + 1U];
        if (a->op == VSPEC_OP_RMSNORM && b->op == VSPEC_OP_LINEAR && b->input_a == a->output) {
            a->op = VSPEC_OP_RMSNORM_LINEAR_FUSED;
            a->output = b->output;
            b->op = VSPEC_OP_INVALID;
            if (stats) {
                stats->fused_rmsnorm_linear += 1U;
            }
        }
    }
}

void vspec_graph_optimize_fuse_silu_mul_linear(VspecGraph* graph, VspecGraphOptStats* stats) {
    if (!graph) {
        return;
    }
    if (graph->node_count < 3U) {
        return;
    }

    for (size_t i = 0U; i + 2U < graph->node_count; ++i) {
        VspecNode* a = &graph->nodes[i];
        VspecNode* b = &graph->nodes[i + 1U];
        VspecNode* c = &graph->nodes[i + 2U];

        if (a->op == VSPEC_OP_ACT_SILU &&
            b->op == VSPEC_OP_MUL && b->input_a == a->output &&
            c->op == VSPEC_OP_LINEAR && c->input_a == b->output) {
            a->op = VSPEC_OP_SILU_MUL_LINEAR_FUSED;
            a->output = c->output;
            b->op = VSPEC_OP_INVALID;
            c->op = VSPEC_OP_INVALID;
            if (stats) {
                stats->fused_silu_mul_linear += 1U;
            }
        }
    }
}

void vspec_graph_optimize_all(
    VspecGraph* graph,
    const uint32_t* live_outputs,
    size_t live_count,
    VspecGraphOptStats* stats
) {
    if (!graph) {
        return;
    }

    if (stats) {
        (void)memset(stats, 0, sizeof(*stats));
    }

    vspec_graph_optimize_fuse_attention(graph, stats);
    vspec_graph_optimize_fuse_rmsnorm_linear(graph, stats);
    vspec_graph_optimize_fuse_silu_mul_linear(graph, stats);
    vspec_graph_optimize_dead_nodes(graph, live_outputs, live_count, stats);
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

    int live_nodes[VSPEC_GRAPH_MAX_NODES];
    for (size_t i = 0U; i < VSPEC_GRAPH_MAX_NODES; ++i) {
        live_nodes[i] = 0;
    }

    for (size_t i = 0U; i < graph->node_count; ++i) {
        VspecNode* n = &graph->nodes[i];
        if (n->op == VSPEC_OP_INVALID) {
            continue;
        }
        if (output_is_live(n->output, live_outputs, live_count)) {
            live_nodes[i] = 1;
        }
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (size_t i = 0U; i < graph->node_count; ++i) {
            if (!live_nodes[i]) {
                continue;
            }

            const VspecNode* n = &graph->nodes[i];
            const int pa = find_node_by_output(graph, n->input_a);
            if (pa >= 0 && !live_nodes[(size_t)pa]) {
                live_nodes[(size_t)pa] = 1;
                changed = 1;
            }

            const int pb = find_node_by_output(graph, n->input_b);
            if (pb >= 0 && !live_nodes[(size_t)pb]) {
                live_nodes[(size_t)pb] = 1;
                changed = 1;
            }
        }
    }

    if (stats) {
        stats->removed_dead_nodes = 0U;
    }

    for (size_t i = 0; i < graph->node_count; ++i) {
        VspecNode* n = &graph->nodes[i];
        if (n->op == VSPEC_OP_INVALID) {
            continue;
        }

        if (!live_nodes[i]) {
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
