#ifndef VSPEC_GRAPH_OPTIMIZER_H
#define VSPEC_GRAPH_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/graph/ir.h"

typedef struct VspecGraphOptStats {
    size_t removed_dead_nodes;
    size_t fused_attention;
    size_t fused_rmsnorm_linear;
    size_t fused_silu_mul_linear;
} VspecGraphOptStats;

void vspec_graph_optimize_fuse_attention(VspecGraph* graph, VspecGraphOptStats* stats);
void vspec_graph_optimize_fuse_rmsnorm_linear(VspecGraph* graph, VspecGraphOptStats* stats);
void vspec_graph_optimize_fuse_silu_mul_linear(VspecGraph* graph, VspecGraphOptStats* stats);

void vspec_graph_optimize_all(
    VspecGraph* graph,
    const uint32_t* live_outputs,
    size_t live_count,
    VspecGraphOptStats* stats
);

void vspec_graph_optimize_dead_nodes(
    VspecGraph* graph,
    const uint32_t* live_outputs,
    size_t live_count,
    VspecGraphOptStats* stats
);

#endif
