#ifndef VSPEC_COMPAT_GRAPH_REWRITE_H
#define VSPEC_COMPAT_GRAPH_REWRITE_H

#include <stddef.h>

#include "vspec/graph/ir.h"

typedef struct VspecGraphRewriteStats {
    size_t fused_linear_relu;
    size_t removed_invalid_nodes;
} VspecGraphRewriteStats;

void vspec_graph_rewrite_fuse_linear_relu(VspecGraph* graph, VspecGraphRewriteStats* stats);
void vspec_graph_rewrite_compact(VspecGraph* graph, VspecGraphRewriteStats* stats);

#endif
