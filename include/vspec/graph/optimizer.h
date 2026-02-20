#ifndef VSPEC_GRAPH_OPTIMIZER_H
#define VSPEC_GRAPH_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/graph/ir.h"

typedef struct VspecGraphOptStats {
    size_t removed_dead_nodes;
} VspecGraphOptStats;

void vspec_graph_optimize_dead_nodes(
    VspecGraph* graph,
    const uint32_t* live_outputs,
    size_t live_count,
    VspecGraphOptStats* stats
);

#endif
