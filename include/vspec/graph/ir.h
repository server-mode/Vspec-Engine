#ifndef VSPEC_GRAPH_IR_H
#define VSPEC_GRAPH_IR_H

#include <stddef.h>
#include <stdint.h>

#define VSPEC_GRAPH_MAX_NODES 256

typedef enum VspecOpType {
    VSPEC_OP_INVALID = 0,
    VSPEC_OP_LINEAR = 1,
    VSPEC_OP_ATTENTION = 2,
    VSPEC_OP_ACT_RELU = 3,
    VSPEC_OP_LINEAR_RELU_FUSED = 4,
    VSPEC_OP_ATTENTION_FUSED = 5
} VspecOpType;

typedef struct VspecNode {
    uint32_t id;
    VspecOpType op;
    uint32_t input_a;
    uint32_t input_b;
    uint32_t output;
} VspecNode;

typedef struct VspecGraph {
    VspecNode nodes[VSPEC_GRAPH_MAX_NODES];
    size_t node_count;
} VspecGraph;

void vspec_graph_init(VspecGraph* graph);
int vspec_graph_add_node(VspecGraph* graph, VspecOpType op, uint32_t input_a, uint32_t input_b, uint32_t output);
const VspecNode* vspec_graph_get_node(const VspecGraph* graph, size_t index);

#endif
