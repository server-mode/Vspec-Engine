#include "vspec/graph/ir.h"

#include <stdio.h>
#include <string.h>

static int write_err(char* error_buf, size_t error_buf_len, const char* msg) {
    if (error_buf && error_buf_len > 0U) {
        (void)snprintf(error_buf, error_buf_len, "%s", msg ? msg : "graph_error");
    }
    return 0;
}

static int is_reserved_input(uint32_t tid) {
    return tid <= 1U;
}

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

int vspec_graph_find_producer(const VspecGraph* graph, uint32_t output_tensor_id) {
    if (!graph) {
        return -1;
    }
    for (size_t i = 0U; i < graph->node_count; ++i) {
        const VspecNode* n = &graph->nodes[i];
        if (n->op != VSPEC_OP_INVALID && n->output == output_tensor_id) {
            return (int)i;
        }
    }
    return -1;
}

int vspec_graph_validate(const VspecGraph* graph, char* error_buf, size_t error_buf_len) {
    if (!graph) {
        return write_err(error_buf, error_buf_len, "graph_null");
    }
    if (graph->node_count > VSPEC_GRAPH_MAX_NODES) {
        return write_err(error_buf, error_buf_len, "graph_node_count_exceeds_capacity");
    }

    for (size_t i = 0U; i < graph->node_count; ++i) {
        const VspecNode* n = &graph->nodes[i];
        if (n->op == VSPEC_OP_INVALID) {
            return write_err(error_buf, error_buf_len, "graph_contains_invalid_node");
        }
        if ((size_t)n->id != i) {
            return write_err(error_buf, error_buf_len, "graph_node_id_mismatch");
        }

        for (size_t j = i + 1U; j < graph->node_count; ++j) {
            if (graph->nodes[j].op != VSPEC_OP_INVALID && graph->nodes[j].output == n->output) {
                return write_err(error_buf, error_buf_len, "graph_duplicate_output_tensor");
            }
        }

        if (!is_reserved_input(n->input_a) && vspec_graph_find_producer(graph, n->input_a) < 0) {
            return write_err(error_buf, error_buf_len, "graph_missing_input_a_producer");
        }
        if (n->input_b != 0U && !is_reserved_input(n->input_b) && vspec_graph_find_producer(graph, n->input_b) < 0) {
            return write_err(error_buf, error_buf_len, "graph_missing_input_b_producer");
        }
    }

    return 1;
}

size_t vspec_graph_build_topological_order(const VspecGraph* graph, uint32_t* node_ids_out, size_t capacity) {
    if (!graph || !node_ids_out || capacity == 0U || graph->node_count == 0U) {
        return 0U;
    }

    uint32_t indeg[VSPEC_GRAPH_MAX_NODES];
    for (size_t i = 0U; i < graph->node_count; ++i) {
        indeg[i] = 0U;
    }

    for (size_t i = 0U; i < graph->node_count; ++i) {
        const VspecNode* n = &graph->nodes[i];
        if (n->op == VSPEC_OP_INVALID) {
            continue;
        }
        const int pa = vspec_graph_find_producer(graph, n->input_a);
        if (pa >= 0) {
            indeg[i] += 1U;
        }
        const int pb = vspec_graph_find_producer(graph, n->input_b);
        if (pb >= 0 && pb != pa) {
            indeg[i] += 1U;
        }
    }

    uint32_t queue[VSPEC_GRAPH_MAX_NODES];
    size_t qh = 0U;
    size_t qt = 0U;
    for (size_t i = 0U; i < graph->node_count; ++i) {
        if (indeg[i] == 0U) {
            queue[qt++] = (uint32_t)i;
        }
    }

    size_t written = 0U;
    while (qh < qt && written < capacity) {
        const uint32_t cur = queue[qh++];
        node_ids_out[written++] = cur;
        const uint32_t out_tid = graph->nodes[cur].output;

        for (size_t j = 0U; j < graph->node_count; ++j) {
            const VspecNode* n = &graph->nodes[j];
            if (n->op == VSPEC_OP_INVALID) {
                continue;
            }
            if (n->input_a == out_tid || n->input_b == out_tid) {
                if (indeg[j] > 0U) {
                    indeg[j] -= 1U;
                    if (indeg[j] == 0U) {
                        queue[qt++] = (uint32_t)j;
                    }
                }
            }
        }
    }

    return written;
}
