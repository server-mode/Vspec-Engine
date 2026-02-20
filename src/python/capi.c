#include <stdio.h>
#include <string.h>

#include "vspec/python/capi.h"
#include "vspec/compat/pytorch_loader.h"
#include "vspec/compat/safetensors_parser.h"
#include "vspec/compat/graph_rewrite.h"
#include "vspec/graph/ir.h"

const char* vspec_py_version(void) {
    return "0.1.0-week10";
}

int vspec_py_load_manifest_count(const char* path) {
    VspecCompatModel m;
    if (!vspec_pytorch_load_manifest(path, &m)) {
        return -1;
    }
    return (int)m.tensor_count;
}

int vspec_py_parse_safetensors_count(const char* path) {
    VspecCompatModel m;
    if (!vspec_safetensors_parse_header_file(path, &m)) {
        return -1;
    }
    return (int)m.tensor_count;
}

int vspec_py_rewrite_demo_final_nodes(void) {
    VspecGraph g;
    VspecGraphRewriteStats st = {0};

    vspec_graph_init(&g);
    vspec_graph_add_node(&g, VSPEC_OP_LINEAR, 0, 1, 2);
    vspec_graph_add_node(&g, VSPEC_OP_ACT_RELU, 2, 0, 3);
    vspec_graph_add_node(&g, VSPEC_OP_ATTENTION, 3, 4, 5);

    vspec_graph_rewrite_fuse_linear_relu(&g, &st);
    vspec_graph_rewrite_compact(&g, &st);

    return (int)g.node_count;
}

int vspec_py_generate(const char* prompt, char* out, size_t out_size) {
    if (!out || out_size == 0U) {
        return 0;
    }

    const char* p = prompt ? prompt : "";
    const int n = snprintf(out, out_size, "vspec> %s", p);
    if (n < 0) {
        out[0] = '\0';
        return 0;
    }
    return 1;
}
