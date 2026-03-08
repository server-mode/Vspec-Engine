#include <string.h>
#include <stdio.h>

#include "vspec/compat/weight_mapper.h"

int vspec_weight_map_identity(const VspecCompatModel* in, VspecCompatModel* out) {
    if (!in || !out) {
        return 0;
    }

    *out = *in;
    return 1;
}

static int copy_name(const char* src, char* out_name, size_t out_name_size) {
    if (!src || !out_name || out_name_size == 0U) {
        return 0;
    }
    out_name[0] = '\0';
    (void)snprintf(out_name, out_name_size, "%s", src);
    return 1;
}

static int match_layer_pattern(const char* raw_name, const char* pattern, int* layer_idx) {
    int consumed = 0;
    if (!raw_name || !pattern || !layer_idx) {
        return 0;
    }
    if (sscanf(raw_name, pattern, layer_idx, &consumed) == 1 && raw_name[consumed] == '\0') {
        return 1;
    }
    return 0;
}

int vspec_weight_canonical_name(const char* raw_name, char* out_name, size_t out_name_size) {
    int layer_idx = 0;
    if (!raw_name || !out_name || out_name_size == 0U) {
        return 0;
    }

    if (strcmp(raw_name, "tok_embeddings.weight") == 0 || strcmp(raw_name, "transformer.wte.weight") == 0) {
        return copy_name("model.embed_tokens.weight", out_name, out_name_size);
    }
    if (strcmp(raw_name, "output.weight") == 0 || strcmp(raw_name, "transformer.lm_head.weight") == 0) {
        return copy_name("lm_head.weight", out_name, out_name_size);
    }
    if (strcmp(raw_name, "norm.weight") == 0 || strcmp(raw_name, "transformer.ln_f.weight") == 0) {
        return copy_name("model.norm.weight", out_name, out_name_size);
    }

    if (match_layer_pattern(raw_name, "model.layers.%d.attention.wq.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.attention.wq.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.self_attn.q_proj.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.attention.wk.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.attention.wk.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.self_attn.k_proj.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.attention.wv.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.attention.wv.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.self_attn.v_proj.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.attention.wo.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.attention.wo.weight%n", &layer_idx) || match_layer_pattern(raw_name, "model.layers.%d.self_attn.out_proj.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.self_attn.o_proj.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.attention_norm.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.attention_norm.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.input_layernorm.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.ffn_norm.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.ffn_norm.weight%n", &layer_idx) || match_layer_pattern(raw_name, "model.layers.%d.mlp_norm.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.post_attention_layernorm.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.feed_forward.w1.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.feed_forward.w1.weight%n", &layer_idx) || match_layer_pattern(raw_name, "model.layers.%d.mlp.w1.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.mlp.gate_proj.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.feed_forward.w2.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.feed_forward.w2.weight%n", &layer_idx) || match_layer_pattern(raw_name, "model.layers.%d.mlp.w2.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.mlp.down_proj.weight", layer_idx);
        return 1;
    }
    if (match_layer_pattern(raw_name, "model.layers.%d.feed_forward.w3.weight%n", &layer_idx) || match_layer_pattern(raw_name, "layers.%d.feed_forward.w3.weight%n", &layer_idx) || match_layer_pattern(raw_name, "model.layers.%d.mlp.w3.weight%n", &layer_idx)) {
        (void)snprintf(out_name, out_name_size, "model.layers.%d.mlp.up_proj.weight", layer_idx);
        return 1;
    }

    return copy_name(raw_name, out_name, out_name_size);
}
