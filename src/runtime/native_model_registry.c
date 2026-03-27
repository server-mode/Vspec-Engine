#include "vspec/runtime/native_model_registry.h"

#include <string.h>

static int has_tensor_name(const VspecCompatModel* model, const char* needle) {
    if (!model || !needle || !needle[0]) {
        return 0;
    }
    for (size_t i = 0; i < model->tensor_count; ++i) {
        if (strcmp(model->tensors[i].name, needle) == 0) {
            return 1;
        }
    }
    return 0;
}

const char* vspec_native_model_family_name(VspecNativeModelFamily family) {
    switch (family) {
        case VSPEC_NATIVE_MODEL_QWEN:
            return "qwen";
        case VSPEC_NATIVE_MODEL_LLAMA:
            return "llama";
        case VSPEC_NATIVE_MODEL_GPT2:
            return "gpt2";
        default:
            return "unknown";
    }
}

VspecNativeModelFamily vspec_native_model_detect_family(const VspecCompatModel* model) {
    if (!model || model->tensor_count == 0U) {
        return VSPEC_NATIVE_MODEL_UNKNOWN;
    }

    if (has_tensor_name(model, "model.layers.0.self_attn.q_proj.weight")
        && has_tensor_name(model, "model.layers.0.mlp.gate_proj.weight")) {
        return VSPEC_NATIVE_MODEL_QWEN;
    }

    if (has_tensor_name(model, "model.layers.0.self_attn.q_proj.weight")
        && has_tensor_name(model, "model.layers.0.mlp.up_proj.weight")) {
        return VSPEC_NATIVE_MODEL_LLAMA;
    }

    if (has_tensor_name(model, "transformer.wte.weight")
        && has_tensor_name(model, "transformer.h.0.attn.c_attn.weight")) {
        return VSPEC_NATIVE_MODEL_GPT2;
    }

    return VSPEC_NATIVE_MODEL_UNKNOWN;
}

int vspec_native_model_family_supported(VspecNativeModelFamily family) {
    switch (family) {
        case VSPEC_NATIVE_MODEL_QWEN:
        case VSPEC_NATIVE_MODEL_LLAMA:
        case VSPEC_NATIVE_MODEL_GPT2:
            return 1;
        default:
            return 0;
    }
}
