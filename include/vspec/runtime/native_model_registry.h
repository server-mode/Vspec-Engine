#ifndef VSPEC_RUNTIME_NATIVE_MODEL_REGISTRY_H
#define VSPEC_RUNTIME_NATIVE_MODEL_REGISTRY_H

#include "vspec/compat/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VspecNativeModelFamily {
    VSPEC_NATIVE_MODEL_UNKNOWN = 0,
    VSPEC_NATIVE_MODEL_QWEN,
    VSPEC_NATIVE_MODEL_LLAMA,
    VSPEC_NATIVE_MODEL_GPT2,
} VspecNativeModelFamily;

const char* vspec_native_model_family_name(VspecNativeModelFamily family);
VspecNativeModelFamily vspec_native_model_detect_family(const VspecCompatModel* model);
int vspec_native_model_family_supported(VspecNativeModelFamily family);

#ifdef __cplusplus
}
#endif

#endif
