#include <stdio.h>

#include "vspec/compat/safetensors_parser.h"
#include "vspec/runtime/native_model_registry.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: native_model_probe <safetensors-file>\n");
        return 2;
    }

    VspecCompatModel model;
    if (!vspec_safetensors_parse_header_file(argv[1], &model)) {
        fprintf(stderr, "[native-probe] parse_header_failed file=%s\n", argv[1]);
        return 1;
    }

    const VspecNativeModelFamily family = vspec_native_model_detect_family(&model);
    printf("[native-probe] tensors=%zu\n", model.tensor_count);
    printf("[native-probe] family=%s\n", vspec_native_model_family_name(family));
    printf("[native-probe] supported=%s\n", vspec_native_model_family_supported(family) ? "yes" : "no");

    return vspec_native_model_family_supported(family) ? 0 : 3;
}
