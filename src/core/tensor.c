#include "vspec/core/tensor.h"

size_t vspec_tensor_numel(const VspecTensor* tensor) {
    if (!tensor || tensor->ndim == 0) {
        return 0;
    }

    size_t numel = 1;
    for (size_t i = 0; i < tensor->ndim; ++i) {
        numel *= tensor->shape[i];
    }
    return numel;
}

size_t vspec_dtype_size(VspecDType dtype) {
    switch (dtype) {
        case VSPEC_DTYPE_F32:
            return sizeof(float);
        case VSPEC_DTYPE_Q4_PACKED:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}
