#include <string.h>

#include "vspec/core/tensor_meta.h"

void vspec_tensor_meta_from_tensor(const VspecTensor* tensor, VspecTensorMeta* meta) {
    if (!tensor || !meta) {
        return;
    }

    meta->version = VSPEC_IR_VERSION;
    meta->dtype = tensor->dtype;
    meta->ndim = tensor->ndim;
    memcpy(meta->shape, tensor->shape, sizeof(meta->shape));
    memcpy(meta->stride, tensor->stride, sizeof(meta->stride));
}
