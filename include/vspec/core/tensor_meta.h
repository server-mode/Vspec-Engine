#ifndef VSPEC_CORE_TENSOR_META_H
#define VSPEC_CORE_TENSOR_META_H

#include <stddef.h>

#include "vspec/core/tensor.h"
#include "vspec/version.h"

typedef struct VspecTensorMeta {
    unsigned int version;
    VspecDType dtype;
    size_t ndim;
    size_t shape[4];
    size_t stride[4];
} VspecTensorMeta;

void vspec_tensor_meta_from_tensor(const VspecTensor* tensor, VspecTensorMeta* meta);

#endif
