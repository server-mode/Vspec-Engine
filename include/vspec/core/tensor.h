#ifndef VSPEC_CORE_TENSOR_H
#define VSPEC_CORE_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum VspecDType {
    VSPEC_DTYPE_F32 = 0,
    VSPEC_DTYPE_Q4_PACKED = 1
} VspecDType;

typedef struct VspecTensor {
    void* data;
    VspecDType dtype;
    size_t ndim;
    size_t shape[4];
    size_t stride[4];
} VspecTensor;

size_t vspec_tensor_numel(const VspecTensor* tensor);
size_t vspec_dtype_size(VspecDType dtype);

#endif
