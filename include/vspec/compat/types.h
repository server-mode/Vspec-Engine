#ifndef VSPEC_COMPAT_TYPES_H
#define VSPEC_COMPAT_TYPES_H

#include <stddef.h>
#include <stdint.h>

#define VSPEC_COMPAT_MAX_TENSORS 512
#define VSPEC_COMPAT_NAME_MAX 128
#define VSPEC_COMPAT_MAX_DIMS 8

typedef struct VspecCompatTensorInfo {
    char name[VSPEC_COMPAT_NAME_MAX];
    char dtype[16];
    size_t ndim;
    size_t shape[VSPEC_COMPAT_MAX_DIMS];
    uint64_t data_offset_start;
    uint64_t data_offset_end;
} VspecCompatTensorInfo;

typedef struct VspecCompatModel {
    VspecCompatTensorInfo tensors[VSPEC_COMPAT_MAX_TENSORS];
    size_t tensor_count;
} VspecCompatModel;

void vspec_compat_model_init(VspecCompatModel* model);

#endif
