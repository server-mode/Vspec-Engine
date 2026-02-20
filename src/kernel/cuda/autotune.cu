#include <stddef.h>

#include "vspec/kernel/cuda_fused.h"

typedef struct VspecCudaTuneResult {
    unsigned int block_x;
    unsigned int block_y;
} VspecCudaTuneResult;

VspecCudaTuneResult vspec_cuda_autotune_linear(void) {
    VspecCudaTuneResult r = {16, 16};
    return r;
}
