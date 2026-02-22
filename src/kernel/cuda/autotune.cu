#include <stddef.h>
#include <cuda_runtime.h>

#include "vspec/kernel/cuda_fused.h"

typedef struct VspecCudaTuneResult {
    unsigned int block_x;
    unsigned int block_y;
} VspecCudaTuneResult;

VspecCudaTuneResult vspec_cuda_autotune_linear(void) {
    VspecCudaTuneResult r = {16, 16};
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return r;
    }
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return r;
    }

    if (prop.major >= 8) {
        r.block_x = 32;
        r.block_y = 8;
    } else if (prop.major >= 7) {
        r.block_x = 16;
        r.block_y = 16;
    } else {
        r.block_x = 16;
        r.block_y = 8;
    }
    return r;
}
