#ifndef VSPEC_KERNEL_CUDA_ULTIMATE_H
#define VSPEC_KERNEL_CUDA_ULTIMATE_H

#include "vspec/kernel/context.h"

#ifdef __cplusplus
extern "C" {
#endif

int vspec_cuda_ultimate_tensorcore_available(void);
void vspec_cuda_launch_linear_ultimate(VspecKernelContext* ctx);

#ifdef __cplusplus
}
#endif

#endif