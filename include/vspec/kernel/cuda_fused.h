#ifndef VSPEC_KERNEL_CUDA_FUSED_H
#define VSPEC_KERNEL_CUDA_FUSED_H

#include "vspec/kernel/context.h"

#ifdef __cplusplus
extern "C" {
#endif

int vspec_cuda_fused_available(void);
void vspec_cuda_launch_fused_linear_int4(VspecKernelContext* ctx);
void vspec_cuda_launch_fused_linear_int3(VspecKernelContext* ctx);
void vspec_cuda_launch_attention_fused(VspecKernelContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
