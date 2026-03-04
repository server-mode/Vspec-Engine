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
void vspec_cuda_attention_fused_single_f32(
	const float* query,
	const float* keys,
	const float* values,
	size_t seq_len,
	size_t head_dim,
	float* output
);

void vspec_cuda_fused_linear_int4_device(
	const float* d_a,
	const unsigned char* d_b_packed,
	const float* d_scales,
	float* d_c,
	size_t m,
	size_t n,
	size_t k
);

void vspec_cuda_fused_linear_int3_device(
	const float* d_a,
	const unsigned char* d_b_packed,
	const float* d_scales,
	float* d_c,
	size_t m,
	size_t n,
	size_t k,
	int stochastic_rounding
);

void vspec_cuda_expand_int3_to_int4_device(
	const unsigned char* d_b_int3,
	unsigned char* d_b_int4,
	size_t n,
	size_t k
);

void vspec_cuda_launch_fused_linear_int3_storage(VspecKernelContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
