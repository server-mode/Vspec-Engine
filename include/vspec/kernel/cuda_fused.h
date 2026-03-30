#ifndef VSPEC_KERNEL_CUDA_FUSED_H
#define VSPEC_KERNEL_CUDA_FUSED_H

#include "vspec/kernel/context.h"

#ifdef __cplusplus
extern "C" {
#endif

int vspec_cuda_fused_available(void);
void vspec_cuda_launch_fused_linear_int4(VspecKernelContext* ctx);
void vspec_cuda_launch_fused_linear_int3(VspecKernelContext* ctx);
void vspec_cuda_launch_fused_linear_int2(VspecKernelContext* ctx);
void vspec_cuda_launch_attention_fused(VspecKernelContext* ctx);
void vspec_cuda_attention_fused_single_f32(
	const float* query,
	const float* keys,
	const float* values,
	size_t seq_len,
	size_t head_dim,
	float* output
);

void vspec_cuda_attention_flash_single_f32(
	const float* query,
	const float* keys,
	const float* values,
	size_t seq_len,
	size_t head_dim,
	size_t block_tokens,
	float* output
);

void vspec_cuda_fused_linear_int4_device(
	const float* d_a,
	const unsigned char* d_b_packed,
	const float* d_scales,
	const float* d_zero_points,
	float* d_c,
	size_t m,
	size_t n,
	size_t k,
	size_t n_blocks
);

void vspec_cuda_dequant_int4_to_f32_device(
	const unsigned char* d_b_packed,
	const float* d_scales,
	const float* d_zero_points,
	float* d_w_f32,
	size_t n,
	size_t k,
	size_t n_blocks
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

void vspec_cuda_fused_linear_int2_device(
	const float* d_a,
	const unsigned char* d_b_packed,
	const float* d_scales,
	float* d_c,
	size_t m,
	size_t n,
	size_t k
);

void vspec_cuda_dequant_int2_to_f32_device(
	const unsigned char* d_b_packed,
	const float* d_scales,
	float* d_w_f32,
	size_t n,
	size_t k
);

void vspec_cuda_fused_linear_hybrid_device(
	const float* d_a,
	const unsigned char* d_b_int2_packed,
	const float* d_s_int2,
	const unsigned char* d_b_int4_packed,
	const float* d_s_int4,
	const float* d_zp_int4,
	float* d_c,
	const uint32_t* d_hot_indices,
	size_t hot_count,
	size_t m,
	size_t n,
	size_t k,
	size_t n_blocks_int4
);

void vspec_cuda_refine_hot_int4_device(
	const float* d_a,
	const unsigned char* d_b_int4_packed,
	const float* d_s_int4,
	const float* d_zp_int4,
	float* d_c,
	const uint32_t* d_hot_indices,
	size_t hot_count,
	size_t m,
	size_t n,
	size_t k,
	size_t n_blocks_int4
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
