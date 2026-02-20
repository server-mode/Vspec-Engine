#include "vspec/kernel/backend.h"

#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
void vspec_cuda_launch_linear_impl(VspecKernelContext* ctx);
void vspec_cuda_launch_attention_impl(VspecKernelContext* ctx);
int vspec_cuda_runtime_available(void);
#endif

static void cuda_launch_linear(VspecKernelContext* ctx) {
#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
    vspec_cuda_launch_linear_impl(ctx);
#else
    (void)ctx;
#endif
}

static void cuda_launch_attention(VspecKernelContext* ctx) {
#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
    vspec_cuda_launch_attention_impl(ctx);
#else
    (void)ctx;
#endif
}

int vspec_cuda_backend_available(void) {
#if defined(VSPEC_HAS_CUDA) && VSPEC_HAS_CUDA
    return vspec_cuda_runtime_available();
#else
    return 0;
#endif
}

VspecBackend vspec_make_cuda_backend(void) {
    VspecBackend backend;
    backend.name = "cuda-stub";
    backend.launch_linear = cuda_launch_linear;
    backend.launch_attention = cuda_launch_attention;
    return backend;
}
