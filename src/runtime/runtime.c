#include "vspec/kernel/backend.h"
#include "vspec/runtime/runtime.h"

void vspec_runtime_init_default(void) {
    if (vspec_cuda_backend_available()) {
        vspec_set_backend(vspec_make_cuda_backend());
        return;
    }

    if (vspec_rocm_backend_available()) {
        vspec_set_backend(vspec_make_rocm_backend());
        return;
    }

    if (vspec_sycl_backend_available()) {
        vspec_set_backend(vspec_make_sycl_backend());
        return;
    }

    vspec_set_backend(vspec_make_cpu_backend());
}

void vspec_linear_forward(VspecKernelContext* ctx) {
    const VspecBackend* backend = vspec_get_backend();
    if (!backend || !backend->launch_linear) {
        return;
    }
    backend->launch_linear(ctx);
}

void vspec_attention_forward(VspecKernelContext* ctx) {
    const VspecBackend* backend = vspec_get_backend();
    if (!backend || !backend->launch_attention) {
        return;
    }
    backend->launch_attention(ctx);
}
