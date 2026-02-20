#include "vspec/kernel/backend.h"

static void rocm_launch_linear(VspecKernelContext* ctx) {
    (void)ctx;
}

static void rocm_launch_attention(VspecKernelContext* ctx) {
    (void)ctx;
}

int vspec_rocm_backend_available(void) {
    return 0;
}

VspecBackend vspec_make_rocm_backend(void) {
    VspecBackend backend;
    backend.name = "rocm-stub";
    backend.launch_linear = rocm_launch_linear;
    backend.launch_attention = rocm_launch_attention;
    return backend;
}
