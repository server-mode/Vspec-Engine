#include "vspec/kernel/backend.h"

static void sycl_launch_linear(VspecKernelContext* ctx) {
    (void)ctx;
}

static void sycl_launch_attention(VspecKernelContext* ctx) {
    (void)ctx;
}

int vspec_sycl_backend_available(void) {
    return 0;
}

VspecBackend vspec_make_sycl_backend(void) {
    VspecBackend backend;
    backend.name = "sycl-stub";
    backend.launch_linear = sycl_launch_linear;
    backend.launch_attention = sycl_launch_attention;
    return backend;
}
