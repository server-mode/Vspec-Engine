#ifndef VSPEC_KERNEL_BACKEND_H
#define VSPEC_KERNEL_BACKEND_H

#include "vspec/kernel/context.h"

typedef struct VspecBackend {
    const char* name;
    void (*launch_linear)(VspecKernelContext* ctx);
    void (*launch_attention)(VspecKernelContext* ctx);
} VspecBackend;

void vspec_set_backend(VspecBackend backend);
const VspecBackend* vspec_get_backend(void);
VspecBackend vspec_make_cpu_backend(void);
int vspec_cuda_backend_available(void);
VspecBackend vspec_make_cuda_backend(void);
int vspec_rocm_backend_available(void);
VspecBackend vspec_make_rocm_backend(void);
int vspec_sycl_backend_available(void);
VspecBackend vspec_make_sycl_backend(void);

#endif
