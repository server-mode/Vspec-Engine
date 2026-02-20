#ifndef VSPEC_KERNEL_CONTEXT_H
#define VSPEC_KERNEL_CONTEXT_H

#include <stddef.h>

#include "vspec/quant/quant.h"

typedef struct VspecExecConfig {
    size_t m;
    size_t n;
    size_t k;
    size_t flags;
} VspecExecConfig;

typedef struct VspecKernelContext {
    void* input;
    void* weight;
    void* output;
    VspecQuantMeta qmeta;
    VspecExecConfig config;
} VspecKernelContext;

#endif
