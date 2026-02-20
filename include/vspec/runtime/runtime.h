#ifndef VSPEC_RUNTIME_RUNTIME_H
#define VSPEC_RUNTIME_RUNTIME_H

#include "vspec/kernel/context.h"

void vspec_runtime_init_default(void);
void vspec_linear_forward(VspecKernelContext* ctx);
void vspec_attention_forward(VspecKernelContext* ctx);

#endif
