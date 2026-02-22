#ifndef VSPEC_RUNTIME_RUNTIME_H
#define VSPEC_RUNTIME_RUNTIME_H

#include "vspec/kernel/context.h"
#include "vspec/runtime/hw_performance_manager.h"

void vspec_runtime_init_default(void);
void vspec_runtime_init_with_hw_config(const char* config_path);
const VspecRuntimeHwState* vspec_runtime_get_hw_state(void);
void vspec_linear_forward(VspecKernelContext* ctx);
void vspec_attention_forward(VspecKernelContext* ctx);

#endif
