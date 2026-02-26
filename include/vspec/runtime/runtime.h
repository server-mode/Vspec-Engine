#ifndef VSPEC_RUNTIME_RUNTIME_H
#define VSPEC_RUNTIME_RUNTIME_H

#include "vspec/kernel/context.h"
#include "vspec/runtime/hw_performance_manager.h"
#include "vspec/runtime/language_structure_guard.h"
#include "vspec/runtime/runtime_behavior_monitor.h"
#include "vspec/runtime/three_bit_runtime_modules.h"
#include "vspec/runtime/torch_compat_module.h"
#include "vspec/runtime/ultimate_optimizer.h"
#include "vspec/runtime/qlora_adapter.h"

void vspec_runtime_init_default(void);
void vspec_runtime_init_with_hw_config(const char* config_path);
const VspecRuntimeHwState* vspec_runtime_get_hw_state(void);
void vspec_linear_forward(VspecKernelContext* ctx);
void vspec_attention_forward(VspecKernelContext* ctx);

void vspec_runtime_language_guard_init(const char* prompt_text, float strictness);
int vspec_runtime_language_guard_allow(const char* token_text);
float vspec_runtime_language_guard_compensate(const char* token_text);
void vspec_runtime_language_guard_observe(const char* token_text);
void vspec_runtime_language_guard_report(VspecLanguageStructureGuardReport* report);

void vspec_runtime_behavior_observe(
	float observed_gpu_utilization,
	float observed_vram_utilization,
	float observed_effective_bits
);
void vspec_runtime_behavior_set_workload_scale(float workload_scale);
void vspec_runtime_behavior_set_integrity_pass(int integrity_pass);
void vspec_runtime_behavior_report(VspecRuntimeBehaviorReport* report);

VspecQuantType vspec_runtime_ultimate_recommend_quant_for_input(
	const float* input,
	size_t count
);
void vspec_runtime_get_ultimate_report(VspecRuntimeUltimateReport* report);

int vspec_runtime_qlora_load_file(const char* path);
int vspec_runtime_qlora_load_manifest_json(const char* manifest_path);
void vspec_runtime_qlora_clear(void);

#endif
