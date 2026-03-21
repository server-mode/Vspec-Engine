#ifndef VSPEC_RUNTIME_PLUGIN_PLUGIN_API_H
#define VSPEC_RUNTIME_PLUGIN_PLUGIN_API_H

#include <stddef.h>

#include "vspec/runtime/adaptive/runtime_controller.h"
#include "vspec/runtime/adaptive/token_scheduler.h"

typedef struct VspecRuntimePluginHooks {
    void (*on_controller_decision)(const VspecRuntimeAdaptiveTelemetry* telemetry, const VspecRuntimeAdaptiveDecision* decision);
    void (*on_token_scheduled)(const char* token_text, const VspecTokenScheduleDecision* decision);
} VspecRuntimePluginHooks;

int vspec_plugin_register(const char* name, const VspecRuntimePluginHooks* hooks);
int vspec_plugin_unregister(const char* name);
size_t vspec_plugin_count(void);
void vspec_plugin_emit_controller_decision(const VspecRuntimeAdaptiveTelemetry* telemetry, const VspecRuntimeAdaptiveDecision* decision);
void vspec_plugin_emit_token_scheduled(const char* token_text, const VspecTokenScheduleDecision* decision);
int vspec_plugin_load_dynamic(const char* path, const char* symbol_name, char* err_buf, size_t err_buf_len);
int vspec_plugin_unload_dynamic(const char* name, char* err_buf, size_t err_buf_len);

#endif
