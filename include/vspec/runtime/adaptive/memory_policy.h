#ifndef VSPEC_RUNTIME_ADAPTIVE_MEMORY_POLICY_H
#define VSPEC_RUNTIME_ADAPTIVE_MEMORY_POLICY_H

#include <stddef.h>

typedef enum VspecKvPolicyAction {
    VSPEC_KV_POLICY_STORE = 0,
    VSPEC_KV_POLICY_COMPRESS = 1,
    VSPEC_KV_POLICY_RECOMPUTE = 2
} VspecKvPolicyAction;

typedef struct VspecMemoryPolicyConfig {
    float pressure_compress;
    float pressure_recompute;
    float importance_floor;
} VspecMemoryPolicyConfig;

typedef struct VspecMemoryPolicyInput {
    float vram_pressure;
    float token_importance;
    size_t active_tokens;
    size_t kv_bytes;
} VspecMemoryPolicyInput;

typedef struct VspecMemoryPolicy {
    VspecMemoryPolicyConfig cfg;
} VspecMemoryPolicy;

void vspec_memory_policy_config_default(VspecMemoryPolicyConfig* cfg);
void vspec_memory_policy_init(VspecMemoryPolicy* policy, const VspecMemoryPolicyConfig* cfg);
VspecKvPolicyAction vspec_memory_policy_decide(const VspecMemoryPolicy* policy, const VspecMemoryPolicyInput* input);

#endif
