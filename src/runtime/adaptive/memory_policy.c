#include "vspec/runtime/adaptive/memory_policy.h"

void vspec_memory_policy_config_default(VspecMemoryPolicyConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->pressure_compress = 0.80f;
    cfg->pressure_recompute = 0.93f;
    cfg->importance_floor = 0.30f;
}

void vspec_memory_policy_init(VspecMemoryPolicy* policy, const VspecMemoryPolicyConfig* cfg) {
    if (!policy) {
        return;
    }
    if (cfg) {
        policy->cfg = *cfg;
    } else {
        vspec_memory_policy_config_default(&policy->cfg);
    }
}

VspecKvPolicyAction vspec_memory_policy_decide(const VspecMemoryPolicy* policy, const VspecMemoryPolicyInput* input) {
    if (!policy || !input) {
        return VSPEC_KV_POLICY_STORE;
    }

    if (input->vram_pressure >= policy->cfg.pressure_recompute && input->token_importance <= policy->cfg.importance_floor) {
        return VSPEC_KV_POLICY_RECOMPUTE;
    }

    if (input->vram_pressure >= policy->cfg.pressure_compress) {
        return VSPEC_KV_POLICY_COMPRESS;
    }

    return VSPEC_KV_POLICY_STORE;
}
