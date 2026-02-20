#include <stdio.h>
#include <string.h>

#include "vspec/quant/dynamic_map.h"
#include "vspec/runtime/mixed_bit.h"
#include "vspec/compat/weight_mapper.h"
#include "vspec/attention/rotary.h"
#include "vspec/attention/kv_ring.h"
#include "vspec/memory/vram_scheduler.h"

int main(void) {
    VspecDynamicQuantConfig cfg;
    vspec_dynamic_quant_default(&cfg);

    float data[8] = {0.1f, -0.2f, 0.05f, 0.7f, -0.3f, 0.02f, 0.9f, -0.4f};
    VspecDynamicQuantDecision d = vspec_dynamic_quant_decide(data, 8, &cfg);
    printf("dynamic-quant bits=%u scale=%.4f\n", (unsigned)d.bits, d.scale);

    VspecMixedBitPlan plan;
    vspec_mixed_bit_plan_init(&plan);
    printf("mixed-bit plan bits=%u group=%zu\n", (unsigned)plan.bits, plan.group_size);

    VspecCompatModel in_model;
    VspecCompatModel out_model;
    vspec_compat_model_init(&in_model);
    in_model.tensor_count = 1;
    snprintf(in_model.tensors[0].name, VSPEC_COMPAT_NAME_MAX, "w0");
    snprintf(in_model.tensors[0].dtype, sizeof(in_model.tensors[0].dtype), "F16");
    in_model.tensors[0].ndim = 2;
    in_model.tensors[0].shape[0] = 4;
    in_model.tensors[0].shape[1] = 4;

    if (vspec_weight_map_identity(&in_model, &out_model)) {
        printf("weight-map tensors=%zu name=%s\n", out_model.tensor_count, out_model.tensors[0].name);
    }

    float q[4] = {1.0f, 0.0f, 0.5f, -0.5f};
    float k[4] = {0.2f, 0.1f, -0.2f, 0.3f};
    vspec_rotary_apply(q, k, 4, 1);
    printf("rotary q=[%.3f %.3f %.3f %.3f]\n", q[0], q[1], q[2], q[3]);

    float key_buf[8] = {0};
    float val_buf[8] = {0};
    VspecKVCacheRing ring;
    vspec_kv_ring_init(&ring, key_buf, val_buf, 1, 1, 4);
    vspec_kv_ring_push(&ring, q, k);
    const float* k0 = vspec_kv_ring_key_at(&ring, 0, 0);
    printf("kv-ring key0=[%.3f %.3f %.3f %.3f]\n", k0[0], k0[1], k0[2], k0[3]);

    VspecVramBudget budget;
    vspec_vram_budget_init(&budget, 1024);
    printf("vram reserve 512=%d\n", vspec_vram_try_reserve(&budget, 512));
    printf("vram reserve 768=%d\n", vspec_vram_try_reserve(&budget, 768));

    return 0;
}
