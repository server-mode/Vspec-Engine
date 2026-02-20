#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/quant/int3.h"
#include "vspec/quant/pack.h"
#include "vspec/quant/dynamic_map.h"
#include "vspec/runtime/mixed_bit.h"
#include "vspec/scheduler/dynamic.h"
#include "vspec/attention/flash_block.h"

int main(void) {
    const size_t m = 2, k = 6, n = 2;
    float a[12] = {0.2f, 0.1f, -0.1f, 0.3f, 0.4f, -0.2f, 0.0f, 0.2f, 0.1f, -0.3f, 0.2f, 0.1f};
    int8_t wq[12] = {1, -2, 0, 3, -1, 2, 2, 1, -1, 0, 1, -2};
    float scales[2] = {0.5f, 0.25f};

    const size_t row_bytes = vspec_int3_packed_bytes(k);
    uint8_t* wpacked = (uint8_t*)malloc(n * row_bytes);
    if (!wpacked) return 1;
    for (size_t row = 0; row < n; ++row) {
        vspec_quant_pack_signed(&wq[row * k], k, 3, &wpacked[row * row_bytes]);
    }

    float out[4] = {0};
    vspec_int3_matmul_ref_f32_q3(a, m, k, wpacked, n, scales, out);
    printf("int3 matmul out: %.3f %.3f %.3f %.3f\n", out[0], out[1], out[2], out[3]);

    VspecDynamicQuantConfig cfg;
    vspec_dynamic_quant_default(&cfg);
    VspecDynamicQuantDecision d = vspec_dynamic_quant_decide(a, m * k, &cfg);
    printf("dynamic-quant bits=%u scale=%.4f\n", (unsigned)d.bits, d.scale);

    VspecMixedBitPlan plan;
    vspec_mixed_bit_plan_init(&plan);
    printf("mixed-bit plan bits=%u group=%zu\n", (unsigned)plan.bits, plan.group_size);

    VspecDynamicScheduler sched;
    vspec_dynamic_scheduler_init(&sched, 1024, 2);
    VspecScheduleRequest req = {512, 64};
    printf("schedule req1=%d\n", vspec_dynamic_scheduler_try_schedule(&sched, &req));
    printf("schedule req2=%d\n", vspec_dynamic_scheduler_try_schedule(&sched, &req));
    printf("schedule req3=%d\n", vspec_dynamic_scheduler_try_schedule(&sched, &req));
    vspec_dynamic_scheduler_finish(&sched, &req);

    float q[4] = {0.2f, 0.1f, 0.0f, -0.1f};
    float kblk[8] = {0.1f, 0.2f, 0.0f, 0.1f, 0.3f, -0.1f, 0.2f, 0.0f};
    float vblk[8] = {0.2f, 0.1f, 0.0f, 0.3f, 0.1f, 0.2f, 0.1f, -0.1f};
    float attn_out[4] = {0};
    vspec_flash_attention_block_ref(q, kblk, vblk, 2, 4, 2, attn_out);
    printf("flash-block out: %.3f %.3f %.3f %.3f\n", attn_out[0], attn_out[1], attn_out[2], attn_out[3]);

    free(wpacked);
    return 0;
}
