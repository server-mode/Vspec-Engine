#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vspec/graph/ir.h"
#include "vspec/graph/optimizer.h"
#include "vspec/memory/memory_metrics.h"
#include "vspec/memory/vram_scheduler.h"
#include "vspec/parallel/multi_gpu.h"
#include "vspec/quant/int3.h"
#include "vspec/quant/int4.h"
#include "vspec/quant/pack.h"

static int test_graph_liveness_backward(void) {
    VspecGraph graph;
    vspec_graph_init(&graph);
    if (!vspec_graph_add_node(&graph, VSPEC_OP_LINEAR, 0, 1, 10)) return 0;
    if (!vspec_graph_add_node(&graph, VSPEC_OP_ACT_RELU, 10, 0, 11)) return 0;
    if (!vspec_graph_add_node(&graph, VSPEC_OP_ATTENTION, 11, 12, 13)) return 0;

    const uint32_t live[] = {13};
    VspecGraphOptStats stats = {0};
    vspec_graph_optimize_dead_nodes(&graph, live, 1U, &stats);

    if (graph.node_count != 3U) {
        return 0;
    }
    if (stats.removed_dead_nodes != 0U) {
        return 0;
    }
    return 1;
}

static int test_multi_gpu_plan(void) {
    VspecMultiGpuPlan plan;
    if (!vspec_multi_gpu_plan_build(&plan, 8U, 40U, 4U)) {
        return 0;
    }

    if (plan.device_count != 8U) return 0;
    if (plan.tensor_parallel != 4U) return 0;
    if (plan.pipeline_parallel != 2U) return 0;

    if (vspec_multi_gpu_plan_stage_for_layer(&plan, 0U) != 0U) return 0;
    if (vspec_multi_gpu_plan_stage_for_layer(&plan, 39U) != 1U) return 0;
    if (vspec_multi_gpu_plan_device_for_layer(&plan, 25U) != 4U) return 0;
    if (vspec_multi_gpu_plan_shard_width(&plan, 4096U) != 1024U) return 0;

    return 1;
}

static int test_memory_metrics_and_vram(void) {
    VspecMemoryMetrics metrics;
    vspec_memory_metrics_reset(&metrics);
    vspec_memory_metrics_add(&metrics, 1024U, 512U, 256U, 128U);

    const size_t total = vspec_memory_metrics_total(&metrics);
    if (total != 1920U) {
        return 0;
    }

    const float pressure = vspec_memory_metrics_pressure(&metrics, 4096U);
    if (!(pressure > 0.46f && pressure < 0.48f)) {
        return 0;
    }

    VspecVramBudget budget;
    vspec_vram_budget_init(&budget, 2048U);
    if (!vspec_vram_try_reserve(&budget, 1024U)) {
        return 0;
    }
    if (vspec_vram_try_reserve(&budget, 2048U)) {
        return 0;
    }
    if (budget.reserve_fail_count == 0U) {
        return 0;
    }
    if (vspec_vram_available(&budget) != 1024U) {
        return 0;
    }
    if (!(vspec_vram_utilization(&budget) > 0.49f && vspec_vram_utilization(&budget) < 0.51f)) {
        return 0;
    }
    return 1;
}

static void pack_int4_row(const int8_t* src, size_t k, uint8_t* dst) {
    vspec_int4_pack(src, k, dst);
}

static void pack_int3_row(const int8_t* src, size_t k, uint8_t* dst) {
    vspec_quant_pack_signed(src, k, 3U, dst);
}

static int nearly_equal(float a, float b) {
    const float d = fabsf(a - b);
    return d <= 1e-4f;
}

static int test_quant_zero_point_paths(void) {
    const size_t m = 2U;
    const size_t k = 8U;
    const size_t n = 2U;

    const float a[16] = {
        0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
        -0.5f, 0.4f, -0.3f, 0.2f, -0.1f, 0.9f, -0.7f, 0.6f
    };

    const int8_t w0[8] = {3, -2, 1, -1, 2, -3, 0, 1};
    const int8_t w1[8] = {-1, 2, -2, 3, -3, 1, 0, -1};

    const float scales[2] = {0.1f, 0.2f};

    uint8_t b4[2 * 4] = {0};
    pack_int4_row(w0, k, b4 + 0U * vspec_int4_packed_bytes(k));
    pack_int4_row(w1, k, b4 + 1U * vspec_int4_packed_bytes(k));

    float out4_a[4] = {0};
    float out4_b[4] = {0};
    float zp4[2] = {0};

    vspec_int4_matmul_ref_f32_q4(a, m, k, b4, n, scales, out4_a);
    vspec_int4_compute_zero_points(b4, k, n, zp4);
    vspec_int4_matmul_ref_f32_q4_with_zero_points(a, m, k, b4, n, scales, zp4, out4_b);

    for (size_t i = 0U; i < 4U; ++i) {
        if (!nearly_equal(out4_a[i], out4_b[i])) {
            return 0;
        }
    }

    uint8_t b3[2 * 3] = {0};
    pack_int3_row(w0, k, b3 + 0U * vspec_int3_packed_bytes(k));
    pack_int3_row(w1, k, b3 + 1U * vspec_int3_packed_bytes(k));

    float out3_a[4] = {0};
    float out3_b[4] = {0};
    float zp3[2] = {0};

    vspec_int3_matmul_ref_f32_q3(a, m, k, b3, n, scales, out3_a);
    vspec_int3_compute_zero_points(b3, k, n, zp3);
    vspec_int3_matmul_ref_f32_q3_with_zero_points(a, m, k, b3, n, scales, zp3, out3_b);

    for (size_t i = 0U; i < 4U; ++i) {
        if (!nearly_equal(out3_a[i], out3_b[i])) {
            return 0;
        }
    }

    return 1;
}

int main(void) {
    int ok = 1;

    ok = ok && test_graph_liveness_backward();
    ok = ok && test_multi_gpu_plan();
    ok = ok && test_memory_metrics_and_vram();
    ok = ok && test_quant_zero_point_paths();

    if (!ok) {
        fprintf(stderr, "[job29-now-validation] FAILED\n");
        return 1;
    }

    printf("[job29-now-validation] OK\n");
    return 0;
}
