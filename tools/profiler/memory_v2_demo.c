#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "vspec/memory/arena_allocator.h"
#include "vspec/memory/block_pool.h"
#include "vspec/memory/kv_arena.h"
#include "vspec/memory/activation_reuse.h"
#include "vspec/memory/memory_metrics.h"

int main(void) {
    uint8_t* weight_buf = (uint8_t*)malloc(1024);
    uint8_t* act_buf = (uint8_t*)malloc(2048);
    uint8_t* kv_buf = (uint8_t*)malloc(4096);

    VspecArenaAllocator weight_arena;
    vspec_arena_init(&weight_arena, weight_buf, 1024);

    VspecActivationReuse act_reuse;
    vspec_activation_reuse_init(&act_reuse, act_buf, 256, 8);

    VspecKVArena kv_arena;
    vspec_kv_arena_init(&kv_arena, kv_buf, 4096, 128);

    void* w0 = vspec_arena_alloc(&weight_arena, 128, 16);
    void* w1 = vspec_arena_alloc(&weight_arena, 256, 16);
    void* a0 = vspec_activation_acquire(&act_reuse);
    void* a1 = vspec_activation_acquire(&act_reuse);
    void* k0 = vspec_kv_arena_reserve(&kv_arena, 8);

    VspecMemoryMetrics metrics;
    vspec_memory_metrics_reset(&metrics);
    vspec_memory_metrics_add(&metrics, 384, 512, 1024, 0);

    printf("weight arena used=%zu\n", weight_arena.offset);
    printf("activation free=%zu\n", vspec_block_pool_free_count(&act_reuse.pool));
    printf("kv arena used=%zu\n", kv_arena.arena.offset);
    printf("metrics weight=%zu act=%zu kv=%zu scratch=%zu\n",
        metrics.weight_bytes, metrics.activation_bytes, metrics.kv_bytes, metrics.scratch_bytes);

    (void)w0; (void)w1; (void)a0; (void)a1; (void)k0;

    free(weight_buf);
    free(act_buf);
    free(kv_buf);
    return 0;
}
