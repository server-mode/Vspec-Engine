#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "vspec/memory/pool.h"

int main(void) {
    uint8_t* arena = (uint8_t*)malloc(1024 * 1024);
    if (!arena) {
        return 1;
    }

    VspecMemoryPool pool;
    if (!vspec_memory_pool_init(&pool, arena, 1024 * 1024)) {
        free(arena);
        return 1;
    }

    const size_t sizes[] = {64, 128, 192, 256, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288};
    const size_t aligns[] = {8, 16, 32, 64};

    size_t success = 0;
    for (size_t round = 0; round < 4; ++round) {
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
            void* p = vspec_memory_pool_alloc(&pool, sizes[i], aligns[i % 4]);
            if (p) {
                success += 1U;
            }
        }
        if (round == 1) {
            vspec_memory_pool_reset(&pool);
        }
    }

    printf("[mem-profiler] capacity=%zu used=%zu remaining=%zu peak=%zu\n",
        pool.capacity,
        vspec_memory_pool_used(&pool),
        vspec_memory_pool_remaining(&pool),
        vspec_memory_pool_peak_used(&pool));

    printf("[mem-profiler] alloc_success=%zu alloc_count=%zu alloc_failed=%zu\n",
        success,
        vspec_memory_pool_alloc_count(&pool),
        vspec_memory_pool_failed_alloc_count(&pool));

    free(arena);
    return 0;
}
