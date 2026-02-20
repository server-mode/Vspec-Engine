#include <stdio.h>
#include <stdint.h>

#include "vspec/memory/pool.h"

int main(void) {
    uint8_t arena[128];
    VspecMemoryPool pool;

    if (!vspec_memory_pool_init(&pool, arena, sizeof(arena))) {
        return 1;
    }

    void* a = vspec_memory_pool_alloc(&pool, 24, 8);
    void* b = vspec_memory_pool_alloc(&pool, 40, 16);

    printf("pool used=%zu remaining=%zu a=%p b=%p\n",
        vspec_memory_pool_used(&pool),
        vspec_memory_pool_remaining(&pool),
        a,
        b);

    vspec_memory_pool_reset(&pool);

    printf("after reset used=%zu remaining=%zu\n",
        vspec_memory_pool_used(&pool),
        vspec_memory_pool_remaining(&pool));

    return 0;
}
