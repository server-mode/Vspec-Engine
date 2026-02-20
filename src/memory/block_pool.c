#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "vspec/memory/block_pool.h"

int vspec_block_pool_init(VspecBlockPool* pool, void* buffer, size_t block_size, size_t block_count) {
    if (!pool || !buffer || block_size == 0U || block_count == 0U) {
        return 0;
    }

    pool->base = (uint8_t*)buffer;
    pool->block_size = block_size;
    pool->block_count = block_count;

    pool->free_list = (uint8_t*)malloc(block_count);
    if (!pool->free_list) {
        return 0;
    }
    memset(pool->free_list, 1, block_count);
    return 1;
}

void* vspec_block_pool_acquire(VspecBlockPool* pool) {
    if (!pool || !pool->free_list) {
        return NULL;
    }

    for (size_t i = 0; i < pool->block_count; ++i) {
        if (pool->free_list[i]) {
            pool->free_list[i] = 0;
            return pool->base + i * pool->block_size;
        }
    }

    return NULL;
}

void vspec_block_pool_release(VspecBlockPool* pool, void* block) {
    if (!pool || !pool->free_list || !block) {
        return;
    }

    const uintptr_t offset = (uintptr_t)((uint8_t*)block - pool->base);
    const size_t idx = (size_t)(offset / pool->block_size);
    if (idx < pool->block_count) {
        pool->free_list[idx] = 1;
    }
}

size_t vspec_block_pool_free_count(const VspecBlockPool* pool) {
    if (!pool || !pool->free_list) {
        return 0U;
    }

    size_t count = 0;
    for (size_t i = 0; i < pool->block_count; ++i) {
        if (pool->free_list[i]) {
            count += 1U;
        }
    }
    return count;
}
