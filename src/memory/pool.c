#include <stddef.h>
#include <stdint.h>

#include "vspec/memory/pool.h"

static size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0U) {
        return value;
    }
    const size_t mask = alignment - 1U;
    return (value + mask) & ~mask;
}

int vspec_memory_pool_init(VspecMemoryPool* pool, void* buffer, size_t capacity) {
    if (!pool || !buffer || capacity == 0U) {
        return 0;
    }

    pool->base = (uint8_t*)buffer;
    pool->capacity = capacity;
    pool->offset = 0U;
    pool->peak_offset = 0U;
    pool->alloc_count = 0U;
    pool->failed_alloc_count = 0U;
    return 1;
}

void* vspec_memory_pool_alloc(VspecMemoryPool* pool, size_t size, size_t alignment) {
    if (!pool || !pool->base || size == 0U) {
        return NULL;
    }

    if (alignment == 0U) {
        alignment = sizeof(void*);
    }

    const size_t start = align_up(pool->offset, alignment);
    if (start > pool->capacity || size > (pool->capacity - start)) {
        pool->failed_alloc_count += 1U;
        return NULL;
    }

    void* out = pool->base + start;
    pool->offset = start + size;
    pool->alloc_count += 1U;
    if (pool->offset > pool->peak_offset) {
        pool->peak_offset = pool->offset;
    }
    return out;
}

void vspec_memory_pool_reset(VspecMemoryPool* pool) {
    if (!pool) {
        return;
    }
    pool->offset = 0U;
}

size_t vspec_memory_pool_used(const VspecMemoryPool* pool) {
    return pool ? pool->offset : 0U;
}

size_t vspec_memory_pool_remaining(const VspecMemoryPool* pool) {
    if (!pool || pool->offset > pool->capacity) {
        return 0U;
    }
    return pool->capacity - pool->offset;
}

size_t vspec_memory_pool_peak_used(const VspecMemoryPool* pool) {
    return pool ? pool->peak_offset : 0U;
}

size_t vspec_memory_pool_alloc_count(const VspecMemoryPool* pool) {
    return pool ? pool->alloc_count : 0U;
}

size_t vspec_memory_pool_failed_alloc_count(const VspecMemoryPool* pool) {
    return pool ? pool->failed_alloc_count : 0U;
}
