#ifndef VSPEC_MEMORY_POOL_H
#define VSPEC_MEMORY_POOL_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecMemoryPool {
    uint8_t* base;
    size_t capacity;
    size_t offset;
    size_t peak_offset;
    size_t alloc_count;
    size_t failed_alloc_count;
} VspecMemoryPool;

int vspec_memory_pool_init(VspecMemoryPool* pool, void* buffer, size_t capacity);
void* vspec_memory_pool_alloc(VspecMemoryPool* pool, size_t size, size_t alignment);
void vspec_memory_pool_reset(VspecMemoryPool* pool);
size_t vspec_memory_pool_used(const VspecMemoryPool* pool);
size_t vspec_memory_pool_remaining(const VspecMemoryPool* pool);
size_t vspec_memory_pool_peak_used(const VspecMemoryPool* pool);
size_t vspec_memory_pool_alloc_count(const VspecMemoryPool* pool);
size_t vspec_memory_pool_failed_alloc_count(const VspecMemoryPool* pool);

#endif
