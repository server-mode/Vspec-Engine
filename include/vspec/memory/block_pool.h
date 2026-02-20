#ifndef VSPEC_MEMORY_BLOCK_POOL_H
#define VSPEC_MEMORY_BLOCK_POOL_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecBlockPool {
    uint8_t* base;
    size_t block_size;
    size_t block_count;
    uint8_t* free_list;
} VspecBlockPool;

int vspec_block_pool_init(VspecBlockPool* pool, void* buffer, size_t block_size, size_t block_count);
void* vspec_block_pool_acquire(VspecBlockPool* pool);
void vspec_block_pool_release(VspecBlockPool* pool, void* block);
size_t vspec_block_pool_free_count(const VspecBlockPool* pool);

#endif
