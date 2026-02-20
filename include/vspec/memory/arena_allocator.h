#ifndef VSPEC_MEMORY_ARENA_ALLOCATOR_H
#define VSPEC_MEMORY_ARENA_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecArenaAllocator {
    uint8_t* base;
    size_t capacity;
    size_t offset;
} VspecArenaAllocator;

int vspec_arena_init(VspecArenaAllocator* arena, void* buffer, size_t capacity);
void* vspec_arena_alloc(VspecArenaAllocator* arena, size_t size, size_t alignment);
void vspec_arena_reset(VspecArenaAllocator* arena);

#endif
