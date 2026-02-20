#include <stddef.h>
#include <stdint.h>

#include "vspec/memory/arena_allocator.h"

static size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0U) {
        return value;
    }
    const size_t mask = alignment - 1U;
    return (value + mask) & ~mask;
}

int vspec_arena_init(VspecArenaAllocator* arena, void* buffer, size_t capacity) {
    if (!arena || !buffer || capacity == 0U) {
        return 0;
    }
    arena->base = (uint8_t*)buffer;
    arena->capacity = capacity;
    arena->offset = 0U;
    return 1;
}

void* vspec_arena_alloc(VspecArenaAllocator* arena, size_t size, size_t alignment) {
    if (!arena || !arena->base || size == 0U) {
        return NULL;
    }

    if (alignment == 0U) {
        alignment = sizeof(void*);
    }

    const size_t start = align_up(arena->offset, alignment);
    if (start > arena->capacity || size > (arena->capacity - start)) {
        return NULL;
    }

    void* out = arena->base + start;
    arena->offset = start + size;
    return out;
}

void vspec_arena_reset(VspecArenaAllocator* arena) {
    if (!arena) {
        return;
    }
    arena->offset = 0U;
}
