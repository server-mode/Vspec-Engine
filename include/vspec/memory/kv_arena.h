#ifndef VSPEC_MEMORY_KV_ARENA_H
#define VSPEC_MEMORY_KV_ARENA_H

#include <stddef.h>
#include "vspec/memory/arena_allocator.h"

typedef struct VspecKVArena {
    VspecArenaAllocator arena;
    size_t token_stride_bytes;
} VspecKVArena;

int vspec_kv_arena_init(VspecKVArena* kv, void* buffer, size_t capacity, size_t token_stride_bytes);
void* vspec_kv_arena_reserve(VspecKVArena* kv, size_t tokens);
void vspec_kv_arena_reset(VspecKVArena* kv);

#endif
