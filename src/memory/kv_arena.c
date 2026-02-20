#include "vspec/memory/kv_arena.h"

int vspec_kv_arena_init(VspecKVArena* kv, void* buffer, size_t capacity, size_t token_stride_bytes) {
    if (!kv || !buffer || capacity == 0U || token_stride_bytes == 0U) {
        return 0;
    }

    if (!vspec_arena_init(&kv->arena, buffer, capacity)) {
        return 0;
    }

    kv->token_stride_bytes = token_stride_bytes;
    return 1;
}

void* vspec_kv_arena_reserve(VspecKVArena* kv, size_t tokens) {
    if (!kv || tokens == 0U) {
        return NULL;
    }
    const size_t bytes = tokens * kv->token_stride_bytes;
    return vspec_arena_alloc(&kv->arena, bytes, 16);
}

void vspec_kv_arena_reset(VspecKVArena* kv) {
    if (!kv) {
        return;
    }
    vspec_arena_reset(&kv->arena);
}
