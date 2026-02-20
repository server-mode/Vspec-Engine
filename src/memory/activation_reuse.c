#include "vspec/memory/activation_reuse.h"

int vspec_activation_reuse_init(VspecActivationReuse* ar, void* buffer, size_t block_size, size_t block_count) {
    if (!ar) {
        return 0;
    }
    return vspec_block_pool_init(&ar->pool, buffer, block_size, block_count);
}

void* vspec_activation_acquire(VspecActivationReuse* ar) {
    if (!ar) {
        return NULL;
    }
    return vspec_block_pool_acquire(&ar->pool);
}

void vspec_activation_release(VspecActivationReuse* ar, void* block) {
    if (!ar) {
        return;
    }
    vspec_block_pool_release(&ar->pool, block);
}
