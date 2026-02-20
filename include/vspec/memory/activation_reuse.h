#ifndef VSPEC_MEMORY_ACTIVATION_REUSE_H
#define VSPEC_MEMORY_ACTIVATION_REUSE_H

#include <stddef.h>
#include "vspec/memory/block_pool.h"

typedef struct VspecActivationReuse {
    VspecBlockPool pool;
} VspecActivationReuse;

int vspec_activation_reuse_init(VspecActivationReuse* ar, void* buffer, size_t block_size, size_t block_count);
void* vspec_activation_acquire(VspecActivationReuse* ar);
void vspec_activation_release(VspecActivationReuse* ar, void* block);

#endif
