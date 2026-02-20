#ifndef VSPEC_ATTENTION_SPARSE_H
#define VSPEC_ATTENTION_SPARSE_H

#include <stddef.h>

size_t vspec_sparse_mask_allow(const size_t* allowed_indices, size_t allowed_count, size_t index);

#endif
