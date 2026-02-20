#include "vspec/attention/sparse.h"

size_t vspec_sparse_mask_allow(const size_t* allowed_indices, size_t allowed_count, size_t index) {
    if (!allowed_indices || allowed_count == 0) {
        return 0U;
    }
    for (size_t i = 0; i < allowed_count; ++i) {
        if (allowed_indices[i] == index) {
            return 1U;
        }
    }
    return 0U;
}
