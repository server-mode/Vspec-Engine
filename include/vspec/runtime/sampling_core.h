#ifndef VSPEC_RUNTIME_SAMPLING_CORE_H
#define VSPEC_RUNTIME_SAMPLING_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int vspec_sampling_select_candidate(
    const int* token_ids,
    const float* scores,
    size_t count,
    int greedy,
    uint64_t random_bits,
    int* out_token_id
);

#ifdef __cplusplus
}
#endif

#endif