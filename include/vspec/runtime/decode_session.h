#ifndef VSPEC_RUNTIME_DECODE_SESSION_H
#define VSPEC_RUNTIME_DECODE_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "vspec/scheduler/request_scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VspecDecodeSession {
    VspecRequestScheduler scheduler;
    uint64_t request_id;
    size_t max_new_tokens;
    size_t generated_tokens;
    int active;
    int finished;
} VspecDecodeSession;

void vspec_decode_session_init(
    VspecDecodeSession* session,
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_tokens,
    size_t token_quantum
);

int vspec_decode_session_begin(
    VspecDecodeSession* session,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority
);

size_t vspec_decode_session_next_quota(VspecDecodeSession* session);

int vspec_decode_session_commit(
    VspecDecodeSession* session,
    size_t generated_tokens,
    int reached_eos
);

int vspec_decode_session_cancel(VspecDecodeSession* session);
int vspec_decode_session_is_active(const VspecDecodeSession* session);
size_t vspec_decode_session_remaining_tokens(const VspecDecodeSession* session);

#ifdef __cplusplus
}
#endif

#endif