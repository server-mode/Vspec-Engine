#include "vspec/runtime/decode_session.h"

#include <string.h>

void vspec_decode_session_init(
    VspecDecodeSession* session,
    size_t total_vram_bytes,
    size_t max_active,
    size_t max_batch_tokens,
    size_t token_quantum
) {
    if (!session) {
        return;
    }

    (void)memset(session, 0, sizeof(*session));
    vspec_request_scheduler_init(
        &session->scheduler,
        total_vram_bytes,
        max_active,
        max_batch_tokens,
        token_quantum
    );
}

int vspec_decode_session_begin(
    VspecDecodeSession* session,
    size_t reserve_bytes,
    size_t prompt_tokens,
    size_t max_new_tokens,
    uint16_t priority
) {
    VspecSchedEnqueueArgs args;

    if (!session || session->active) {
        return 0;
    }

    (void)memset(&args, 0, sizeof(args));
    args.reserve_bytes = (reserve_bytes == 0U) ? 1U : reserve_bytes;
    args.prompt_tokens = prompt_tokens;
    args.max_new_tokens = max_new_tokens;
    args.priority = priority;

    if (!vspec_request_scheduler_enqueue(&session->scheduler, &args, &session->request_id)) {
        return 0;
    }

    session->max_new_tokens = max_new_tokens;
    session->generated_tokens = 0U;
    session->active = 1;
    session->finished = 0;
    return 1;
}

size_t vspec_decode_session_next_quota(VspecDecodeSession* session) {
    VspecSchedBatchItem item;
    size_t count = 0U;

    if (!session || !session->active || session->finished) {
        return 0U;
    }

    (void)memset(&item, 0, sizeof(item));
    count = vspec_request_scheduler_build_batch(&session->scheduler, &item, 1U);
    if (count == 0U || item.request_id != session->request_id) {
        return 0U;
    }
    return item.token_quota;
}

int vspec_decode_session_commit(
    VspecDecodeSession* session,
    size_t generated_tokens,
    int reached_eos
) {
    if (!session || !session->active || session->finished) {
        return 0;
    }

    if (!vspec_request_scheduler_commit(&session->scheduler, session->request_id, generated_tokens, reached_eos)) {
        return 0;
    }

    session->generated_tokens += generated_tokens;
    if (reached_eos ||
        (session->max_new_tokens > 0U && session->generated_tokens >= session->max_new_tokens)) {
        session->active = 0;
        session->finished = 1;
        session->request_id = 0U;
    }
    return 1;
}

int vspec_decode_session_cancel(VspecDecodeSession* session) {
    if (!session || !session->active) {
        return 0;
    }
    if (!vspec_request_scheduler_cancel(&session->scheduler, session->request_id)) {
        return 0;
    }
    session->active = 0;
    session->finished = 1;
    session->request_id = 0U;
    return 1;
}

int vspec_decode_session_is_active(const VspecDecodeSession* session) {
    if (!session) {
        return 0;
    }
    return session->active && !session->finished;
}

size_t vspec_decode_session_remaining_tokens(const VspecDecodeSession* session) {
    if (!session || session->finished || session->max_new_tokens <= session->generated_tokens) {
        return 0U;
    }
    return session->max_new_tokens - session->generated_tokens;
}