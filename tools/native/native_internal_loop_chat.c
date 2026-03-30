#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vspec/compat/safetensors_parser.h"
#include "vspec/runtime/decode_session.h"
#include "vspec/runtime/native_inference.h"
#include "vspec/runtime/native_model_registry.h"
#include "vspec/runtime/output_guard.h"
#include "vspec/runtime/runtime.h"

static void append_token(char* out, size_t cap, const char* token, int with_space) {
    if (!out || cap == 0U || !token) {
        return;
    }
    const size_t used = strlen(out);
    if (used + 1U >= cap) {
        return;
    }
    if (with_space && used > 0U && out[used - 1] != ' ') {
        strncat(out, " ", cap - strlen(out) - 1U);
    }
    strncat(out, token, cap - strlen(out) - 1U);
}

int main(int argc, char** argv) {
    const char* allow_toy = getenv("VSPEC_ALLOW_TOY_NATIVE");
    const int toy_enabled = (allow_toy && (strcmp(allow_toy, "1") == 0 || _stricmp(allow_toy, "true") == 0 || _stricmp(allow_toy, "yes") == 0));
    if (!toy_enabled) {
        fprintf(stderr, "[native-chat] disabled: toy/scaffold generator cannot be used for real model inference.\n");
        fprintf(stderr, "[native-chat] use vspec_native_startup_chat for real runtime path.\n");
        fprintf(stderr, "[native-chat] set VSPEC_ALLOW_TOY_NATIVE=1 only for internal debug.\n");
        return 9;
    }

    if (argc < 3) {
        fprintf(stderr, "usage: native_internal_loop_chat <safetensors-file> <prompt> [max_steps]\n");
        return 2;
    }

    const char* model_file = argv[1];
    const char* prompt = argv[2];
    const size_t max_steps = (argc >= 4) ? (size_t)strtoul(argv[3], NULL, 10) : 16U;

    vspec_runtime_init_default();

    VspecCompatModel model;
    if (!vspec_safetensors_parse_header_file(model_file, &model)) {
        fprintf(stderr, "[native-chat] parse_header_failed file=%s\n", model_file);
        return 1;
    }

    const VspecNativeModelFamily family = vspec_native_model_detect_family(&model);
    printf("[native-chat] model_family=%s supported=%s tensors=%zu\n",
        vspec_native_model_family_name(family),
        vspec_native_model_family_supported(family) ? "yes" : "no",
        model.tensor_count);

    VspecDecodeSession session;
    vspec_decode_session_init(&session, (size_t)8 * 1024U * 1024U * 1024U, 1U, 8U, 1U);
    if (!vspec_decode_session_begin(&session, (size_t)256U * 1024U * 1024U, strlen(prompt) / 4U + 1U, max_steps, 0)) {
        fprintf(stderr, "[native-chat] decode_session_begin_failed\n");
        return 3;
    }

    VspecRuntimeOutputGuardConfig guard_cfg;
    VspecRuntimeOutputGuard guard;
    vspec_output_guard_config_default(&guard_cfg);
    guard_cfg.strictness = 0.72f;
    vspec_output_guard_init(&guard, &guard_cfg);

    static const char* candidates[VSPEC_NATIVE_TOKEN_COUNT] = {
        "The", "answer", "is", "5", "9", "cents", ".",
        "I", "can", "help", "with", "math", "using", "native", "inference",
        "multi", "model", "engine", "today", "models"
    };
    static const int candidate_ids[VSPEC_NATIVE_TOKEN_COUNT] = {
        VSPEC_NATIVE_TOKEN_THE,
        VSPEC_NATIVE_TOKEN_ANSWER,
        VSPEC_NATIVE_TOKEN_IS,
        VSPEC_NATIVE_TOKEN_5,
        VSPEC_NATIVE_TOKEN_9,
        VSPEC_NATIVE_TOKEN_CENTS,
        VSPEC_NATIVE_TOKEN_DOT,
        VSPEC_NATIVE_TOKEN_I,
        VSPEC_NATIVE_TOKEN_CAN,
        VSPEC_NATIVE_TOKEN_HELP,
        VSPEC_NATIVE_TOKEN_WITH,
        VSPEC_NATIVE_TOKEN_MATH,
        VSPEC_NATIVE_TOKEN_USING,
        VSPEC_NATIVE_TOKEN_NATIVE,
        VSPEC_NATIVE_TOKEN_INFERENCE,
        VSPEC_NATIVE_TOKEN_MULTI,
        VSPEC_NATIVE_TOKEN_MODEL,
        VSPEC_NATIVE_TOKEN_ENGINE,
        VSPEC_NATIVE_TOKEN_TODAY,
        VSPEC_NATIVE_TOKEN_MODELS
    };

    char output[1024];
    output[0] = '\0';

    uint64_t rng = (uint64_t)time(NULL);
    size_t produced = 0U;
    int emitted_decode_error = 0;

    VspecNativeForwardContext forward_ctx;
    if (!vspec_native_forward_init(&forward_ctx, &model, model_file, rng)) {
        fprintf(stderr, "[native-chat] native_forward_init_failed insufficient_model_signal\n");
        return 4;
    }

    while (vspec_decode_session_is_active(&session) && produced < max_steps) {
        const size_t quota = vspec_decode_session_next_quota(&session);
        if (quota == 0U) {
            break;
        }
        for (size_t q = 0U; q < quota && produced < max_steps; ++q) {
            float scores[VSPEC_NATIVE_TOKEN_COUNT];
            if (!vspec_native_forward_step(
                    &forward_ctx,
                    prompt,
                    produced,
                    candidate_ids,
                    VSPEC_NATIVE_TOKEN_COUNT,
                    scores)) {
                fprintf(stderr, "[native-chat] native_forward_step_failed\n");
                emitted_decode_error = 1;
                append_token(output, sizeof(output), "[vspec-decode-error]", 1);
                (void)vspec_decode_session_commit(&session, 1U, 1);
                break;
            }

            int picked = candidate_ids[0];
            float best_score = scores[0];
            for (size_t i = 1U; i < VSPEC_NATIVE_TOKEN_COUNT; ++i) {
                if (scores[i] > best_score) {
                    best_score = scores[i];
                    picked = candidate_ids[i];
                }
            }

            const char* token = candidates[picked];
            if (!vspec_output_guard_allow(&guard, token)) {
                token = "[vspec-decode-error]";
                emitted_decode_error = 1;
            }
            if (strcmp(token, "[vspec-decode-error]") == 0) {
                emitted_decode_error = 1;
            }
            const int with_space = (strcmp(token, ".") != 0 && strcmp(token, ",") != 0 && strcmp(token, "!") != 0 && strcmp(token, "?") != 0);
            append_token(output, sizeof(output), token, with_space);
            vspec_output_guard_observe(&guard, token);

            produced += 1U;
            const int reached_eos = (strcmp(token, ".") == 0) || emitted_decode_error;
            (void)vspec_decode_session_commit(&session, 1U, reached_eos);
            if (reached_eos) {
                break;
            }
        }
        if (emitted_decode_error) {
            break;
        }
    }

    if (output[0] == '\0') {
        snprintf(output, sizeof(output), "[vspec-decode-error] Generation failed cleanly.");
    }

    VspecRuntimeOutputGuardReport report;
    vspec_output_guard_report(&guard, &report);

    printf("[native-chat] output: %s\n", output);
    printf("[native-chat] produced_tokens=%zu remaining=%zu guard_integrity=%d\n",
        produced,
        vspec_decode_session_remaining_tokens(&session),
        report.integrity_pass);

    (void)vspec_decode_session_cancel(&session);
    return 0;
}
