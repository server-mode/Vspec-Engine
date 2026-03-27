#include <ctype.h>
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

static void trim_line(char* s) {
    if (!s) {
        return;
    }
    size_t n = strlen(s);
    while (n > 0U) {
        char c = s[n - 1U];
        if (c == '\n' || c == '\r' || isspace((unsigned char)c)) {
            s[n - 1U] = '\0';
            n -= 1U;
        } else {
            break;
        }
    }
}

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

static int generate_turn(
    VspecNativeForwardContext* forward_ctx,
    const char* prompt,
    size_t max_steps,
    char* out,
    size_t out_cap,
    size_t* out_produced,
    int* out_integrity
) {
    if (!forward_ctx || !prompt || !out || out_cap == 0U) {
        return 0;
    }

    out[0] = '\0';

    VspecDecodeSession session;
    vspec_decode_session_init(&session, (size_t)8 * 1024U * 1024U * 1024U, 1U, 8U, 1U);
    if (!vspec_decode_session_begin(&session, (size_t)256U * 1024U * 1024U, strlen(prompt) / 4U + 1U, max_steps, 0)) {
        return 0;
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

    size_t produced = 0U;
    int emitted_decode_error = 0;

    while (vspec_decode_session_is_active(&session) && produced < max_steps) {
        const size_t quota = vspec_decode_session_next_quota(&session);
        if (quota == 0U) {
            break;
        }

        for (size_t q = 0U; q < quota && produced < max_steps; ++q) {
            float scores[VSPEC_NATIVE_TOKEN_COUNT];
            if (!vspec_native_forward_step(
                    forward_ctx,
                    prompt,
                    produced,
                    candidate_ids,
                    VSPEC_NATIVE_TOKEN_COUNT,
                    scores)) {
                emitted_decode_error = 1;
                append_token(out, out_cap, "[vspec-decode-error]", 1);
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
            append_token(out, out_cap, token, with_space);
            vspec_output_guard_observe(&guard, token);

            produced += 1U;
            {
                const int reached_eos = (strcmp(token, ".") == 0) || emitted_decode_error;
                (void)vspec_decode_session_commit(&session, 1U, reached_eos);
                if (reached_eos) {
                    break;
                }
            }
        }
        if (emitted_decode_error) {
            break;
        }
    }

    if (out[0] == '\0') {
        snprintf(out, out_cap, "[vspec-decode-error] Generation failed cleanly.");
    }

    {
        VspecRuntimeOutputGuardReport report;
        vspec_output_guard_report(&guard, &report);
        if (out_integrity) {
            *out_integrity = report.integrity_pass;
        }
    }
    if (out_produced) {
        *out_produced = produced;
    }

    (void)vspec_decode_session_cancel(&session);
    return 1;
}

int main(int argc, char** argv) {
    const char* allow_toy = getenv("VSPEC_ALLOW_TOY_NATIVE");
    const int toy_enabled = (allow_toy && (strcmp(allow_toy, "1") == 0 || _stricmp(allow_toy, "true") == 0 || _stricmp(allow_toy, "yes") == 0));
    if (!toy_enabled) {
        fprintf(stderr, "[native-session] disabled: this binary is toy/scaffold and may produce non-model outputs.\n");
        fprintf(stderr, "[native-session] use vspec_native_startup_chat (real model runtime) instead.\n");
        fprintf(stderr, "[native-session] set VSPEC_ALLOW_TOY_NATIVE=1 only for internal debug.\n");
        return 9;
    }

    if (argc < 2) {
        fprintf(stderr, "usage: native_session_chat <safetensors-file> [max_steps]\n");
        return 2;
    }

    const char* model_file = argv[1];
    const size_t max_steps = (argc >= 3) ? (size_t)strtoul(argv[2], NULL, 10) : 256U;

    VspecCompatModel model;
    if (!vspec_safetensors_parse_header_file(model_file, &model)) {
        fprintf(stderr, "[native-session] parse_header_failed file=%s\n", model_file);
        return 1;
    }

    {
        const VspecNativeModelFamily family = vspec_native_model_detect_family(&model);
        printf("[native-session] model_family=%s supported=%s tensors=%zu\n",
            vspec_native_model_family_name(family),
            vspec_native_model_family_supported(family) ? "yes" : "no",
            model.tensor_count);
    }

    printf("[native-session][warning] This executable is a demo/native-scaffold.\n");
    printf("[native-session][warning] It does NOT run full transformer inference or a real tokenizer/vocab.\n");
    printf("[native-session][warning] Output is selected from a tiny fixed candidate list for testing.\n");

    VspecNativeForwardContext forward_ctx;
    if (!vspec_native_forward_init(&forward_ctx, &model, model_file, (uint64_t)time(NULL))) {
        fprintf(stderr, "[native-session] native_forward_init_failed\n");
        return 3;
    }

    printf("[native-session] type /exit to quit\n");

    char line[4096];
    while (1) {
        printf("you> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n[native-session] bye\n");
            break;
        }
        trim_line(line);
        if (line[0] == '\0') {
            continue;
        }
        if (strcmp(line, "/exit") == 0 || strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) {
            printf("[native-session] bye\n");
            break;
        }

        char output[2048];
        size_t produced = 0U;
        int integrity = 1;
        if (!generate_turn(&forward_ctx, line, max_steps, output, sizeof(output), &produced, &integrity)) {
            printf("assistant> [vspec-decode-error] Generation failed on native session turn.\n");
            continue;
        }
        printf("[native-session] produced_tokens=%zu guard_integrity=%d\n", produced, integrity);
        printf("assistant> %s\n", output);
    }

    return 0;
}
