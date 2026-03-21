#include "vspec/runtime/output_guard.h"

#include <ctype.h>
#include <string.h>

static float vspec_clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static int contains_replacement_char(const unsigned char* text) {
    if (!text) {
        return 0;
    }
    while (*text) {
        if (text[0] == 0xEF && text[1] != '\0' && text[2] != '\0' && text[1] == 0xBF && text[2] == 0xBD) {
            return 1;
        }
        text += 1;
    }
    return 0;
}

static int has_control_char(const unsigned char* text) {
    if (!text) {
        return 0;
    }
    while (*text) {
        unsigned char c = *text;
        if ((c < 32u && c != '\n' && c != '\r' && c != '\t') || c == 127u) {
            return 1;
        }
        text += 1;
    }
    return 0;
}

static int has_repeat_run(const unsigned char* text, uint32_t max_repeat_run) {
    if (!text || max_repeat_run < 2u) {
        return 0;
    }

    unsigned char last = 0u;
    uint32_t run = 0u;
    while (*text) {
        unsigned char c = *text;
        if (c == last) {
            run += 1u;
        } else {
            last = c;
            run = 1u;
        }
        if (run > max_repeat_run) {
            return 1;
        }
        text += 1;
    }
    return 0;
}

static int text_is_in_recent_tail(const VspecRuntimeOutputGuard* guard, const char* text) {
    if (!guard || !text) {
        return 0;
    }
    if (text[0] == '\0') {
        return 0;
    }
    return strstr(guard->recent_tail, text) != NULL;
}

void vspec_output_guard_config_default(VspecRuntimeOutputGuardConfig* config) {
    if (!config) {
        return;
    }
    config->strictness = 0.55f;
    config->reject_control_chars = 1;
    config->reject_replacement_char = 1;
    config->max_repeat_run = 4u;
}

void vspec_output_guard_init(VspecRuntimeOutputGuard* guard, const VspecRuntimeOutputGuardConfig* config) {
    if (!guard) {
        return;
    }
    (void)memset(guard, 0, sizeof(*guard));
    if (config) {
        guard->config = *config;
    } else {
        vspec_output_guard_config_default(&guard->config);
    }
}

void vspec_output_guard_observe(VspecRuntimeOutputGuard* guard, const char* text_fragment) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return;
    }

    size_t frag_len = strlen(text_fragment);
    size_t capped_len = frag_len;
    if (capped_len > sizeof(guard->recent_tail) - 1u) {
        capped_len = sizeof(guard->recent_tail) - 1u;
    }

    if (guard->recent_len + (uint32_t)capped_len >= (uint32_t)sizeof(guard->recent_tail)) {
        uint32_t overflow = (guard->recent_len + (uint32_t)capped_len) - (uint32_t)sizeof(guard->recent_tail) + 1u;
        if (overflow >= guard->recent_len) {
            guard->recent_len = 0u;
            guard->recent_tail[0] = '\0';
        } else {
            (void)memmove(guard->recent_tail, guard->recent_tail + overflow, (size_t)(guard->recent_len - overflow));
            guard->recent_len -= overflow;
            guard->recent_tail[guard->recent_len] = '\0';
        }
    }

    (void)memcpy(guard->recent_tail + guard->recent_len, text_fragment, capped_len);
    guard->recent_len += (uint32_t)capped_len;
    guard->recent_tail[guard->recent_len] = '\0';

    guard->total_observed_chars += (uint64_t)frag_len;
}

int vspec_output_guard_allow(VspecRuntimeOutputGuard* guard, const char* text_fragment) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return 1;
    }

    const unsigned char* bytes = (const unsigned char*)text_fragment;
    if (guard->config.reject_replacement_char && contains_replacement_char(bytes)) {
        guard->replacement_char_count += 1u;
        return 0;
    }

    if (guard->config.reject_control_chars && has_control_char(bytes)) {
        guard->control_char_count += 1u;
        return 0;
    }

    if (has_repeat_run(bytes, guard->config.max_repeat_run)) {
        guard->repeat_run_reject_count += 1u;
        return 0;
    }

    if (strlen(text_fragment) >= 4u && text_is_in_recent_tail(guard, text_fragment)) {
        guard->recent_repeat_reject_count += 1u;
        return 0;
    }

    return 1;
}

float vspec_output_guard_score_adjustment(const VspecRuntimeOutputGuard* guard, const char* text_fragment) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return 0.0f;
    }

    float score = 0.0f;
    const unsigned char* bytes = (const unsigned char*)text_fragment;
    if (contains_replacement_char(bytes)) {
        score -= 0.90f;
    }
    if (has_control_char(bytes)) {
        score -= 1.10f;
    }
    if (has_repeat_run(bytes, guard->config.max_repeat_run)) {
        score -= 0.75f;
    }
    if (strlen(text_fragment) >= 4u && text_is_in_recent_tail(guard, text_fragment)) {
        score -= 0.60f;
    }

    {
        size_t alpha_count = 0u;
        size_t vowel_count = 0u;
        const char* cursor = text_fragment;
        while (*cursor) {
            unsigned char c = (unsigned char)*cursor;
            if (isalpha(c)) {
                alpha_count += 1u;
                switch ((char)tolower((unsigned char)c)) {
                    case 'a':
                    case 'e':
                    case 'i':
                    case 'o':
                    case 'u':
                    case 'y':
                        vowel_count += 1u;
                        break;
                    default:
                        break;
                }
            }
            cursor += 1;
        }
        if (alpha_count >= 6u) {
            float ratio = (float)vowel_count / (float)alpha_count;
            if (ratio < 0.12f) {
                score -= 0.45f;
            }
        }
    }

    return vspec_clampf(score * (0.5f + guard->config.strictness), -2.5f, 0.5f);
}

void vspec_output_guard_report(const VspecRuntimeOutputGuard* guard, VspecRuntimeOutputGuardReport* report) {
    if (!report) {
        return;
    }
    (void)memset(report, 0, sizeof(*report));
    if (!guard) {
        report->integrity_pass = 1;
        return;
    }

    report->total_observed_chars = guard->total_observed_chars;
    report->replacement_char_count = guard->replacement_char_count;
    report->control_char_count = guard->control_char_count;
    report->repeat_run_reject_count = guard->repeat_run_reject_count;
    report->recent_repeat_reject_count = guard->recent_repeat_reject_count;

    report->integrity_pass = (guard->replacement_char_count == 0u && guard->control_char_count == 0u) ? 1 : 0;
}
