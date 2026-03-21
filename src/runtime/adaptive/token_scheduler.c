#include "vspec/runtime/adaptive/token_scheduler.h"

#include <ctype.h>
#include <string.h>

static float clampf(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

void vspec_token_scheduler_config_default(VspecTokenSchedulerConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->low_threshold = 0.35f;
    cfg->high_threshold = 0.70f;
    cfg->base_attention_depth = 32U;
    cfg->max_attention_depth = 128U;
}

void vspec_token_scheduler_init(VspecTokenScheduler* sched, const VspecTokenSchedulerConfig* cfg) {
    if (!sched) {
        return;
    }
    if (cfg) {
        sched->cfg = *cfg;
    } else {
        vspec_token_scheduler_config_default(&sched->cfg);
    }
    sched->total_tokens = 0U;
    sched->high_tokens = 0U;
}

float vspec_token_scheduler_score_token(const char* token_text, float entropy_hint) {
    if (!token_text || token_text[0] == '\0') {
        return clampf(0.1f + entropy_hint * 0.05f, 0.0f, 1.0f);
    }

    const size_t len = strlen(token_text);
    size_t digits = 0U;
    size_t punct = 0U;
    size_t letters = 0U;
    size_t upper = 0U;

    for (size_t i = 0U; i < len; ++i) {
        const unsigned char ch = (unsigned char)token_text[i];
        if (isdigit(ch)) {
            digits += 1U;
        } else if (ispunct(ch)) {
            punct += 1U;
        } else if (isalpha(ch)) {
            letters += 1U;
            if (isupper(ch)) {
                upper += 1U;
            }
        }
    }

    float score = 0.15f;
    score += (len >= 8U) ? 0.18f : 0.04f;
    score += (digits > 0U) ? 0.12f : 0.0f;
    score += (upper > 0U) ? 0.10f : 0.0f;
    score -= (punct > letters && punct > 2U) ? 0.10f : 0.0f;
    score += clampf(entropy_hint / 8.0f, 0.0f, 0.20f);

    if (strcmp(token_text, "the") == 0 || strcmp(token_text, "and") == 0 || strcmp(token_text, "to") == 0) {
        score -= 0.12f;
    }

    return clampf(score, 0.0f, 1.0f);
}

VspecTokenScheduleDecision vspec_token_scheduler_schedule_token(
    VspecTokenScheduler* sched,
    const char* token_text,
    float entropy_hint
) {
    VspecTokenScheduleDecision out;
    out.tier = VSPEC_TOKEN_TIER_MEDIUM;
    out.importance = 0.5f;
    out.attention_depth_hint = 64U;
    out.precision_hint_bits = 3U;

    if (!sched) {
        return out;
    }

    out.importance = vspec_token_scheduler_score_token(token_text, entropy_hint);
    if (out.importance < sched->cfg.low_threshold) {
        out.tier = VSPEC_TOKEN_TIER_LOW;
        out.precision_hint_bits = 2U;
        out.attention_depth_hint = sched->cfg.base_attention_depth;
    } else if (out.importance >= sched->cfg.high_threshold) {
        out.tier = VSPEC_TOKEN_TIER_HIGH;
        out.precision_hint_bits = 4U;
        out.attention_depth_hint = sched->cfg.max_attention_depth;
        sched->high_tokens += 1U;
    } else {
        out.tier = VSPEC_TOKEN_TIER_MEDIUM;
        out.precision_hint_bits = 3U;
        out.attention_depth_hint = (sched->cfg.base_attention_depth + sched->cfg.max_attention_depth) / 2U;
    }

    sched->total_tokens += 1U;
    return out;
}
