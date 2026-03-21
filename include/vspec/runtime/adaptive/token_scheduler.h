#ifndef VSPEC_RUNTIME_ADAPTIVE_TOKEN_SCHEDULER_H
#define VSPEC_RUNTIME_ADAPTIVE_TOKEN_SCHEDULER_H

#include <stddef.h>

typedef enum VspecTokenTier {
    VSPEC_TOKEN_TIER_LOW = 0,
    VSPEC_TOKEN_TIER_MEDIUM = 1,
    VSPEC_TOKEN_TIER_HIGH = 2
} VspecTokenTier;

typedef struct VspecTokenScheduleDecision {
    VspecTokenTier tier;
    float importance;
    unsigned int attention_depth_hint;
    unsigned int precision_hint_bits;
} VspecTokenScheduleDecision;

typedef struct VspecTokenSchedulerConfig {
    float low_threshold;
    float high_threshold;
    unsigned int base_attention_depth;
    unsigned int max_attention_depth;
} VspecTokenSchedulerConfig;

typedef struct VspecTokenScheduler {
    VspecTokenSchedulerConfig cfg;
    size_t total_tokens;
    size_t high_tokens;
} VspecTokenScheduler;

void vspec_token_scheduler_config_default(VspecTokenSchedulerConfig* cfg);
void vspec_token_scheduler_init(VspecTokenScheduler* sched, const VspecTokenSchedulerConfig* cfg);
float vspec_token_scheduler_score_token(const char* token_text, float entropy_hint);
VspecTokenScheduleDecision vspec_token_scheduler_schedule_token(
    VspecTokenScheduler* sched,
    const char* token_text,
    float entropy_hint
);

#endif
