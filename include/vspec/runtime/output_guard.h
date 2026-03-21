#ifndef VSPEC_RUNTIME_OUTPUT_GUARD_H
#define VSPEC_RUNTIME_OUTPUT_GUARD_H

#include <stddef.h>
#include <stdint.h>

typedef struct VspecRuntimeOutputGuardConfig {
    float strictness;
    int reject_control_chars;
    int reject_replacement_char;
    uint32_t max_repeat_run;
} VspecRuntimeOutputGuardConfig;

typedef struct VspecRuntimeOutputGuard {
    VspecRuntimeOutputGuardConfig config;
    char recent_tail[160];
    uint32_t recent_len;
    uint64_t total_observed_chars;
    uint64_t replacement_char_count;
    uint64_t control_char_count;
    uint64_t repeat_run_reject_count;
    uint64_t recent_repeat_reject_count;
} VspecRuntimeOutputGuard;

typedef struct VspecRuntimeOutputGuardReport {
    int integrity_pass;
    uint64_t total_observed_chars;
    uint64_t replacement_char_count;
    uint64_t control_char_count;
    uint64_t repeat_run_reject_count;
    uint64_t recent_repeat_reject_count;
} VspecRuntimeOutputGuardReport;

void vspec_output_guard_config_default(VspecRuntimeOutputGuardConfig* config);
void vspec_output_guard_init(VspecRuntimeOutputGuard* guard, const VspecRuntimeOutputGuardConfig* config);
void vspec_output_guard_observe(VspecRuntimeOutputGuard* guard, const char* text_fragment);
int vspec_output_guard_allow(VspecRuntimeOutputGuard* guard, const char* text_fragment);
float vspec_output_guard_score_adjustment(const VspecRuntimeOutputGuard* guard, const char* text_fragment);
void vspec_output_guard_report(const VspecRuntimeOutputGuard* guard, VspecRuntimeOutputGuardReport* report);

#endif
