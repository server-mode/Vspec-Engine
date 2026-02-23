#ifndef VSPEC_RUNTIME_LANGUAGE_STRUCTURE_GUARD_H
#define VSPEC_RUNTIME_LANGUAGE_STRUCTURE_GUARD_H

#include <stddef.h>
#include <stdint.h>

#define VSPEC_LANGUAGE_STRUCTURE_MAX_SECTIONS 64u

typedef struct VspecLanguageStructureGuardConfig {
    float strictness;
    int require_balanced_code_fence;
    int reject_control_chars;
} VspecLanguageStructureGuardConfig;

typedef struct VspecLanguageTokenCompensationConfig {
    float heading_reward;
    float fence_reward;
    float noise_penalty;
    float max_credit;
    float decay;
} VspecLanguageTokenCompensationConfig;

typedef struct VspecLanguageStructureGuard {
    VspecLanguageStructureGuardConfig config;
    VspecLanguageTokenCompensationConfig compensation;

    uint32_t expected_sections[VSPEC_LANGUAGE_STRUCTURE_MAX_SECTIONS];
    uint32_t expected_section_count;

    uint32_t seen_sections[VSPEC_LANGUAGE_STRUCTURE_MAX_SECTIONS];
    uint32_t seen_section_count;

    uint64_t total_observed_chars;
    uint64_t replacement_char_count;
    uint64_t control_char_count;
    uint64_t symbol_char_count;
    uint64_t alpha_num_char_count;

    uint32_t code_fence_toggle_count;
    float compensation_credit;
} VspecLanguageStructureGuard;

typedef struct VspecLanguageStructureGuardReport {
    int integrity_pass;
    uint32_t expected_section_count;
    uint32_t seen_section_count;
    float section_coverage;
    int code_fence_balanced;
    uint64_t replacement_char_count;
    uint64_t control_char_count;
    uint64_t total_observed_chars;
} VspecLanguageStructureGuardReport;

void vspec_language_structure_guard_config_default(VspecLanguageStructureGuardConfig* config);
void vspec_language_token_compensation_config_default(VspecLanguageTokenCompensationConfig* config);

void vspec_language_structure_guard_init(
    VspecLanguageStructureGuard* guard,
    const VspecLanguageStructureGuardConfig* config,
    const char* prompt_text
);

void vspec_language_structure_guard_observe_text(
    VspecLanguageStructureGuard* guard,
    const char* text_fragment
);

int vspec_language_structure_guard_allow_text(
    const VspecLanguageStructureGuard* guard,
    const char* text_fragment
);

float vspec_language_structure_guard_score_adjustment(
    const VspecLanguageStructureGuard* guard,
    const char* text_fragment
);

float vspec_language_structure_guard_token_compensation(
    VspecLanguageStructureGuard* guard,
    const char* text_fragment
);

void vspec_language_structure_guard_report(
    const VspecLanguageStructureGuard* guard,
    VspecLanguageStructureGuardReport* report
);

#endif