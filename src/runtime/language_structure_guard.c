#include "vspec/runtime/language_structure_guard.h"

#include <ctype.h>
#include <stddef.h>
#include <string.h>

static int vspec_ascii_startswith_ci(const char* text, const char* prefix) {
    if (!text || !prefix) {
        return 0;
    }

    while (*prefix) {
        if (*text == '\0') {
            return 0;
        }
        if (toupper((unsigned char)*text) != toupper((unsigned char)*prefix)) {
            return 0;
        }
        ++text;
        ++prefix;
    }
    return 1;
}

static int vspec_parse_section_id_after_keyword(const char* pos, uint32_t* section_id) {
    if (!pos || !section_id) {
        return 0;
    }

    pos += 7;
    while (*pos && isspace((unsigned char)*pos)) {
        ++pos;
    }
    if (!isdigit((unsigned char)*pos)) {
        return 0;
    }

    uint32_t value = 0;
    while (isdigit((unsigned char)*pos)) {
        value = (value * 10u) + (uint32_t)(*pos - '0');
        ++pos;
    }

    if (value == 0u) {
        return 0;
    }

    *section_id = value;
    return 1;
}

static int vspec_guard_has_section(const uint32_t* sections, uint32_t count, uint32_t id) {
    uint32_t index = 0u;
    for (index = 0u; index < count; ++index) {
        if (sections[index] == id) {
            return 1;
        }
    }
    return 0;
}

static void vspec_guard_add_section(uint32_t* sections, uint32_t* count, uint32_t id) {
    if (!sections || !count || id == 0u) {
        return;
    }
    if (*count >= VSPEC_LANGUAGE_STRUCTURE_MAX_SECTIONS) {
        return;
    }
    if (vspec_guard_has_section(sections, *count, id)) {
        return;
    }
    sections[*count] = id;
    *count += 1u;
}

static void vspec_extract_expected_sections(VspecLanguageStructureGuard* guard, const char* text) {
    if (!guard || !text) {
        return;
    }

    const char* cursor = text;
    while (*cursor) {
        if (vspec_ascii_startswith_ci(cursor, "SECTION")) {
            uint32_t section_id = 0u;
            if (vspec_parse_section_id_after_keyword(cursor, &section_id)) {
                vspec_guard_add_section(
                    guard->expected_sections,
                    &guard->expected_section_count,
                    section_id
                );
            }
        }
        ++cursor;
    }
}

static void vspec_extract_seen_sections(VspecLanguageStructureGuard* guard, const char* text) {
    if (!guard || !text) {
        return;
    }

    const char* cursor = text;
    while (*cursor) {
        if (*cursor == '#') {
            const char* probe = cursor + 1;
            while (*probe && isspace((unsigned char)*probe)) {
                ++probe;
            }
            if (vspec_ascii_startswith_ci(probe, "SECTION")) {
                uint32_t section_id = 0u;
                if (vspec_parse_section_id_after_keyword(probe, &section_id)) {
                    vspec_guard_add_section(
                        guard->seen_sections,
                        &guard->seen_section_count,
                        section_id
                    );
                }
            }
        }
        ++cursor;
    }
}

static uint32_t vspec_count_code_fences(const char* text) {
    if (!text) {
        return 0u;
    }

    uint32_t count = 0u;
    const char* cursor = text;
    while (*cursor) {
        if (cursor[0] == '`' && cursor[1] == '`' && cursor[2] == '`') {
            count += 1u;
            cursor += 3;
            continue;
        }
        ++cursor;
    }
    return count;
}

static void vspec_count_text_features(
    const char* text,
    uint64_t* total_chars,
    uint64_t* replacement_chars,
    uint64_t* control_chars,
    uint64_t* symbol_chars,
    uint64_t* alpha_num_chars
) {
    if (!text) {
        return;
    }

    const unsigned char* cursor = (const unsigned char*)text;
    while (*cursor) {
        const unsigned char c = *cursor;
        if (total_chars) {
            *total_chars += 1u;
        }

        if (c == 0xEF && cursor[1] != '\0' && cursor[2] != '\0' && cursor[1] == 0xBF && cursor[2] == 0xBD) {
            if (replacement_chars) {
                *replacement_chars += 1u;
            }
            cursor += 3;
            continue;
        }

        if ((c < 32u && c != '\n' && c != '\r' && c != '\t') || c == 127u) {
            if (control_chars) {
                *control_chars += 1u;
            }
        } else if (isalnum(c)) {
            if (alpha_num_chars) {
                *alpha_num_chars += 1u;
            }
        } else if (!isspace(c)) {
            if (symbol_chars) {
                *symbol_chars += 1u;
            }
        }

        cursor += 1;
    }
}

static float vspec_clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

void vspec_language_structure_guard_config_default(VspecLanguageStructureGuardConfig* config) {
    if (!config) {
        return;
    }
    config->strictness = 0.6f;
    config->require_balanced_code_fence = 1;
    config->reject_control_chars = 1;
}

void vspec_language_token_compensation_config_default(VspecLanguageTokenCompensationConfig* config) {
    if (!config) {
        return;
    }
    config->heading_reward = 0.20f;
    config->fence_reward = 0.10f;
    config->noise_penalty = 0.25f;
    config->max_credit = 0.80f;
    config->decay = 0.04f;
}

void vspec_language_structure_guard_init(
    VspecLanguageStructureGuard* guard,
    const VspecLanguageStructureGuardConfig* config,
    const char* prompt_text
) {
    if (!guard) {
        return;
    }

    (void)memset(guard, 0, sizeof(*guard));
    vspec_language_structure_guard_config_default(&guard->config);
    vspec_language_token_compensation_config_default(&guard->compensation);
    if (config) {
        guard->config = *config;
    }
    if (guard->config.strictness < 0.0f) {
        guard->config.strictness = 0.0f;
    }
    if (guard->config.strictness > 1.0f) {
        guard->config.strictness = 1.0f;
    }

    vspec_extract_expected_sections(guard, prompt_text);
}

void vspec_language_structure_guard_observe_text(
    VspecLanguageStructureGuard* guard,
    const char* text_fragment
) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return;
    }

    guard->code_fence_toggle_count += vspec_count_code_fences(text_fragment);
    vspec_extract_seen_sections(guard, text_fragment);

    vspec_count_text_features(
        text_fragment,
        &guard->total_observed_chars,
        &guard->replacement_char_count,
        &guard->control_char_count,
        &guard->symbol_char_count,
        &guard->alpha_num_char_count
    );
}

int vspec_language_structure_guard_allow_text(
    const VspecLanguageStructureGuard* guard,
    const char* text_fragment
) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return 1;
    }

    uint64_t total_chars = 0u;
    uint64_t replacement_chars = 0u;
    uint64_t control_chars = 0u;
    uint64_t symbol_chars = 0u;
    uint64_t alpha_num_chars = 0u;

    vspec_count_text_features(
        text_fragment,
        &total_chars,
        &replacement_chars,
        &control_chars,
        &symbol_chars,
        &alpha_num_chars
    );

    if (replacement_chars > 0u && guard->config.strictness >= 0.3f) {
        return 0;
    }

    if (guard->config.reject_control_chars && control_chars > 0u && guard->config.strictness >= 0.5f) {
        return 0;
    }

    if (guard->config.strictness >= 0.8f && total_chars >= 10u) {
        const double ratio = (double)symbol_chars / (double)total_chars;
        if (ratio > 0.60) {
            return 0;
        }
    }

    return 1;
}

float vspec_language_structure_guard_score_adjustment(
    const VspecLanguageStructureGuard* guard,
    const char* text_fragment
) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return 0.0f;
    }

    float adjustment = 0.0f;

    if (guard->expected_section_count > 0u && text_fragment[0] == '#') {
        const char* probe = text_fragment + 1;
        while (*probe && isspace((unsigned char)*probe)) {
            ++probe;
        }
        if (vspec_ascii_startswith_ci(probe, "SECTION")) {
            uint32_t section_id = 0u;
            if (vspec_parse_section_id_after_keyword(probe, &section_id)) {
                if (vspec_guard_has_section(guard->expected_sections, guard->expected_section_count, section_id) &&
                    !vspec_guard_has_section(guard->seen_sections, guard->seen_section_count, section_id)) {
                    adjustment += 0.25f * guard->config.strictness;
                }
            }
        }
    }

    if (guard->config.require_balanced_code_fence &&
        text_fragment[0] == '`' &&
        text_fragment[1] == '`' &&
        text_fragment[2] == '`') {
        adjustment += 0.10f;
    }

    adjustment += guard->compensation_credit;

    return adjustment;
}

float vspec_language_structure_guard_token_compensation(
    VspecLanguageStructureGuard* guard,
    const char* text_fragment
) {
    if (!guard || !text_fragment || text_fragment[0] == '\0') {
        return 0.0f;
    }

    float delta = 0.0f;

    if (text_fragment[0] == '#') {
        const char* probe = text_fragment + 1;
        while (*probe && isspace((unsigned char)*probe)) {
            ++probe;
        }
        if (vspec_ascii_startswith_ci(probe, "SECTION")) {
            uint32_t section_id = 0u;
            if (vspec_parse_section_id_after_keyword(probe, &section_id)) {
                if (vspec_guard_has_section(guard->expected_sections, guard->expected_section_count, section_id) &&
                    !vspec_guard_has_section(guard->seen_sections, guard->seen_section_count, section_id)) {
                    delta += guard->compensation.heading_reward * guard->config.strictness;
                }
            }
        }
    }

    if (text_fragment[0] == '`' && text_fragment[1] == '`' && text_fragment[2] == '`') {
        delta += guard->compensation.fence_reward;
    }

    if (!vspec_language_structure_guard_allow_text(guard, text_fragment)) {
        delta -= guard->compensation.noise_penalty;
    }

    guard->compensation_credit += delta;
    guard->compensation_credit -= guard->compensation.decay;
    guard->compensation_credit = vspec_clampf(
        guard->compensation_credit,
        -guard->compensation.max_credit,
        guard->compensation.max_credit
    );

    return guard->compensation_credit;
}

void vspec_language_structure_guard_report(
    const VspecLanguageStructureGuard* guard,
    VspecLanguageStructureGuardReport* report
) {
    if (!guard || !report) {
        return;
    }

    uint32_t matched_sections = 0u;
    uint32_t index = 0u;

    for (index = 0u; index < guard->expected_section_count; ++index) {
        if (vspec_guard_has_section(guard->seen_sections, guard->seen_section_count, guard->expected_sections[index])) {
            matched_sections += 1u;
        }
    }

    const int code_fence_balanced = ((guard->code_fence_toggle_count % 2u) == 0u) ? 1 : 0;
    float coverage = 1.0f;
    if (guard->expected_section_count > 0u) {
        coverage = (float)matched_sections / (float)guard->expected_section_count;
    }

    report->expected_section_count = guard->expected_section_count;
    report->seen_section_count = guard->seen_section_count;
    report->section_coverage = coverage;
    report->code_fence_balanced = code_fence_balanced;
    report->replacement_char_count = guard->replacement_char_count;
    report->control_char_count = guard->control_char_count;
    report->total_observed_chars = guard->total_observed_chars;

    report->integrity_pass = 1;
    if (guard->config.strictness >= 0.5f && guard->replacement_char_count > 0u) {
        report->integrity_pass = 0;
    }
    if (guard->config.reject_control_chars && guard->config.strictness >= 0.6f && guard->control_char_count > 0u) {
        report->integrity_pass = 0;
    }
    if (guard->config.require_balanced_code_fence && !code_fence_balanced && guard->config.strictness >= 0.6f) {
        report->integrity_pass = 0;
    }
    if (guard->expected_section_count > 0u && coverage < (0.50f + (0.30f * guard->config.strictness))) {
        report->integrity_pass = 0;
    }
}