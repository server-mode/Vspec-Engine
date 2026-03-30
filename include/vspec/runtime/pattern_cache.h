#ifndef VSPEC_RUNTIME_PATTERN_CACHE_H
#define VSPEC_RUNTIME_PATTERN_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VspecAnfPatternCacheConfig {
    float max_changed_ratio_for_hit;
    float changed_ratio_ema_decay;
    uint32_t warmup_updates;
    uint32_t max_idle_updates;
} VspecAnfPatternCacheConfig;

typedef struct VspecAnfPatternCacheReport {
    uint32_t updates;
    uint32_t cache_hits;
    uint32_t cache_misses;
    uint32_t warmup_updates;
    uint32_t idle_updates;
    uint32_t last_hot_count;
    uint32_t last_reused_count;
    float last_changed_ratio;
    float changed_ratio_ema;
    float last_skip_ratio;
    float skip_ratio_ema;
    float confidence;
} VspecAnfPatternCacheReport;

typedef struct VspecAnfPatternCache {
    VspecAnfPatternCacheConfig config;
    uint32_t* current_marks;
    uint32_t current_epoch;
    uint32_t* prev_hot_indices;
    size_t prev_hot_count;
    size_t hot_capacity;
    size_t neuron_capacity;
    VspecAnfPatternCacheReport report;
} VspecAnfPatternCache;

void vspec_anf_pattern_cache_config_default(VspecAnfPatternCacheConfig* cfg);
void vspec_anf_pattern_cache_init(VspecAnfPatternCache* cache, const VspecAnfPatternCacheConfig* cfg);
void vspec_anf_pattern_cache_reset(VspecAnfPatternCache* cache);
void vspec_anf_pattern_cache_destroy(VspecAnfPatternCache* cache);
void vspec_anf_pattern_cache_update(
    VspecAnfPatternCache* cache,
    const uint32_t* hot_indices,
    size_t hot_count,
    size_t neuron_count
);
void vspec_anf_pattern_cache_report(const VspecAnfPatternCache* cache, VspecAnfPatternCacheReport* report);

#ifdef __cplusplus
}
#endif

#endif
