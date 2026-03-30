#include "vspec/runtime/pattern_cache.h"

#include <stdlib.h>
#include <string.h>

static float vspec_clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

static int vspec_anf_pattern_cache_ensure_capacity(
    VspecAnfPatternCache* cache,
    size_t neuron_count,
    size_t hot_count
) {
    if (!cache) {
        return 0;
    }

    if (neuron_count > cache->neuron_capacity) {
        size_t target = (cache->neuron_capacity > 0U) ? cache->neuron_capacity : 1024U;
        uint32_t* grown;
        while (target < neuron_count) {
            target *= 2U;
        }
        grown = (uint32_t*)realloc(cache->current_marks, target * sizeof(uint32_t));
        if (!grown) {
            return 0;
        }
        if (target > cache->neuron_capacity) {
            (void)memset(grown + cache->neuron_capacity, 0, (target - cache->neuron_capacity) * sizeof(uint32_t));
        }
        cache->current_marks = grown;
        cache->neuron_capacity = target;
    }

    if (hot_count > cache->hot_capacity) {
        size_t target = (cache->hot_capacity > 0U) ? cache->hot_capacity : 64U;
        uint32_t* grown;
        while (target < hot_count) {
            target *= 2U;
        }
        grown = (uint32_t*)realloc(cache->prev_hot_indices, target * sizeof(uint32_t));
        if (!grown) {
            return 0;
        }
        cache->prev_hot_indices = grown;
        cache->hot_capacity = target;
    }

    return 1;
}

void vspec_anf_pattern_cache_config_default(VspecAnfPatternCacheConfig* cfg) {
    if (!cfg) {
        return;
    }
    cfg->max_changed_ratio_for_hit = 0.15f;
    cfg->changed_ratio_ema_decay = 0.85f;
    cfg->warmup_updates = 8U;
    cfg->max_idle_updates = 16U;
}

void vspec_anf_pattern_cache_init(VspecAnfPatternCache* cache, const VspecAnfPatternCacheConfig* cfg) {
    VspecAnfPatternCacheConfig local_cfg;
    if (!cache) {
        return;
    }
    vspec_anf_pattern_cache_config_default(&local_cfg);
    (void)memset(cache, 0, sizeof(*cache));
    cache->config = cfg ? *cfg : local_cfg;
    cache->config.max_changed_ratio_for_hit = vspec_clamp01(cache->config.max_changed_ratio_for_hit);
    cache->config.changed_ratio_ema_decay = vspec_clamp01(cache->config.changed_ratio_ema_decay);
    if (cache->config.changed_ratio_ema_decay < 0.01f) {
        cache->config.changed_ratio_ema_decay = 0.01f;
    }
    cache->current_epoch = 1U;
}

void vspec_anf_pattern_cache_reset(VspecAnfPatternCache* cache) {
    if (!cache) {
        return;
    }
    cache->prev_hot_count = 0U;
    cache->current_epoch = 1U;
    if (cache->current_marks && cache->neuron_capacity > 0U) {
        (void)memset(cache->current_marks, 0, cache->neuron_capacity * sizeof(uint32_t));
    }
    (void)memset(&cache->report, 0, sizeof(cache->report));
}

void vspec_anf_pattern_cache_destroy(VspecAnfPatternCache* cache) {
    if (!cache) {
        return;
    }
    free(cache->current_marks);
    free(cache->prev_hot_indices);
    cache->current_marks = NULL;
    cache->prev_hot_indices = NULL;
    cache->neuron_capacity = 0U;
    cache->hot_capacity = 0U;
    cache->prev_hot_count = 0U;
    cache->current_epoch = 0U;
    (void)memset(&cache->report, 0, sizeof(cache->report));
}

void vspec_anf_pattern_cache_update(
    VspecAnfPatternCache* cache,
    const uint32_t* hot_indices,
    size_t hot_count,
    size_t neuron_count
) {
    size_t intersection = 0U;
    size_t changed_count = 0U;
    float changed_ratio;
    float skip_ratio;
    float alpha;
    int is_warmup;
    int cache_hit;

    if (!cache || neuron_count == 0U) {
        return;
    }

    if (hot_count > 0U && !hot_indices) {
        return;
    }

    if (!vspec_anf_pattern_cache_ensure_capacity(cache, neuron_count, hot_count)) {
        return;
    }

    if (cache->current_epoch == 0xffffffffU) {
        (void)memset(cache->current_marks, 0, cache->neuron_capacity * sizeof(uint32_t));
        cache->current_epoch = 1U;
    }
    cache->current_epoch += 1U;

    for (size_t i = 0U; i < hot_count; ++i) {
        const size_t idx = (size_t)hot_indices[i];
        if (idx < neuron_count) {
            cache->current_marks[idx] = cache->current_epoch;
        }
    }

    for (size_t i = 0U; i < cache->prev_hot_count; ++i) {
        const size_t idx = (size_t)cache->prev_hot_indices[i];
        if (idx < neuron_count && cache->current_marks[idx] == cache->current_epoch) {
            intersection += 1U;
        }
    }

    changed_count = cache->prev_hot_count + hot_count - (2U * intersection);
    changed_ratio = (float)((double)changed_count / (double)neuron_count);
    changed_ratio = vspec_clamp01(changed_ratio);
    skip_ratio = 1.0f - changed_ratio;
    alpha = cache->config.changed_ratio_ema_decay;

    if (cache->report.updates == 0U) {
        cache->report.changed_ratio_ema = changed_ratio;
        cache->report.skip_ratio_ema = skip_ratio;
    } else {
        cache->report.changed_ratio_ema =
            (alpha * cache->report.changed_ratio_ema) + ((1.0f - alpha) * changed_ratio);
        cache->report.skip_ratio_ema =
            (alpha * cache->report.skip_ratio_ema) + ((1.0f - alpha) * skip_ratio);
    }

    cache->report.updates += 1U;
    cache->report.last_hot_count = (uint32_t)hot_count;
    cache->report.last_reused_count = (uint32_t)intersection;
    cache->report.last_changed_ratio = changed_ratio;
    cache->report.last_skip_ratio = skip_ratio;
    cache->report.confidence = 1.0f - cache->report.changed_ratio_ema;

    is_warmup = (cache->report.updates <= cache->config.warmup_updates) ? 1 : 0;
    cache_hit = (!is_warmup &&
        changed_ratio <= cache->config.max_changed_ratio_for_hit &&
        cache->report.idle_updates <= cache->config.max_idle_updates) ? 1 : 0;

    if (is_warmup) {
        cache->report.warmup_updates += 1U;
        cache->report.cache_misses += 1U;
    } else if (cache_hit) {
        cache->report.cache_hits += 1U;
        cache->report.idle_updates = 0U;
    } else {
        cache->report.cache_misses += 1U;
        cache->report.idle_updates += 1U;
    }

    if (hot_count > 0U) {
        (void)memcpy(cache->prev_hot_indices, hot_indices, hot_count * sizeof(uint32_t));
    }
    cache->prev_hot_count = hot_count;
}

void vspec_anf_pattern_cache_report(const VspecAnfPatternCache* cache, VspecAnfPatternCacheReport* report) {
    if (!report) {
        return;
    }
    if (!cache) {
        (void)memset(report, 0, sizeof(*report));
        return;
    }
    *report = cache->report;
}
