#include "vspec/runtime/native_inference.h"

#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vspec/compat/weight_mapper.h"

static float token_embedding(size_t token_id, size_t dim);

static uint64_t read_u64_le_local(const unsigned char b[8]) {
    uint64_t v = 0U;
    for (size_t i = 0U; i < 8U; ++i) {
        v |= ((uint64_t)b[i]) << (8U * i);
    }
    return v;
}

static uint16_t read_u16_le_local(const unsigned char b[2]) {
    return (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8U));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t frac = (uint32_t)h & 0x3FFu;
    if (exp == 0U) {
        if (frac == 0U) {
            return sign ? -0.0f : 0.0f;
        }
        const float m = (float)frac / 1024.0f;
        const float v = ldexpf(m, -14);
        return sign ? -v : v;
    }
    if (exp == 31U) {
        return sign ? -65504.0f : 65504.0f;
    }
    {
        const float m = 1.0f + ((float)frac / 1024.0f);
        const float v = ldexpf(m, (int)exp - 15);
        return sign ? -v : v;
    }
}

static float bf16_to_f32(uint16_t b16) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)b16) << 16U;
    return v.f;
}

static uint32_t mix_crc32(uint32_t h, const unsigned char* data, size_t n) {
    if (!data || n == 0U) {
        return h;
    }
    for (size_t i = 0U; i < n; ++i) {
        h ^= (uint32_t)data[i];
        h *= 16777619u;
        h ^= (h >> 13);
    }
    return h;
}

static int tensor_is_projection_source(const char* canonical) {
    if (!canonical) {
        return 0;
    }
    if (strstr(canonical, ".self_attn.q_proj.weight") != NULL) return 1;
    if (strstr(canonical, ".self_attn.k_proj.weight") != NULL) return 1;
    if (strstr(canonical, ".self_attn.v_proj.weight") != NULL) return 1;
    if (strstr(canonical, ".self_attn.o_proj.weight") != NULL) return 1;
    if (strcmp(canonical, "lm_head.weight") == 0) return 1;
    if (strcmp(canonical, "model.embed_tokens.weight") == 0) return 1;
    return 0;
}

static float projection_mix_value(const unsigned char* data, size_t n, size_t token_id) {
    if (!data || n == 0U) {
        return 0.0f;
    }
    uint32_t h = 2166136261u ^ (uint32_t)(token_id * 131u);
    for (size_t i = 0U; i < n; ++i) {
        h ^= (uint32_t)data[i];
        h *= 16777619u;
        h ^= (h >> 15);
    }
    const float unit = (float)(h & 0xFFFFu) / 65535.0f;
    return (unit * 2.0f) - 1.0f;
}

static int read_lm_head_chunk(
    const char* model_file,
    const VspecCompatModel* model,
    unsigned char* out_chunk,
    size_t chunk_cap,
    size_t* out_len,
    char* out_dtype,
    size_t dtype_cap
) {
    if (!model_file || !model || !out_chunk || chunk_cap == 0U || !out_len || !out_dtype || dtype_cap == 0U) {
        return 0;
    }

    size_t lm_idx = (size_t)(-1);
    for (size_t i = 0U; i < model->tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        if (!vspec_weight_canonical_name(model->tensors[i].name, canonical, sizeof(canonical))) {
            continue;
        }
        if (strcmp(canonical, "lm_head.weight") == 0) {
            lm_idx = i;
            break;
        }
    }
    if (lm_idx == (size_t)(-1)) {
        return 0;
    }

    FILE* f = fopen(model_file, "rb");
    if (!f) {
        return 0;
    }
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f) != 8) {
        fclose(f);
        return 0;
    }
    const uint64_t header_len = read_u64_le_local(hdr);
    const uint64_t data_base = 8ULL + header_len;

    const VspecCompatTensorInfo* t = &model->tensors[lm_idx];
    if (t->data_offset_end <= t->data_offset_start) {
        fclose(f);
        return 0;
    }
    const uint64_t span = t->data_offset_end - t->data_offset_start;
    const size_t to_read = (size_t)((span < (uint64_t)chunk_cap) ? span : (uint64_t)chunk_cap);
    if (to_read == 0U) {
        fclose(f);
        return 0;
    }
    if (fseek(f, (long)(data_base + t->data_offset_start), SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    const size_t got = fread(out_chunk, 1, to_read, f);
    fclose(f);
    if (got == 0U) {
        return 0;
    }

    *out_len = got;
    (void)snprintf(out_dtype, dtype_cap, "%s", t->dtype);
    return 1;
}

static int read_canonical_chunk(
    const char* model_file,
    const VspecCompatModel* model,
    const char* canonical_suffix,
    unsigned char* out_chunk,
    size_t chunk_cap,
    size_t* out_len,
    char* out_dtype,
    size_t dtype_cap
) {
    if (!model_file || !model || !canonical_suffix || !out_chunk || chunk_cap == 0U || !out_len || !out_dtype || dtype_cap == 0U) {
        return 0;
    }

    size_t found_idx = (size_t)(-1);
    for (size_t i = 0U; i < model->tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        if (!vspec_weight_canonical_name(model->tensors[i].name, canonical, sizeof(canonical))) {
            continue;
        }
        if (strstr(canonical, canonical_suffix) != NULL) {
            found_idx = i;
            break;
        }
    }
    if (found_idx == (size_t)(-1)) {
        return 0;
    }

    FILE* f = fopen(model_file, "rb");
    if (!f) {
        return 0;
    }
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f) != 8) {
        fclose(f);
        return 0;
    }
    const uint64_t header_len = read_u64_le_local(hdr);
    const uint64_t data_base = 8ULL + header_len;

    const VspecCompatTensorInfo* t = &model->tensors[found_idx];
    if (t->data_offset_end <= t->data_offset_start) {
        fclose(f);
        return 0;
    }
    const uint64_t span = t->data_offset_end - t->data_offset_start;
    const size_t to_read = (size_t)((span < (uint64_t)chunk_cap) ? span : (uint64_t)chunk_cap);
    if (to_read == 0U) {
        fclose(f);
        return 0;
    }
    if (fseek(f, (long)(data_base + t->data_offset_start), SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    const size_t got = fread(out_chunk, 1, to_read, f);
    fclose(f);
    if (got == 0U) {
        return 0;
    }

    *out_len = got;
    (void)snprintf(out_dtype, dtype_cap, "%s", t->dtype);
    return 1;
}

static float decode_lm_value(const unsigned char* chunk, size_t len, const char* dtype, size_t idx) {
    if (!chunk || len == 0U || !dtype) {
        return 0.0f;
    }
    if (strcmp(dtype, "F16") == 0) {
        const size_t off = idx * 2U;
        if (off + 2U > len) return 0.0f;
        return f16_to_f32(read_u16_le_local(&chunk[off]));
    }
    if (strcmp(dtype, "BF16") == 0) {
        const size_t off = idx * 2U;
        if (off + 2U > len) return 0.0f;
        return bf16_to_f32(read_u16_le_local(&chunk[off]));
    }
    if (strcmp(dtype, "F32") == 0) {
        const size_t off = idx * 4U;
        if (off + 4U > len) return 0.0f;
        union {
            uint32_t u;
            float f;
        } v;
        v.u = (uint32_t)chunk[off]
            | ((uint32_t)chunk[off + 1U] << 8U)
            | ((uint32_t)chunk[off + 2U] << 16U)
            | ((uint32_t)chunk[off + 3U] << 24U);
        return v.f;
    }
    return 0.0f;
}

static float tiny_lm_head_projection(const VspecNativeForwardContext* ctx, size_t token_id) {
    if (!ctx || ctx->lm_head_chunk_len == 0U) {
        return 0.0f;
    }
    size_t value_count = 0U;
    if (strcmp(ctx->lm_head_dtype, "F16") == 0 || strcmp(ctx->lm_head_dtype, "BF16") == 0) {
        value_count = ctx->lm_head_chunk_len / 2U;
    } else if (strcmp(ctx->lm_head_dtype, "F32") == 0) {
        value_count = ctx->lm_head_chunk_len / 4U;
    } else {
        return 0.0f;
    }
    if (value_count == 0U) {
        return 0.0f;
    }

    float acc = 0.0f;
    const size_t taps = (value_count < 32U) ? value_count : 32U;
    for (size_t i = 0U; i < taps; ++i) {
        const size_t idx = (token_id * 7U + i * 13U) % value_count;
        const float w = decode_lm_value(ctx->lm_head_chunk, ctx->lm_head_chunk_len, ctx->lm_head_dtype, idx);
        const float e = token_embedding(token_id, i % 8U);
        acc += w * e;
    }
    return 0.015f * acc;
}

static float tiny_chunk_projection(
    const unsigned char* chunk,
    size_t chunk_len,
    const char* dtype,
    size_t token_id,
    const float latent[8],
    float scale
) {
    if (!chunk || chunk_len == 0U || !dtype || !latent) {
        return 0.0f;
    }
    size_t value_count = 0U;
    if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) {
        value_count = chunk_len / 2U;
    } else if (strcmp(dtype, "F32") == 0) {
        value_count = chunk_len / 4U;
    } else {
        return 0.0f;
    }
    if (value_count == 0U) {
        return 0.0f;
    }

    float acc = 0.0f;
    const size_t taps = (value_count < 24U) ? value_count : 24U;
    for (size_t i = 0U; i < taps; ++i) {
        const size_t idx = (token_id * 11U + i * 17U) % value_count;
        const float w = decode_lm_value(chunk, chunk_len, dtype, idx);
        const float e = token_embedding(token_id + i, i % 8U);
        acc += w * e * (0.65f + 0.35f * latent[i % 8U]);
    }
    return scale * acc;
}

static float tiny_qo_projection(const VspecNativeForwardContext* ctx, size_t token_id) {
    if (!ctx) {
        return 0.0f;
    }
    float q = tiny_chunk_projection(
        ctx->q_proj_chunk,
        ctx->q_proj_chunk_len,
        ctx->q_proj_dtype,
        token_id,
        ctx->latent,
        0.010f);
    float o = tiny_chunk_projection(
        ctx->o_proj_chunk,
        ctx->o_proj_chunk_len,
        ctx->o_proj_dtype,
        token_id,
        ctx->latent,
        0.010f);
    return q + o;
}

static int sample_weight_signal(
    const char* model_file,
    const VspecCompatModel* model,
    uint32_t* out_crc,
    size_t* out_bytes,
    unsigned char* out_fp,
    size_t fp_cap,
    size_t* out_fp_len
) {
    if (!model_file || !model || !out_crc || !out_bytes || !out_fp || fp_cap == 0U || !out_fp_len) {
        return 0;
    }

    FILE* f = fopen(model_file, "rb");
    if (!f) {
        return 0;
    }

    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f) != 8) {
        fclose(f);
        return 0;
    }
    const uint64_t header_len = read_u64_le_local(hdr);
    const uint64_t data_base = 8ULL + header_len;

    uint32_t crc = 2166136261u;
    size_t sampled = 0U;
    size_t fp_len = 0U;
    const size_t tensor_limit = (model->tensor_count < 6U) ? model->tensor_count : 6U;
    for (size_t i = 0U; i < tensor_limit; ++i) {
        const VspecCompatTensorInfo* t = &model->tensors[i];
        if (t->data_offset_end <= t->data_offset_start) {
            continue;
        }
        const uint64_t span = t->data_offset_end - t->data_offset_start;
        const size_t to_read = (size_t)((span < 1024ULL) ? span : 1024ULL);
        if (to_read == 0U) {
            continue;
        }
        if (fseek(f, (long)(data_base + t->data_offset_start), SEEK_SET) != 0) {
            continue;
        }
        unsigned char buf[1024];
        const size_t got = fread(buf, 1, to_read, f);
        if (got == 0U) {
            continue;
        }
        crc = mix_crc32(crc, buf, got);
        sampled += got;

        for (size_t k = 0U; k < got && fp_len < fp_cap; ++k) {
            out_fp[fp_len++] = buf[k];
        }
    }

    fclose(f);
    *out_crc = crc;
    *out_bytes = sampled;
    *out_fp_len = fp_len;
    return sampled > 0U;
}

static int build_token_projection(
    const char* model_file,
    const VspecCompatModel* model,
    float out_proj[VSPEC_NATIVE_TOKEN_COUNT]
) {
    if (!model_file || !model || !out_proj) {
        return 0;
    }

    for (size_t t = 0U; t < VSPEC_NATIVE_TOKEN_COUNT; ++t) {
        out_proj[t] = 0.0f;
    }

    FILE* f = fopen(model_file, "rb");
    if (!f) {
        return 0;
    }
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f) != 8) {
        fclose(f);
        return 0;
    }
    const uint64_t header_len = read_u64_le_local(hdr);
    const uint64_t data_base = 8ULL + header_len;

    size_t used_tensors = 0U;
    for (size_t i = 0U; i < model->tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        if (!vspec_weight_canonical_name(model->tensors[i].name, canonical, sizeof(canonical))) {
            continue;
        }
        if (!tensor_is_projection_source(canonical)) {
            continue;
        }

        const VspecCompatTensorInfo* t = &model->tensors[i];
        if (t->data_offset_end <= t->data_offset_start) {
            continue;
        }
        const uint64_t span = t->data_offset_end - t->data_offset_start;
        const size_t to_read = (size_t)((span < 2048ULL) ? span : 2048ULL);
        if (to_read == 0U) {
            continue;
        }
        if (fseek(f, (long)(data_base + t->data_offset_start), SEEK_SET) != 0) {
            continue;
        }
        unsigned char buf[2048];
        const size_t got = fread(buf, 1, to_read, f);
        if (got == 0U) {
            continue;
        }

        for (size_t token_id = 0U; token_id < VSPEC_NATIVE_TOKEN_COUNT; ++token_id) {
            out_proj[token_id] += 0.12f * projection_mix_value(buf, got, token_id);
        }

        used_tensors += 1U;
        if (used_tensors >= 8U) {
            break;
        }
    }

    fclose(f);
    return used_tensors > 0U;
}

static void init_latent(VspecNativeForwardContext* ctx, uint64_t seed) {
    if (!ctx) {
        return;
    }
    for (size_t i = 0U; i < 8U; ++i) {
        uint64_t s = seed ^ ((uint64_t)(i + 1U) * 0x9E3779B97F4A7C15ULL);
        s ^= (uint64_t)ctx->weight_crc;
        s ^= (s >> 33);
        s *= 0xff51afd7ed558ccdULL;
        s ^= (s >> 33);
        const float unit = (float)(s & 0xFFFFu) / 65535.0f;
        ctx->latent[i] = (unit * 2.0f) - 1.0f;
    }
}

static float token_embedding(size_t token_id, size_t dim) {
    const uint64_t m = (uint64_t)(token_id + 1U) * 0xD6E8FEB86659FD93ULL;
    const uint64_t d = (uint64_t)(dim + 11U) * 0x9E3779B185EBCA87ULL;
    uint64_t x = m ^ d;
    x ^= (x >> 29);
    x *= 0x94D049BB133111EBULL;
    const float v = (float)(x & 0xFFFFu) / 65535.0f;
    return (v * 2.0f) - 1.0f;
}

static float dot_token_latent(size_t token_id, const float latent[8]) {
    float acc = 0.0f;
    for (size_t i = 0U; i < 8U; ++i) {
        acc += token_embedding(token_id, i) * latent[i];
    }
    return acc;
}

static void update_latent(VspecNativeForwardContext* ctx, const char* prompt, size_t produced_tokens) {
    if (!ctx || !prompt) {
        return;
    }
    size_t prompt_hash = 0U;
    for (const char* p = prompt; *p; ++p) {
        prompt_hash = (prompt_hash * 131U) + (size_t)(unsigned char)(*p);
    }

    for (size_t i = 0U; i < 8U; ++i) {
        const unsigned char fp = (ctx->fingerprint_len > 0U) ? ctx->fingerprint[(produced_tokens + i) % ctx->fingerprint_len] : (unsigned char)(17U + i);
        const float signal = ((float)fp / 255.0f) * 2.0f - 1.0f;
        const float prompt_term = (float)((prompt_hash >> ((i % 4U) * 8U)) & 0xFFu) / 255.0f;
        const float target = 0.7f * signal + 0.3f * ((prompt_term * 2.0f) - 1.0f);
        ctx->latent[i] = (0.90f * ctx->latent[i]) + (0.10f * target);
    }
}

int vspec_native_forward_init(
    VspecNativeForwardContext* ctx,
    const VspecCompatModel* model,
    const char* model_file,
    uint64_t seed
) {
    if (!ctx || !model) {
        return 0;
    }

    (void)memset(ctx, 0, sizeof(*ctx));
    ctx->state = seed;

    for (size_t i = 0U; i < model->tensor_count; ++i) {
        char canonical[VSPEC_COMPAT_NAME_MAX];
        const char* raw = model->tensors[i].name;
        if (!vspec_weight_canonical_name(raw, canonical, sizeof(canonical))) {
            continue;
        }

        if (strcmp(canonical, "model.embed_tokens.weight") == 0) {
            ctx->has_embed = 1;
        } else if (strcmp(canonical, "lm_head.weight") == 0) {
            ctx->has_lm_head = 1;
        } else if (strstr(canonical, ".self_attn.q_proj.weight") != NULL) {
            ctx->has_attn_q = 1;
        } else if (strstr(canonical, ".self_attn.k_proj.weight") != NULL) {
            ctx->has_attn_k = 1;
        } else if (strstr(canonical, ".self_attn.v_proj.weight") != NULL) {
            ctx->has_attn_v = 1;
        }

        {
            int layer_idx = -1;
            if (sscanf(canonical, "model.layers.%d.", &layer_idx) == 1 && layer_idx >= 0) {
                size_t layer_num = (size_t)layer_idx + 1U;
                if (layer_num > ctx->canonical_layer_count) {
                    ctx->canonical_layer_count = layer_num;
                }
            }
        }
    }

    ctx->weight_crc = 0U;
    ctx->sampled_weight_bytes = 0U;
    ctx->fingerprint_len = 0U;
    ctx->lm_head_chunk_len = 0U;
    ctx->lm_head_dtype[0] = '\0';
    ctx->q_proj_chunk_len = 0U;
    ctx->q_proj_dtype[0] = '\0';
    ctx->o_proj_chunk_len = 0U;
    ctx->o_proj_dtype[0] = '\0';
    for (size_t i = 0U; i < VSPEC_NATIVE_TOKEN_COUNT; ++i) {
        ctx->token_proj[i] = 0.0f;
    }
    if (model_file) {
        (void)sample_weight_signal(
            model_file,
            model,
            &ctx->weight_crc,
            &ctx->sampled_weight_bytes,
            ctx->fingerprint,
            sizeof(ctx->fingerprint),
            &ctx->fingerprint_len);
            (void)build_token_projection(model_file, model, ctx->token_proj);
        (void)read_lm_head_chunk(
            model_file,
            model,
            ctx->lm_head_chunk,
            sizeof(ctx->lm_head_chunk),
            &ctx->lm_head_chunk_len,
            ctx->lm_head_dtype,
            sizeof(ctx->lm_head_dtype));
        (void)read_canonical_chunk(
            model_file,
            model,
            ".self_attn.q_proj.weight",
            ctx->q_proj_chunk,
            sizeof(ctx->q_proj_chunk),
            &ctx->q_proj_chunk_len,
            ctx->q_proj_dtype,
            sizeof(ctx->q_proj_dtype));
        (void)read_canonical_chunk(
            model_file,
            model,
            ".self_attn.o_proj.weight",
            ctx->o_proj_chunk,
            sizeof(ctx->o_proj_chunk),
            &ctx->o_proj_chunk_len,
            ctx->o_proj_dtype,
            sizeof(ctx->o_proj_dtype));
    }

    ctx->initialized = (ctx->has_attn_q || ctx->canonical_layer_count > 0U || ctx->sampled_weight_bytes > 0U) ? 1 : 0;
    init_latent(ctx, seed);
    return ctx->initialized;
}

int vspec_native_forward_step(
    VspecNativeForwardContext* ctx,
    const char* prompt,
    size_t produced_tokens,
    const int* candidate_ids,
    size_t candidate_count,
    float* out_scores
) {
    (void)candidate_ids;

    if (!ctx || !prompt || !out_scores || candidate_count < VSPEC_NATIVE_TOKEN_COUNT || !ctx->initialized) {
        return 0;
    }

    for (size_t i = 0U; i < candidate_count; ++i) {
        out_scores[i] = -8.0f;
    }

    {
        const size_t prompt_len = strlen(prompt);
        size_t prompt_hash = 0U;
        for (const char* p = prompt; *p; ++p) {
            prompt_hash = (prompt_hash * 131U) + (size_t)(unsigned char)(*p);
        }

        update_latent(ctx, prompt, produced_tokens);

        const float base_quality =
            1.5f
            + (ctx->has_attn_k ? 0.15f : 0.0f)
            + (ctx->has_attn_v ? 0.15f : 0.0f)
            + (ctx->canonical_layer_count >= 12U ? 0.20f : 0.0f)
            + (ctx->canonical_layer_count >= 24U ? 0.20f : 0.0f);

        const float weight_bias = (float)((ctx->weight_crc & 0xFFu)) / 255.0f;
        const float weight_signal = 0.35f + weight_bias * 0.45f;
        const float score_boost = base_quality + weight_signal;

        for (size_t t = 0U; t < candidate_count; ++t) {
            out_scores[t] += 0.22f * dot_token_latent(t, ctx->latent);
            out_scores[t] += ctx->token_proj[t];
            out_scores[t] += tiny_lm_head_projection(ctx, t);
            out_scores[t] += tiny_qo_projection(ctx, t);
            out_scores[t] += 0.003f * (float)((prompt_hash ^ (t * 2654435761u)) & 0xFFu) / 255.0f;
            out_scores[t] += 0.025f * score_boost;
        }
    }

    return 1;
}
