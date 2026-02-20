#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "vspec/compat/safetensors_parser.h"

static uint64_t read_u64_le(const uint8_t b[8]) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
        v |= ((uint64_t)b[i]) << (8U * i);
    }
    return v;
}

static const char* skip_ws(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    return p;
}

static const char* parse_json_string(const char* p, const char* end, char* out, size_t out_sz) {
    if (p >= end || *p != '"') return NULL;
    p++;
    size_t n = 0;
    while (p < end && *p != '"') {
        if (n + 1U < out_sz) {
            out[n++] = *p;
        }
        p++;
    }
    if (p >= end || *p != '"') return NULL;
    out[n] = '\0';
    return p + 1;
}

int vspec_safetensors_parse_header_json(const char* json, size_t len, VspecCompatModel* out_model) {
    if (!json || !out_model || len == 0U) {
        return 0;
    }

    vspec_compat_model_init(out_model);

    const char* p = json;
    const char* end = json + len;

    p = skip_ws(p, end);
    if (p >= end || *p != '{') {
        return 0;
    }
    p++;

    while (p < end) {
        p = skip_ws(p, end);
        if (p < end && *p == '}') {
            return out_model->tensor_count > 0U;
        }

        if (out_model->tensor_count >= VSPEC_COMPAT_MAX_TENSORS) {
            return out_model->tensor_count > 0U;
        }

        char key[VSPEC_COMPAT_NAME_MAX] = {0};
        p = parse_json_string(p, end, key, sizeof(key));
        if (!p) return 0;

        p = skip_ws(p, end);
        if (p >= end || *p != ':') return 0;
        p++;
        p = skip_ws(p, end);

        if (p < end && *p == '{') {
            const char* obj_start = p;
            int brace = 1;
            p++;
            while (p < end && brace > 0) {
                if (*p == '{') brace++;
                else if (*p == '}') brace--;
                p++;
            }
            if (brace != 0) return 0;

            if (strcmp(key, "__metadata__") != 0) {
                const char* obj_end = p;
                VspecCompatTensorInfo* ti = &out_model->tensors[out_model->tensor_count];
                memset(ti, 0, sizeof(*ti));
                strncpy(ti->name, key, VSPEC_COMPAT_NAME_MAX - 1U);

                const char* d = strstr(obj_start, "\"dtype\"");
                if (d && d < obj_end) {
                    d = strchr(d, ':');
                    if (d && d < obj_end) {
                        d = skip_ws(d + 1, obj_end);
                        parse_json_string(d, obj_end, ti->dtype, sizeof(ti->dtype));
                    }
                }

                const char* s = strstr(obj_start, "\"shape\"");
                if (s && s < obj_end) {
                    s = strchr(s, '[');
                    if (s && s < obj_end) {
                        s++;
                        while (s < obj_end && ti->ndim < VSPEC_COMPAT_MAX_DIMS) {
                            s = skip_ws(s, obj_end);
                            char* ep = NULL;
                            long long dim = strtoll(s, &ep, 10);
                            if (ep == s || dim <= 0) break;
                            ti->shape[ti->ndim++] = (size_t)dim;
                            s = ep;
                            s = skip_ws(s, obj_end);
                            if (*s == ',') s++;
                            else break;
                        }
                    }
                }

                const char* off = strstr(obj_start, "\"data_offsets\"");
                if (off && off < obj_end) {
                    off = strchr(off, '[');
                    if (off && off < obj_end) {
                        off++;
                        char* ep = NULL;
                        unsigned long long a = strtoull(off, &ep, 10);
                        if (ep != off) {
                            off = ep;
                            off = strchr(off, ',');
                            if (off && off < obj_end) {
                                off++;
                                unsigned long long b = strtoull(off, &ep, 10);
                                if (ep != off) {
                                    ti->data_offset_start = (uint64_t)a;
                                    ti->data_offset_end = (uint64_t)b;
                                }
                            }
                        }
                    }
                }

                out_model->tensor_count += 1U;
            }
        } else {
            while (p < end && *p != ',' && *p != '}') p++;
        }

        p = skip_ws(p, end);
        if (p < end && *p == ',') {
            p++;
            continue;
        }
        if (p < end && *p == '}') {
            return out_model->tensor_count > 0U;
        }
    }

    return out_model->tensor_count > 0U;
}

int vspec_safetensors_parse_header_file(const char* path, VspecCompatModel* out_model) {
    if (!path || !out_model) {
        return 0;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return 0;
    }

    uint8_t len_buf[8];
    if (fread(len_buf, 1, 8, f) != 8) {
        fclose(f);
        return 0;
    }

    uint64_t header_len = read_u64_le(len_buf);
    if (header_len == 0U || header_len > (64U * 1024U * 1024U)) {
        fclose(f);
        return 0;
    }

    char* header = (char*)malloc((size_t)header_len + 1U);
    if (!header) {
        fclose(f);
        return 0;
    }

    if (fread(header, 1, (size_t)header_len, f) != (size_t)header_len) {
        free(header);
        fclose(f);
        return 0;
    }
    header[header_len] = '\0';

    const int ok = vspec_safetensors_parse_header_json(header, (size_t)header_len, out_model);

    free(header);
    fclose(f);
    return ok;
}
