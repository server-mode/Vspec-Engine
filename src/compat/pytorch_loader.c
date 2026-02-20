#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vspec/compat/pytorch_loader.h"

static int parse_shape_csv(const char* s, VspecCompatTensorInfo* ti) {
    ti->ndim = 0U;
    const char* p = s;
    while (*p && ti->ndim < VSPEC_COMPAT_MAX_DIMS) {
        while (*p == ' ' || *p == '\t') p++;
        char* endp = NULL;
        long long v = strtoll(p, &endp, 10);
        if (endp == p || v <= 0) {
            break;
        }
        ti->shape[ti->ndim++] = (size_t)v;
        p = endp;
        if (*p == ',') {
            p++;
        } else {
            break;
        }
    }
    return ti->ndim > 0U;
}

int vspec_pytorch_load_manifest(const char* path, VspecCompatModel* out_model) {
    if (!path || !out_model) {
        return 0;
    }

    vspec_compat_model_init(out_model);

    FILE* f = fopen(path, "rb");
    if (!f) {
        return 0;
    }

    char line[512];
    while (fgets(line, (int)sizeof(line), f) != NULL) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }

        if (out_model->tensor_count >= VSPEC_COMPAT_MAX_TENSORS) {
            break;
        }

        char name[VSPEC_COMPAT_NAME_MAX] = {0};
        char dtype[16] = {0};
        char shape[128] = {0};

        if (sscanf(line, "%127[^|]|%15[^|]|%127[^\n\r]", name, dtype, shape) == 3) {
            VspecCompatTensorInfo* ti = &out_model->tensors[out_model->tensor_count];
            memset(ti, 0, sizeof(*ti));
            strncpy(ti->name, name, VSPEC_COMPAT_NAME_MAX - 1U);
            strncpy(ti->dtype, dtype, sizeof(ti->dtype) - 1U);
            ti->data_offset_start = 0U;
            ti->data_offset_end = 0U;

            if (parse_shape_csv(shape, ti)) {
                out_model->tensor_count += 1U;
            }
        }
    }

    fclose(f);
    return out_model->tensor_count > 0U;
}
