#include "vspec/compat/types.h"

void vspec_compat_model_init(VspecCompatModel* model) {
    if (!model) {
        return;
    }
    model->tensor_count = 0U;
}
