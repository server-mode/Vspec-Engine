#include "vspec/kernel/backend.h"

static VspecBackend g_backend = {0};

void vspec_set_backend(VspecBackend backend) {
    g_backend = backend;
}

const VspecBackend* vspec_get_backend(void) {
    return &g_backend;
}
