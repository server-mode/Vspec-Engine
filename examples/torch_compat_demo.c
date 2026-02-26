#include <stdio.h>
#include <string.h>

#include "vspec/runtime/torch_compat_module.h"

static int approx_equal(float a, float b, float tol) {
    float d = a - b;
    if (d < 0.0f) {
        d = -d;
    }
    return d <= tol;
}

int main(void) {
    VspecTorchCompatRuntime runtime;
    VspecTorchCompatCapabilities caps;
    vspec_torch_compat_runtime_init(&runtime, VSPEC_TORCH_COMPAT_DEVICE_AUTO);
    vspec_torch_compat_query_capabilities(&runtime, &caps);

    printf("[torch-compat] backend=%s\n", runtime.active_backend ? runtime.active_backend : "unknown");
    printf("[torch-compat] caps matmul=%d layernorm=%d gelu=%d\n", caps.matmul_f32, caps.layernorm_f32, caps.gelu_f32);

    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {1, 0, 0, 1, 1, 1};
    float out[4] = {0};
    vspec_torch_compat_matmul_f32(a, b, 2, 3, 2, out);

    int ok = 1;
    ok &= approx_equal(out[0], 4.0f, 1e-4f);
    ok &= approx_equal(out[1], 5.0f, 1e-4f);
    ok &= approx_equal(out[2], 10.0f, 1e-4f);
    ok &= approx_equal(out[3], 11.0f, 1e-4f);

    float x[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float y[4] = {0};
    vspec_torch_compat_softmax_f32(x, 4, y);
    size_t arg = vspec_torch_compat_argmax_f32(y, 4);
    ok &= (arg == 3);

    printf("[torch-compat] parity_smoke=%s\n", ok ? "pass" : "fail");
    return ok ? 0 : 1;
}
