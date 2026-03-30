#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vspec/runtime/runtime.h"

#if defined(_WIN32)
#include <windows.h>
static void set_env_local(const char* key, const char* value) {
    (void)_putenv_s(key, value ? value : "");
}
static double now_ms(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) {
        (void)QueryPerformanceFrequency(&freq);
    }
    (void)QueryPerformanceCounter(&t);
    return ((double)t.QuadPart * 1000.0) / (double)freq.QuadPart;
}
#else
#include <time.h>
static void set_env_local(const char* key, const char* value) {
    (void)setenv(key, value ? value : "", 1);
}
static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}
#endif

int main(void) {
    VspecRuntimeBehaviorReport report;
    double rollback_ms;
    uint32_t pre_deescalate_count = 0U;
    uint32_t pre_forced_fallback_count = 0U;
    int pre_mode = (int)VSPEC_ANF_MODE_OFF;

    set_env_local("VSPEC_ENABLE_ANF", "1");
    set_env_local("VSPEC_ANF_MODE", "active");
    set_env_local("VSPEC_ANF_AUTO_DEESCALATE_ENABLE", "1");
    set_env_local("VSPEC_ANF_QUALITY_BREACH_STREAK", "3");
    set_env_local("VSPEC_ANF_DEESCALATE_TARGET", "shadow");

    vspec_runtime_init_default();

    if (!vspec_runtime_anf_available()) {
        printf("[anf_phase_d_rollback_smoke] anf_available=0 (compile-time disabled)\n");
        printf("[anf_phase_d_rollback_smoke] status=pass\n");
        return 0;
    }

    for (uint32_t i = 0U; i < 3U; ++i) {
        vspec_runtime_behavior_observe_quality(1.95f, 0.97f, 0.91f);
    }
    vspec_runtime_behavior_report(&report);
    pre_deescalate_count = report.anf_deescalate_count;
    pre_forced_fallback_count = report.anf_forced_fallback_count;
    pre_mode = report.anf_mode;

    {
        const int fallback_ok = (report.anf_deescalate_count >= 1U) && (report.anf_mode == (int)VSPEC_ANF_MODE_SHADOW);
        if (!fallback_ok) {
            printf("[anf_phase_d_rollback_smoke] status=fail (auto de-escalate not triggered)\n");
            return 1;
        }
    }

    {
        const double begin = now_ms();
        set_env_local("VSPEC_ANF_MODE", "off");
        vspec_runtime_init_default();
        rollback_ms = now_ms() - begin;
    }

    vspec_runtime_behavior_report(&report);

    {
        const int pass =
            (rollback_ms < 60000.0) &&
            (report.anf_mode == (int)VSPEC_ANF_MODE_OFF) &&
            (report.anf_silent_stop_count == 0U);

        printf("[anf_phase_d_rollback_smoke] pre_mode=%d pre_deescalate_count=%u pre_forced_fallback_count=%u\n",
            pre_mode,
            (unsigned)pre_deescalate_count,
            (unsigned)pre_forced_fallback_count);
        printf("[anf_phase_d_rollback_smoke] rollback_ms=%.3f final_mode=%d deescalate_count=%u forced_fallback_count=%u silent_stop_count=%u\n",
            rollback_ms,
            report.anf_mode,
            (unsigned)report.anf_deescalate_count,
            (unsigned)report.anf_forced_fallback_count,
            (unsigned)report.anf_silent_stop_count);
        printf("[anf_phase_d_rollback_smoke] status=%s\n", pass ? "pass" : "fail");
        return pass ? 0 : 1;
    }
}
