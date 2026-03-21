#include "vspec/runtime/plugin/plugin_api.h"

#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define VSPEC_PLUGIN_MAX 16

typedef struct VspecPluginSlot {
    int used;
    int dynamic_loaded;
    char name[64];
    VspecRuntimePluginHooks hooks;
#if defined(_WIN32)
    HMODULE module;
#else
    void* module;
#endif
} VspecPluginSlot;

static VspecPluginSlot g_plugins[VSPEC_PLUGIN_MAX];

typedef int (*VspecPluginRegisterEntryFn)(VspecRuntimePluginHooks* out_hooks, char* out_name, size_t out_name_len);

static void write_err(char* err_buf, size_t err_buf_len, const char* msg) {
    if (err_buf && err_buf_len > 0U) {
        (void)snprintf(err_buf, err_buf_len, "%s", msg ? msg : "plugin_error");
    }
}

static int find_slot_by_name(const char* name) {
    if (!name || name[0] == '\0') {
        return -1;
    }
    for (int i = 0; i < VSPEC_PLUGIN_MAX; ++i) {
        if (g_plugins[i].used && strcmp(g_plugins[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int vspec_plugin_register(const char* name, const VspecRuntimePluginHooks* hooks) {
    if (!name || !hooks || name[0] == '\0') {
        return 0;
    }

    int idx = find_slot_by_name(name);
    if (idx >= 0) {
        g_plugins[idx].hooks = *hooks;
        return 1;
    }

    for (int i = 0; i < VSPEC_PLUGIN_MAX; ++i) {
        if (!g_plugins[i].used) {
            g_plugins[i].used = 1;
            (void)memset(g_plugins[i].name, 0, sizeof(g_plugins[i].name));
            (void)strncpy(g_plugins[i].name, name, sizeof(g_plugins[i].name) - 1U);
            g_plugins[i].hooks = *hooks;
            return 1;
        }
    }
    return 0;
}

int vspec_plugin_unregister(const char* name) {
    int idx = find_slot_by_name(name);
    if (idx < 0) {
        return 0;
    }
#if defined(_WIN32)
    if (g_plugins[idx].dynamic_loaded && g_plugins[idx].module) {
        (void)FreeLibrary(g_plugins[idx].module);
    }
#else
    if (g_plugins[idx].dynamic_loaded && g_plugins[idx].module) {
        (void)dlclose(g_plugins[idx].module);
    }
#endif
    (void)memset(&g_plugins[idx], 0, sizeof(g_plugins[idx]));
    return 1;
}

size_t vspec_plugin_count(void) {
    size_t count = 0U;
    for (int i = 0; i < VSPEC_PLUGIN_MAX; ++i) {
        if (g_plugins[i].used) {
            count += 1U;
        }
    }
    return count;
}

void vspec_plugin_emit_controller_decision(const VspecRuntimeAdaptiveTelemetry* telemetry, const VspecRuntimeAdaptiveDecision* decision) {
    for (int i = 0; i < VSPEC_PLUGIN_MAX; ++i) {
        if (g_plugins[i].used && g_plugins[i].hooks.on_controller_decision) {
            g_plugins[i].hooks.on_controller_decision(telemetry, decision);
        }
    }
}

void vspec_plugin_emit_token_scheduled(const char* token_text, const VspecTokenScheduleDecision* decision) {
    for (int i = 0; i < VSPEC_PLUGIN_MAX; ++i) {
        if (g_plugins[i].used && g_plugins[i].hooks.on_token_scheduled) {
            g_plugins[i].hooks.on_token_scheduled(token_text, decision);
        }
    }
}

int vspec_plugin_load_dynamic(const char* path, const char* symbol_name, char* err_buf, size_t err_buf_len) {
    if (!path || path[0] == '\0') {
        write_err(err_buf, err_buf_len, "plugin_path_empty");
        return 0;
    }

    const char* sym = (symbol_name && symbol_name[0] != '\0') ? symbol_name : "vspec_plugin_register_entry";
#if defined(_WIN32)
    HMODULE mod = LoadLibraryA(path);
    if (!mod) {
        write_err(err_buf, err_buf_len, "plugin_load_failed");
        return 0;
    }
    FARPROC raw = GetProcAddress(mod, sym);
    if (!raw) {
        (void)FreeLibrary(mod);
        write_err(err_buf, err_buf_len, "plugin_symbol_not_found");
        return 0;
    }
    VspecPluginRegisterEntryFn entry = (VspecPluginRegisterEntryFn)raw;
#else
    void* mod = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!mod) {
        write_err(err_buf, err_buf_len, dlerror());
        return 0;
    }
    void* raw = dlsym(mod, sym);
    if (!raw) {
        (void)dlclose(mod);
        write_err(err_buf, err_buf_len, "plugin_symbol_not_found");
        return 0;
    }
    VspecPluginRegisterEntryFn entry = (VspecPluginRegisterEntryFn)raw;
#endif

    VspecRuntimePluginHooks hooks;
    char plugin_name[64];
    (void)memset(&hooks, 0, sizeof(hooks));
    (void)memset(plugin_name, 0, sizeof(plugin_name));

    if (!entry(&hooks, plugin_name, sizeof(plugin_name)) || plugin_name[0] == '\0') {
#if defined(_WIN32)
        (void)FreeLibrary(mod);
#else
        (void)dlclose(mod);
#endif
        write_err(err_buf, err_buf_len, "plugin_entry_failed");
        return 0;
    }

    if (!vspec_plugin_register(plugin_name, &hooks)) {
#if defined(_WIN32)
        (void)FreeLibrary(mod);
#else
        (void)dlclose(mod);
#endif
        write_err(err_buf, err_buf_len, "plugin_register_failed");
        return 0;
    }

    {
        int idx = find_slot_by_name(plugin_name);
        if (idx >= 0) {
            g_plugins[idx].dynamic_loaded = 1;
            g_plugins[idx].module = mod;
        } else {
#if defined(_WIN32)
            (void)FreeLibrary(mod);
#else
            (void)dlclose(mod);
#endif
            write_err(err_buf, err_buf_len, "plugin_slot_lost");
            return 0;
        }
    }

    write_err(err_buf, err_buf_len, "ok");
    return 1;
}

int vspec_plugin_unload_dynamic(const char* name, char* err_buf, size_t err_buf_len) {
    if (!name || name[0] == '\0') {
        write_err(err_buf, err_buf_len, "plugin_name_empty");
        return 0;
    }

    int idx = find_slot_by_name(name);
    if (idx < 0) {
        write_err(err_buf, err_buf_len, "plugin_not_found");
        return 0;
    }
    if (!g_plugins[idx].dynamic_loaded) {
        write_err(err_buf, err_buf_len, "plugin_not_dynamic");
        return 0;
    }
    if (!vspec_plugin_unregister(name)) {
        write_err(err_buf, err_buf_len, "plugin_unload_failed");
        return 0;
    }

    write_err(err_buf, err_buf_len, "ok");
    return 1;
}
