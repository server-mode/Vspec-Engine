#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <windows.h>

#define VSPEC_MAX_MODEL_OPTIONS 128

typedef struct VspecModelOption {
    char display_path[MAX_PATH];
    char model_file[MAX_PATH];
} VspecModelOption;

static int parent_dir_of_file(const char* file_path, char* out_dir, size_t out_cap) {
    const char* slash = NULL;
    const char* backslash = NULL;
    const char* sep = NULL;
    size_t n = 0U;
    if (!file_path || !out_dir || out_cap == 0U) {
        return 0;
    }
    slash = strrchr(file_path, '/');
    backslash = strrchr(file_path, '\\');
    sep = (slash && backslash) ? ((slash > backslash) ? slash : backslash) : (slash ? slash : backslash);
    if (!sep) {
        return 0;
    }
    n = (size_t)(sep - file_path);
    if (n == 0U || n >= out_cap) {
        return 0;
    }
    memcpy(out_dir, file_path, n);
    out_dir[n] = '\0';
    return 1;
}

static int file_exists(const char* path) {
    struct _stat st;
    if (!path || !path[0]) {
        return 0;
    }
    if (_stat(path, &st) != 0) {
        return 0;
    }
    return (st.st_mode & _S_IFREG) != 0;
}

static int dir_exists(const char* path) {
    struct _stat st;
    if (!path || !path[0]) {
        return 0;
    }
    if (_stat(path, &st) != 0) {
        return 0;
    }
    return (st.st_mode & _S_IFDIR) != 0;
}

static int has_suffix_ci(const char* s, const char* suffix) {
    size_t n = 0U;
    size_t m = 0U;
    if (!s || !suffix) {
        return 0;
    }
    n = strlen(s);
    m = strlen(suffix);
    if (n < m) {
        return 0;
    }
    return _stricmp(s + (n - m), suffix) == 0;
}

static int join_path(char* out, size_t cap, const char* a, const char* b) {
    if (!out || cap == 0U || !a || !b) {
        return 0;
    }
    if (snprintf(out, cap, "%s\\%s", a, b) >= (int)cap) {
        return 0;
    }
    return 1;
}

static int find_safetensors_recursive(const char* root, int depth, char* out_file, size_t out_cap) {
    char pattern[MAX_PATH];
    WIN32_FIND_DATAA data;
    HANDLE h = INVALID_HANDLE_VALUE;

    if (!root || !out_file || out_cap == 0U || depth < 0) {
        return 0;
    }

    if (snprintf(pattern, sizeof(pattern), "%s\\*", root) >= (int)sizeof(pattern)) {
        return 0;
    }

    h = FindFirstFileA(pattern, &data);
    if (h == INVALID_HANDLE_VALUE) {
        return 0;
    }

    do {
        if (strcmp(data.cFileName, ".") == 0 || strcmp(data.cFileName, "..") == 0) {
            continue;
        }

        {
            char candidate[MAX_PATH];
            if (!join_path(candidate, sizeof(candidate), root, data.cFileName)) {
                continue;
            }

            if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
                if (depth > 0 && find_safetensors_recursive(candidate, depth - 1, out_file, out_cap)) {
                    FindClose(h);
                    return 1;
                }
                continue;
            }

            if (has_suffix_ci(data.cFileName, ".safetensors")) {
                if (snprintf(out_file, out_cap, "%s", candidate) < (int)out_cap) {
                    FindClose(h);
                    return 1;
                }
            }
        }
    } while (FindNextFileA(h, &data));

    FindClose(h);
    return 0;
}

static int resolve_model_file(const char* input_path, char* out_file, size_t out_cap) {
    const char* env_model_file = NULL;
    const char* env_model_dir = NULL;

    if (!out_file || out_cap == 0U) {
        return 0;
    }

    out_file[0] = '\0';

    if (input_path && input_path[0]) {
        if (file_exists(input_path) && has_suffix_ci(input_path, ".safetensors")) {
            return snprintf(out_file, out_cap, "%s", input_path) < (int)out_cap;
        }
        if (dir_exists(input_path)) {
            return find_safetensors_recursive(input_path, 4, out_file, out_cap);
        }
    }

    env_model_file = getenv("VSPEC_MODEL_FILE");
    if (env_model_file && file_exists(env_model_file) && has_suffix_ci(env_model_file, ".safetensors")) {
        return snprintf(out_file, out_cap, "%s", env_model_file) < (int)out_cap;
    }

    env_model_dir = getenv("VSPEC_MODEL_DIR");
    if (env_model_dir && dir_exists(env_model_dir)) {
        if (find_safetensors_recursive(env_model_dir, 4, out_file, out_cap)) {
            return 1;
        }
    }

    if (find_safetensors_recursive("logs\\hf_models", 5, out_file, out_cap)) {
        return 1;
    }

    return 0;
}

static int collect_model_options(const char* root, VspecModelOption* options, size_t cap, size_t* out_count) {
    char pattern[MAX_PATH];
    WIN32_FIND_DATAA data;
    HANDLE h = INVALID_HANDLE_VALUE;
    size_t count = 0U;

    if (!root || !options || cap == 0U || !out_count) {
        return 0;
    }

    *out_count = 0U;
    if (snprintf(pattern, sizeof(pattern), "%s\\*", root) >= (int)sizeof(pattern)) {
        return 0;
    }

    h = FindFirstFileA(pattern, &data);
    if (h == INVALID_HANDLE_VALUE) {
        return 0;
    }

    do {
        if (strcmp(data.cFileName, ".") == 0 || strcmp(data.cFileName, "..") == 0) {
            continue;
        }
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
            continue;
        }
        if (count >= cap) {
            break;
        }

        {
            char model_dir[MAX_PATH];
            char model_file[MAX_PATH];
            if (!join_path(model_dir, sizeof(model_dir), root, data.cFileName)) {
                continue;
            }
            if (!find_safetensors_recursive(model_dir, 4, model_file, sizeof(model_file))) {
                continue;
            }

            if (snprintf(options[count].display_path, sizeof(options[count].display_path), "%s", model_dir) >= (int)sizeof(options[count].display_path)) {
                continue;
            }
            if (snprintf(options[count].model_file, sizeof(options[count].model_file), "%s", model_file) >= (int)sizeof(options[count].model_file)) {
                continue;
            }
            count += 1U;
        }
    } while (FindNextFileA(h, &data));

    FindClose(h);
    *out_count = count;
    return count > 0U;
}

static int choose_model_interactive(char* out_file, size_t out_cap, char* out_model_dir, size_t out_model_dir_cap) {
    VspecModelOption options[VSPEC_MAX_MODEL_OPTIONS];
    size_t count = 0U;
    char line[64];

    if (!out_file || out_cap == 0U || !out_model_dir || out_model_dir_cap == 0U) {
        return 0;
    }
    if (!collect_model_options("logs\\hf_models", options, VSPEC_MAX_MODEL_OPTIONS, &count)) {
        return 0;
    }

    printf("Available models:\n");
    for (size_t i = 0U; i < count; ++i) {
        printf("  %3zu. %s\n", i + 1U, options[i].display_path);
    }

    while (1) {
        long idx = 0;
        char* endp = NULL;
        printf("Select model index (0 to exit): ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) {
            return 0;
        }
        idx = strtol(line, &endp, 10);
        if (endp == line) {
            printf("Please enter a valid number.\n");
            continue;
        }
        if (idx == 0) {
            return 0;
        }
        if (idx < 0 || (size_t)idx > count) {
            printf("Index out of range.\n");
            continue;
        }
        if (snprintf(out_file, out_cap, "%s", options[(size_t)idx - 1U].model_file) >= (int)out_cap) {
            return 0;
        }
        return snprintf(out_model_dir, out_model_dir_cap, "%s", options[(size_t)idx - 1U].display_path) < (int)out_model_dir_cap;
    }
}

static intptr_t launch_real_model_chat(const char* model_dir, const char* max_tokens) {
    const char* local_python = ".venv\\Scripts\\python.exe";
    const char* script = "tools\\cli\\vspec_native_chat_bridge.py";
    const char* argv_local[7] = {0};
    const char* argv_path[7] = {0};

    if (!model_dir || !model_dir[0] || !max_tokens || !max_tokens[0]) {
        return -1;
    }

    (void)_putenv_s("VSPEC_CHAT_MODE", "native");
    (void)_putenv_s("VSPEC_FULL_NATIVE_C", "1");
    (void)_putenv_s("VSPEC_FULL_NATIVE_BYPASS_RUNTIME", "0");
    (void)_putenv_s("VSPEC_NATIVE_CHAT_REPL", "0");

    if (file_exists(local_python)) {
        argv_local[0] = local_python;
        argv_local[1] = script;
        argv_local[2] = "--model-dir";
        argv_local[3] = model_dir;
        argv_local[4] = "--max-new-tokens";
        argv_local[5] = max_tokens;
        argv_local[6] = NULL;
        return _spawnv(_P_WAIT, local_python, argv_local);
    }

    argv_path[0] = "python";
    argv_path[1] = script;
    argv_path[2] = "--model-dir";
    argv_path[3] = model_dir;
    argv_path[4] = "--max-new-tokens";
    argv_path[5] = max_tokens;
    argv_path[6] = NULL;
    return _spawnvp(_P_WAIT, "python", argv_path);
}

int main(int argc, char** argv) {
    char model_file[MAX_PATH];
    char model_dir[MAX_PATH];
    char max_steps_buf[32];
    const char* input_path = NULL;
    const char* env_model_file = NULL;
    const char* env_model_dir = NULL;
    const char* env_max_tokens = NULL;
    int max_tokens = 256;
    intptr_t rc = 0;
    model_dir[0] = '\0';

    if (argc >= 2) {
        input_path = argv[1];
    }

    env_model_file = getenv("VSPEC_MODEL_FILE");
    env_model_dir = getenv("VSPEC_MODEL_DIR");

    if ((!input_path || !input_path[0])
        && (!env_model_file || !env_model_file[0])
        && (!env_model_dir || !env_model_dir[0])) {
        if (!choose_model_interactive(model_file, sizeof(model_file), model_dir, sizeof(model_dir))) {
            fprintf(stderr, "[native-startup] no model selected.\n");
            return 1;
        }
    } else if (!resolve_model_file(input_path, model_file, sizeof(model_file))) {
        fprintf(stderr, "[native-startup] no safetensors model found.\n");
        fprintf(stderr, "usage: native_startup_chat [model-dir-or-safetensors-file]\n");
        fprintf(stderr, "hint: set VSPEC_MODEL_DIR or VSPEC_MODEL_FILE.\n");
        return 2;
    }

    if (model_dir[0] == '\0') {
        if (!parent_dir_of_file(model_file, model_dir, sizeof(model_dir))) {
            fprintf(stderr, "[native-startup] failed to resolve model directory from file.\n");
            return 3;
        }
    }

    env_max_tokens = getenv("VSPEC_MAX_TOKENS");
    if (env_max_tokens && env_max_tokens[0]) {
        int v = atoi(env_max_tokens);
        if (v > 0) {
            max_tokens = v;
        }
    }
    snprintf(max_steps_buf, sizeof(max_steps_buf), "%d", max_tokens);

    printf("Welcome to Vspec Engine\n");
    printf("[native-startup] selected model: %s\n", model_file);
    printf("[native-startup] launching native C decode backend via thin Python menu/tokenizer bridge...\n\n");

    rc = launch_real_model_chat(model_dir, max_steps_buf);
    if (rc == -1) {
        fprintf(stderr, "[native-startup] failed to launch Python chat runtime.\n");
        return 4;
    }
    return (int)rc;
}
