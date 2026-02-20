#ifndef VSPEC_PYTHON_CAPI_H
#define VSPEC_PYTHON_CAPI_H

#include <stddef.h>

#if defined(_WIN32)
  #if defined(VSPEC_PYTHON_CAPI_BUILD)
    #define VSPEC_PY_API __declspec(dllexport)
  #else
    #define VSPEC_PY_API __declspec(dllimport)
  #endif
#else
  #define VSPEC_PY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

VSPEC_PY_API const char* vspec_py_version(void);
VSPEC_PY_API int vspec_py_load_manifest_count(const char* path);
VSPEC_PY_API int vspec_py_parse_safetensors_count(const char* path);
VSPEC_PY_API int vspec_py_rewrite_demo_final_nodes(void);
VSPEC_PY_API int vspec_py_generate(const char* prompt, char* out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif
