#ifndef VSPEC_COMPAT_SAFETENSORS_PARSER_H
#define VSPEC_COMPAT_SAFETENSORS_PARSER_H

#include <stddef.h>

#include "vspec/compat/types.h"

int vspec_safetensors_parse_header_file(const char* path, VspecCompatModel* out_model);
int vspec_safetensors_parse_header_json(const char* json, size_t len, VspecCompatModel* out_model);

#endif
