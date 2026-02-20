#ifndef VSPEC_COMPAT_PYTORCH_LOADER_H
#define VSPEC_COMPAT_PYTORCH_LOADER_H

#include "vspec/compat/types.h"

int vspec_pytorch_load_manifest(const char* path, VspecCompatModel* out_model);

#endif
