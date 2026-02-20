# Vspec Backend ABI Contract (Draft)

## ABI Version

- `VSPEC_RUNTIME_ABI_VERSION = 1`
- All backends must report compatibility with this ABI version.

## Required Symbols

Each backend must provide the following symbols (C ABI):

- `VspecBackend vspec_make_<backend>_backend(void);`
- `int vspec_<backend>_backend_available(void);`

Where `<backend>` is one of: `cuda`, `rocm`, `cpu`.

## Kernel Entry Points

Backends must implement:

- `void (*launch_linear)(VspecKernelContext* ctx)`
- `void (*launch_attention)(VspecKernelContext* ctx)`

The kernel must:

- Validate pointers and configuration.
- Respect quant metadata schema version.
- Avoid modifying input/weight buffers.

## Quant Metadata Schema

Backends must handle:

- `schema_version` in `VspecQuantMeta`.
- Unknown schema versions should return early or report unsupported.

## Thread Safety

- `vspec_set_backend()` is not thread-safe.
- Call it during initialization only.

## Compatibility Policy

- ABI breaking changes must bump `VSPEC_RUNTIME_ABI_VERSION`.
- IR or quant schema changes must bump their respective versions.
