# Phase 3 – Production Capability (Bootstrap)

This document tracks the initial phase-3 bootstraps and placeholders for production hardening.

## Mixed-bit runtime (layer-wise)

- Implemented a layer-wise bit selection table.
- Defaults: Attention 4-bit, MLP 3-bit, Embedding 8-bit.
- Added dynamic quant selection hooks with memory-pressure downshift.

## Backend expansion stubs

- SYCL backend stub added (parity with ROCm stub).

## Multi-GPU experimental plan

- Added a basic multi-GPU plan structure (tensor/pipeline parallel placeholders).

## Next steps

- Replace stubs with real backend bindings.
- Add model certification harness (LLaMA/Mistral/SD UNet).

## Tools

- Added a model certification harness script under tools/validation/model_certification.py.
