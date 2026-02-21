# Vspec-chat

Vspec-chat is a lightweight chat CLI workspace that sits next to the core Vspec runtime.
It is intentionally separated so the runtime stays minimal while the chat stack evolves.

## Layout

- python/: Python CLI (loader + tokenizer + sampling loop)
- c/: C CLI skeleton (runtime-friendly entry point)

## Notes

- The Python CLI is a prototype to validate model loading, tokenization, and sampling flow.
- The C CLI is a placeholder for a future fully native chat pipeline.
- For the Python prototype, install: tokenizers, numpy, safetensors, torch.
- Quick install: pip install -r Vspec-chat/python/requirements.txt
- The loader and sampler are generic and intended to work across models using shared weight names.
- For CPU tests, use --max-layers to limit runtime cost (default 8).
- Use --greedy to reduce randomness in output.
- Use --device cuda to try torch CUDA runtime when available.
- Use --device cuda-native to enable native CUDA RMSNorm via vspec_cuda_bridge.
- Decoding anti-loop controls: --repetition-penalty, --repeat-window, --no-repeat-ngram.
- Language-constrained decode: --lang auto|vi|en and --lang-top-n.
- Generic chat templates: --chat-format auto|plain|chatml|llama3|alpaca.
