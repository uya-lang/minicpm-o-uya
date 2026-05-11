# Tests

This project should use tiny synthetic fixtures first. Full MiniCPM-o weights,
images, audio, and videos must stay outside the repository.

Planned test layers:

1. Binary/GGUF metadata fixtures.
2. Tokenizer golden prompts.
3. Tensor shape and dtype validation.
4. Scalar kernel golden values.
5. Qwen3 text-only small model fixtures.
6. Vision/audio/speech fixtures with raw tensor inputs.
7. Documented external-model smoke targets.

## GGUF Fixtures

Run `python3 tests/make_tiny_gguf.py` to generate:

- `tests/fixtures/tiny.gguf` — deterministic GGUF v3 fixture with Qwen3, tokenizer, vision, audio, speech, scalar, string, and array metadata plus branchable tensors.
- `tests/fixtures/tiny.gguf.part` — truncated file used to verify short-read errors.
- `tests/fixtures/tiny_image.raw` — deterministic raw vision tensor fixture: little-endian `UYAI` magic (`0x49415955`), version `1`, `width`, `height`, `channels`, dtype `0` for NCHW F32, followed by `width * height * channels` float32 values.
- `tests/fixtures/tiny_rgb.raw` — minimal raw RGB fixture: little-endian `UYRG` magic (`0x47525955`), version `1`, `width`, `height`, channels `3`, dtype `0` for row-major uint8 RGB bytes.
- `tests/fixtures/tiny_image.uyim` — image manifest fixture: `UYIM`, version `1`, image count, target shape/tile shape, followed by an embedded `UYRG` payload.
- `tests/fixtures/tiny_video.uyvm` — video frame sequence manifest: `UYVM`, version `1`, frame count, target shape/tile shape, followed by embedded `UYRG` frame payloads; no MP4 decoding is involved.
- `tests/fixtures/bad_schema.gguf` — generated unsupported schema used to verify missing tokenizer/root tensor, unknown dtype, and unknown branch diagnostics.
- `tests/fixtures/tiny_data_truncated.gguf` — generated tensor-data truncation case used to verify weight-table bounds checks.

`make test` builds the CLI, runs `inspect`, `audit`, tokenizer piece/encode/decode, `format-chat`, tensor table/mmap inspection, Qwen3 binding, tiny text-only generation, vision raw tensor smoke, vision preprocessing smoke, sampler/chat CLI, and scalar and quant kernel golden smoke tests; it also confirms truncated fixtures fail cleanly and checks unsupported-schema diagnostics without reading tensor data.

Tokenizer golden coverage includes English, Chinese punctuation, newline-capable vocabulary, and MiniCPM-o media placeholders such as `<image>` and `<audio>` so they are matched as whole tokens.

Kernel smoke coverage includes F32 vector ops, F16/BF16 loads, normalization, RoPE, softmax, dense matvec, activations, Conv1D/Conv2D, and NaN/Inf/empty-length boundaries.

Quant smoke coverage includes Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL fused dequant+dot checks, row-stride Q8_0 matvec, and unsupported dtype diagnostics with tensor name plus dtype.

Qwen3 binding coverage uses a tiny one-layer Qwen3-like GGUF with token embedding, output norm/head, attention projections, optional q/k norms, and FFN projections; it also checks missing tensor diagnostics include layer and tensor name.

Generation smoke coverage uses the deterministic tiny Qwen3-like fixture to validate token embedding lookup, prefill/decode, KV cache writes, causal attention, final logits, and greedy next-token selection for fixed prompts.

Vision smoke coverage uses the tiny raw image fixture to validate patch embedding, a one-block vision transformer, resampler/projector binding, image embedding span injection, stable checksum `0xb5a01b45`, placeholder/span alignment, and observable image-vs-text logits difference. Vision preprocessing coverage validates `UYRG` resize/crop-pad/normalize output checksum `0xea7fa412`, image manifest parity, video manifest frame/tile/placeholder counts, and 2D position embedding interpolation without PNG/JPEG/MP4 decoders.

Sampler/chat coverage validates deterministic seed behavior, sampler-arg validation, stop-token handling, and a minimal text-only `chat` REPL loop that preserves multi-turn context until the tiny fixture hits its context limit.
