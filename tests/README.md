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

## Phase 1 Fixtures

Run `python3 tests/make_tiny_gguf.py` to generate:

- `tests/fixtures/tiny.gguf` — deterministic GGUF v3 fixture with scalar, string, and array metadata plus two tensors.
- `tests/fixtures/tiny.gguf.part` — truncated file used to verify short-read errors.

`make test` builds the CLI, inspects `tiny.gguf`, and confirms the truncated
fixture fails cleanly without reading tensor data.
