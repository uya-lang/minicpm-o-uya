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
- `tests/fixtures/bad_schema.gguf` — generated unsupported schema used to verify missing tokenizer/root tensor, unknown dtype, and unknown branch diagnostics.

`make test` builds the CLI, runs `inspect` and `audit` on `tiny.gguf`, confirms the truncated fixture fails cleanly, and checks unsupported-schema diagnostics without reading tensor data.
