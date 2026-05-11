# minicpm-o-uya

`minicpm-o-uya` is a pure Uya, CPU-first implementation plan for a MiniCPM-o
inference runtime.

The project starts with documentation and a small CLI scaffold, then grows in
validated phases: GGUF/schema inspection, Qwen3 text-only decoding, vision input,
audio input, speech output, and finally streaming omni chat.

## Status

Phase 8 is implemented: `inspect`, `audit`, tokenizer CLI, tensor weight-table/mmap inspection, Qwen3 config/weight binding, tiny Qwen3 text-only generation, scalar reference kernel smoke tests, and first-pass quant kernel smoke tests are available.

## Goals

- Pure Uya source for model/runtime logic.
- Linux x86_64 CPU-first bring-up.
- Small, testable modules rather than a direct Python project translation.
- Honest runtime status: audit/inspection is not generation.
- External model paths only; do not commit large weights.

## CLI

```sh
build/minicpm-o-uya --help
build/minicpm-o-uya inspect /path/to/model.gguf
build/minicpm-o-uya audit /path/to/model.gguf
build/minicpm-o-uya piece /path/to/model.gguf 42
build/minicpm-o-uya encode /path/to/model.gguf "hello"
build/minicpm-o-uya decode /path/to/model.gguf 1 2 3
build/minicpm-o-uya format-chat /path/to/model.gguf "<image> hello"
build/minicpm-o-uya tensors /path/to/model.gguf --mmap
build/minicpm-o-uya qwen3-bind /path/to/model.gguf
build/minicpm-o-uya kernels-smoke
build/minicpm-o-uya quant-smoke
MINICPM_O_GGUF=/path/to/model.gguf make minicpmo-audit
build/minicpm-o-uya generate /path/to/model.gguf "hello"
build/minicpm-o-uya vision-smoke /path/to/model.gguf /path/to/image.raw
build/minicpm-o-uya audio-smoke /path/to/model.gguf /path/to/audio.raw
build/minicpm-o-uya chat /path/to/model.gguf
```

## Documents

- `docs/design.md` — detailed pure Uya architecture and implementation plan.
- `docs/todo.md` — phased TODO checklist and acceptance criteria.


## Validation

```sh
make test
build/minicpm-o-uya inspect tests/fixtures/tiny.gguf
build/minicpm-o-uya audit tests/fixtures/tiny.gguf
build/minicpm-o-uya encode tests/fixtures/tiny.gguf "<image> hello <audio>"
build/minicpm-o-uya format-chat tests/fixtures/tiny.gguf "<image> hello"
build/minicpm-o-uya tensors tests/fixtures/tiny.gguf --mmap
build/minicpm-o-uya qwen3-bind tests/fixtures/tiny.gguf
build/minicpm-o-uya kernels-smoke
build/minicpm-o-uya quant-smoke
```

`tests/make_tiny_gguf.py` generates deterministic GGUF fixtures for inspect/audit, including a truncated `.part` file and an intentionally unsupported schema for diagnostics.
