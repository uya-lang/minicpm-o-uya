# minicpm-o-uya

`minicpm-o-uya` is a pure Uya, CPU-first implementation plan for a MiniCPM-o
inference runtime.

The project starts with documentation and a small CLI scaffold, then grows in
validated phases: GGUF/schema inspection, Qwen3 text-only decoding, vision input,
audio input, speech output, and finally streaming omni chat.

## Status

Phase 1 is implemented: `inspect <model.gguf>` reads GGUF headers, metadata, and tensor directories without reading tensor data. No real MiniCPM-o inference is implemented yet.

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
# planned next phase
build/minicpm-o-uya audit /path/to/model.gguf
build/minicpm-o-uya encode /path/to/model.gguf "hello"
build/minicpm-o-uya generate /path/to/model.gguf "hello"
build/minicpm-o-uya vision-smoke /path/to/model.gguf /path/to/image.raw
build/minicpm-o-uya audio-smoke /path/to/model.gguf /path/to/audio.raw
build/minicpm-o-uya chat /path/to/model.gguf
```

## Documents

- `docs/design.md` — detailed pure Uya architecture and implementation plan.
- `docs/todo.md` — phased TODO checklist and acceptance criteria.


## Phase 1 Validation

```sh
make test
build/minicpm-o-uya inspect tests/fixtures/tiny.gguf
```

`tests/make_tiny_gguf.py` generates a tiny deterministic GGUF fixture and a
truncated `.part` file for short-read diagnostics.
