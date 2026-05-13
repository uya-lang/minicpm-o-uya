# Tests

This project uses tiny synthetic fixtures first. Full MiniCPM-o weights, images, audio, and videos must stay outside the repository and be referenced through external paths.

## Test Layers

1. Binary/GGUF metadata and truncation fixtures.
2. Tokenizer golden prompts and ChatML-like formatting.
3. Tensor shape, dtype, mmap, and bounds validation.
4. Scalar kernel golden values and quant fused dequant+dot checks.
5. Tiny Qwen3-like text binding, generation, sampler, and chat smoke.
6. Vision, audio, speech, omni, stream-chat, and benchmark fixtures with raw inputs/manifests.
7. Documented external-model smoke commands for manual compatibility work.

## Fixture Generation

Run:

```sh
python3 tests/make_tiny_gguf.py
# or
make fixtures
```

Generated files:

- `tests/fixtures/tiny.gguf` — deterministic GGUF v3 fixture with Qwen3, tokenizer, vision, audio, speech, scalar, string, and array metadata plus branchable tensors.
- `tests/fixtures/tiny_template.gguf` — tiny Qwen3 fixture with linear RoPE scaling metadata plus a MiniCPM-o/Qwen-style Jinja chat template, `<think>`, and `<|tts_bos|>` tokens for prompt-format coverage.
- `tests/fixtures/tiny_bpe.gguf` — tokenizer-only byte-BPE fixture covering merge-based English, leading-space, and UTF-8 Chinese byte-piece encoding.
- `tests/fixtures/tiny.gguf.part` — truncated file used to verify short-read errors.
- `tests/fixtures/tiny_data_truncated.gguf` — tensor-data truncation case used to verify weight-table bounds checks.
- `tests/fixtures/bad_schema.gguf` — unsupported schema used to verify missing tokenizer/root tensor, unknown dtype, and unknown branch diagnostics.
- `tests/fixtures/tiny_qwen3_missing_q.gguf` — missing text projection fixture for Qwen3 binding diagnostics.
- `tests/fixtures/tiny_audio_missing.gguf` — missing audio branch fixture for modality diagnostics.
- `tests/fixtures/tiny_image.raw` — `UYAI` raw NCHW F32 image tensor fixture.
- `tests/fixtures/tiny_rgb.raw` — `UYRG` row-major uint8 RGB fixture.
- `tests/fixtures/tiny_image.uyim` — `UYIM` image manifest with embedded `UYRG` payload.
- `tests/fixtures/tiny_video.uyvm` — `UYVM` video-frame manifest with embedded `UYRG` frames; no MP4 decoding is involved.
- `tests/fixtures/tiny_audio.raw` — `UYAM` log-mel F32 audio fixture.
- `tests/fixtures/tiny_audio.pcm` — `UYAP` PCM fixture for preprocessing smoke. `audio-input-fixture` additionally generates temporary 16 kHz mono WAV/UYAP files under `/tmp` to validate real input probing.
- `tests/fixtures/tiny_omni.json` — mixed text/image/audio/speech prompt manifest for omni smoke and benchmark.
- `tests/fixtures/tiny_stream.json` — streaming event manifest for queue/ring/interrupt smoke.

## Default Validation

`make test` builds the CLI, regenerates fixtures, and runs:

- `inspect-fixture` and `audit-fixture` for GGUF parsing, schema summaries, and unsupported diagnostics.
- `tokenizer-fixture` for `piece`, `encode`, `decode`, and `format-chat` goldens, including system/TTS assistant prompt formatting.
- `tensor-fixture` for tensor table conversion, mmap view setup, cache skeletons, unknown dtype errors, and tensor-data bounds errors.
- `kernels-fixture` for scalar reference kernel goldens.
- `quant-fixture` for selected GGML quant kernel goldens and unsupported dtype diagnostics.
- `qwen3-fixture` for tiny Qwen3 config/weight binding and missing tensor diagnostics.
- `generate-fixture` for deterministic prefill/decode, KV cache writes, logits, sampler output, and hidden-state dump summaries.
- `text-real-align` is manual and compares an external official text GGUF against llama.cpp `llama-completion`; it is not part of default CI.
- `vision-fixture` for raw image tensor smoke, RGB preprocessing, image/video manifest handling, tile counts, and checksum stability.
- `audio-real-preprocess-probe` is manual and checks real WAV/UYAP input shape, 100ms alignment, center padding, and encoder-position planning.
- `audio-real-mel-probe` is manual and computes numeric MiniCPM-o mel features from a real audio GGUF plus WAV/UYAP input; it also supports `--dump-f32 out.uyml` for full-array comparison and is not part of default CI because it needs the external official bundle.
- `audio-real-encode-probe` is manual and runs the official audio encoder `conv + transformer + projector + pool` path for a real audio GGUF plus WAV/UYAP input. It is correctness-first and currently much slower than llama.cpp-omni.
- `audio-real-mel-align` is manual and compares a Uya `--dump-f32` mel dump against a `llama.cpp-omni` `log_mel_spectrogram.json` dump from the same audio input. The current default acceptance bound is `max_abs <= 2e-3` and `mean_abs <= 2e-5`.
- `tts-condition-probe` is manual and runs the real `emb_text + projector_semantic + L2 normalize + merge` path from `llama.cpp-omni` `llm_debug/chunk_*` token-id and hidden-state dumps. `tts-merge-align` compares the resulting Uya `merged` dump against `llama.cpp-omni` `merged_embeddings.bin` with default bounds `max_abs <= 1e-5` and `mean_abs <= 1e-6`.
- `tts-simplex-probe` is manual and runs the official 20-layer TTS decoder over a whole `llm_debug/chunk_*` directory, keeps KV cache across chunks, and writes `audio_tokens_chunk_*.txt/bin` in the same relative-token format as `llama.cpp-omni`.
- `tts-token-align` is manual and wraps the simplex probe with per-chunk `count`, `prefix_match`, and `exact` summaries against a reference `tts_wav` directory. It supports non-strict exploratory comparison and strict exact mode for future seeded baselines.
- `token2wav-prompt-cache-probe` is manual and validates prompt-cache metadata plus cache tensor checksums from the official `prompt_cache.gguf`.
- `token2wav-window-probe` is manual and prints the `chunk_total=28`, `chunk_main=25`, `pre_lookahead=3` feed schedule for a real `audio_tokens_chunk.txt`.
- `token2wav-flow-probe` is manual and runs a lightweight reference `flow_matching` forward on official `flow_matching.gguf` and `flow_extra.gguf`, using prompt-cache speaker embedding plus a truncated token window. It is intentionally a probe, not yet the full conformer-token2mel path.
- `audio2audio-real-prefix-audit`, `audio2audio-real-prefix-text`, and `audio2audio-real-prefix-wav` are manual multi-turn `--test-prefix PREFIX --count N` entry points. `audio2audio-real-report` wraps `tests/audio2audio_real_manifest.py` to summarize per-turn `timing.log` artifacts into one JSON report for later CPU-only baseline comparison.
- `audio-fixture` for log-mel encoder smoke and PCM preprocessing.
- `speech-fixture` for tiny speech decoder/vocoder smoke and deterministic WAV output checks.
- `audio2audio-fixture` for PCM/mel audio input through audio encoder, speech decoder, vocoder, and WAV output.
- `omni-fixture` for mixed manifest parsing, placeholder spans, media embedding counts, and prompt compilation.
- `omni-chat-fixture` for blocking text and manifest turns, context preservation, and explicit unsupported speech output diagnostics.
- `stream-chat-fixture` for stream queue, audio ring buffer, partial callbacks, backpressure, interrupt, and pending speech cleanup diagnostics.
- `bench-fixture` for deterministic smoke benchmark output plus a tiny real text benchmark invocation.
- `chat-fixture` for text REPL behavior and context overflow diagnostics.

## Golden Coverage

Tokenizer golden coverage includes English, Chinese punctuation, newline-capable vocabulary, byte-BPE merges for leading-space and UTF-8 byte pieces, and MiniCPM-o media placeholders such as `<image>` and `<audio>` so they are matched as whole tokens.

Kernel smoke coverage includes F32 vector ops, F16/BF16 loads, normalization, RoPE, softmax, dense matvec, activations, Conv1D/Conv2D, and NaN/Inf/empty-length boundaries.

Quant smoke coverage includes `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL` fused dequant+dot checks, row-stride `Q8_0` matvec, and unsupported dtype diagnostics with tensor name plus dtype.

Qwen3 binding coverage uses a tiny one-layer Qwen3-like GGUF with token embedding, output norm/head, attention projections, optional q/k norms, and FFN projections; it also checks missing tensor diagnostics include layer and tensor name.

Generation smoke coverage uses the deterministic tiny Qwen3-like fixture to validate token embedding lookup, prefill/decode, KV cache writes, causal attention, final logits, and greedy/sampler next-token selection for fixed prompts.

Vision smoke coverage uses tiny raw image fixtures to validate patch embedding, one-block vision transformer behavior, resampler/projector binding, image embedding span injection, stable checksum `0xb5a01b45`, placeholder/span alignment, RGB resize/crop-pad/normalize checksum `0xea7fa412`, image manifest parity, video manifest frame/tile counts, and 2D position embedding interpolation.

Audio smoke coverage binds the tiny conv/front-end, one transformer block, output norm, projector, and `<audio>` embedding injection path. Current checks include raw checksum `0xbca0dcc`, embedding checksum `0x625ac595`, non-zero logits diff, PCM preprocessing, real 16 kHz mono WAV/UYAP input probing with explicit rejection of unsupported sample rates/channels, and clear missing-branch errors. Real MiniCPM-o audio GGUF binding is covered by the manual `MINICPM_O_REAL_BUNDLE=/path make audio-real-bind` target. Real TTS GGUF binding is covered by `MINICPM_O_REAL_BUNDLE=/path make tts-real-bind`. Real token2wav / HiFiGAN bind-only coverage is covered by `MINICPM_O_REAL_BUNDLE=/path make token2wav-real-bind`.

Real TTS conditioning coverage is manual because it depends on external `llama.cpp-omni` `llm_debug` artifacts. The current local `chunk_{0..3}` projector/merge alignment results are all in the `1e-8` mean-absolute range with `max_abs=2.384e-7`.
Real TTS decoder coverage is also manual. The current Uya simplex probe can generate non-empty audio token chunks from the same `llm_debug` inputs, but exact token replay against the saved local `round_000/tts_wav` files is not yet treated as deterministic because that baseline was generated without a pinned sampler seed.

Audio-to-audio smoke coverage runs `audio2audio-smoke` from `UYAP` PCM input through mel preprocessing, audio encoder, Qwen prompt prefill, speech token generation, vocoder decode, and WAV writing. It validates the audio-conditioned logits differ from text-only logits, writes a RIFF/WAVE file, checks sample rate/data bytes, and exercises both single-GGUF and split `--audio-model`/`--speech-model`/`--vocoder-model` argument forms.

Speech smoke coverage validates prompt-to-token setup, deterministic decoder features, vocoder sample generation, optional WAV writing, and unsupported/missing speech tensor diagnostics without claiming natural audio quality.

Omni and stream coverage validates fixed JSON manifests, media placeholder spans, cache/accounting, stream queue high-watermark, audio ring usage, partial outputs, interrupt cleanup, and explicit unsupported speech behavior in blocking `omni-chat`.

Benchmark coverage validates the tiny fixture output:

```text
bench load: ms=3 mode=tiny-mmap-metadata
bench text_prompt: 500.000 tokens/s units=1 ms=2
bench text_decode: 333.333 tokens/s units=1 ms=3
bench vision_encode: ms=7 frames=1 tiles=1
bench audio_encode: ms=5 chunks=1
bench vocoder: 16000.000 samples/s units=64 ms=4
bench memory: peak_estimate_bytes=78336 llm=512 vision=4096 audio=4096 speech=4096 scratch=65536
bench reference_error: text=0.000000 vision=0.000000 audio=0.000000 vocoder=0.000000
```

It also checks the real text benchmark mode on the tiny fixture with a small workload:

```text
bench config: mode=text-real prompt_tokens=4 gen_tokens=4 repetitions=1 warmup=0 threads=1
bench load: ms=...
bench text_prompt: ... tokens/s ...
bench text_decode: ... tokens/s ...
bench reference_error: n/a mode=real-timing
```

## External Smoke

Default tests never require external weights. For real/community models, set paths outside the repository and run commands manually:

```sh
export MINICPM_O_GGUF=/path/to/minicpm-o.gguf
export MINICPM_O_TEXT_GGUF=/path/to/qwen3-text.gguf
export MINICPM_O_AUDIO_GGUF=/path/to/MiniCPM-o-4_5-audio-F16.gguf
export MINICPM_O_SPEECH_GGUF=/path/to/MiniCPM-o-4_5-tts-F16.gguf
export MINICPM_O_VOCODER_GGUF=/path/to/token2wav-or-vocoder.gguf
export MINICPM_O_IMAGE_RAW=/path/to/image.raw
export MINICPM_O_AUDIO_RAW=/path/to/audio.raw
export MINICPM_O_MANIFEST=/path/to/omni-manifest.json

MINICPM_O_GGUF="$MINICPM_O_GGUF" make minicpmo-audit
build/minicpm-o-uya inspect "$MINICPM_O_GGUF"
build/minicpm-o-uya audit "$MINICPM_O_GGUF"
build/minicpm-o-uya tensors "$MINICPM_O_GGUF" --mmap
build/minicpm-o-uya qwen3-bind "$MINICPM_O_TEXT_GGUF"
build/minicpm-o-uya vision-smoke "$MINICPM_O_GGUF" "$MINICPM_O_IMAGE_RAW"
build/minicpm-o-uya audio-smoke "$MINICPM_O_GGUF" "$MINICPM_O_AUDIO_RAW"
build/minicpm-o-uya audio-real-preprocess-probe /path/to/user.wav
build/minicpm-o-uya audio-real-mel-probe "$MINICPM_O_AUDIO_GGUF" /path/to/user.wav
build/minicpm-o-uya audio-real-encode-probe "$MINICPM_O_AUDIO_GGUF" /path/to/user.wav
build/minicpm-o-uya token2wav-bind /path/to/token2wav-gguf
build/minicpm-o-uya token2wav-prompt-cache-probe /path/to/token2wav-gguf/prompt_cache.gguf
build/minicpm-o-uya token2wav-window-probe /path/to/token2wav-gguf/prompt_cache.gguf /path/to/audio_tokens_chunk_0.txt
build/minicpm-o-uya tts-condition-probe /path/to/MiniCPM-o-4_5-tts-F16.gguf /path/to/MiniCPM-o-4_5-projector-F16.gguf /path/to/llm_token_ids.txt /path/to/llm_hidden_states.bin --dump-merged /tmp/minicpm-o-uya-merged.bin
build/minicpm-o-uya audio2audio-smoke "$MINICPM_O_TEXT_GGUF" --audio-model "$MINICPM_O_AUDIO_GGUF" --speech-model "$MINICPM_O_SPEECH_GGUF" --vocoder-model "$MINICPM_O_VOCODER_GGUF" "$MINICPM_O_AUDIO_RAW" /tmp/minicpm-o-uya-audio2audio.wav
build/minicpm-o-uya omni-smoke "$MINICPM_O_GGUF" "$MINICPM_O_MANIFEST"
build/minicpm-o-uya bench "$MINICPM_O_GGUF" "$MINICPM_O_MANIFEST"
build/minicpm-o-uya bench "$MINICPM_O_TEXT_GGUF" --n-prompt 512 --n-gen 128 --repetitions 5 --threads 28
```

If external smoke fails, the expected next step is to preserve the exact diagnostic, run `audit`, and add the missing dtype/layout/modality binding. Do not treat tiny fixture pass results as production MiniCPM-o compatibility.
