# MiniCPM-o 4.5 Audio-to-Audio Real Path

This page records the real, split-GGUF audio-to-audio path. It is intentionally separate from `audio2audio-smoke`, which only validates the tiny fixture pipeline.

## Baseline: llama.cpp-omni

`llama.cpp-omni` uses `--test <prefix> <count>` with this convention:

- `<prefix>_0000.wav` is the reference/system voice audio.
- `<prefix>_0001.wav` and later files are user audio turns.
- Passing only `_0000.wav` as count `1` is not a real user question turn and can look like input repetition.

Example baseline command:

```sh
cd llama.cpp-omni
OMNI_VOC_DEVICE=cpu ./build/bin/llama-omni-cli \
  -m "$PWD/../models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf" \
  --ref-audio "$PWD/../outputs/complex_case2/complex2_0000.wav" \
  -ngl 0 \
  -c 4096 \
  --test "$PWD/../outputs/complex_case2/complex2_" 2
```

## Uya Bundle Audit

The first real Uya CLI is an audit-only loader for the official split bundle. It verifies that every required GGUF file can be opened and inventoried before real forward kernels are connected.

```sh
build/minicpm-o-uya audio2audio-real --audit-only \
  --llm models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --audio models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
  --tts models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  --projector models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  --token2wav-dir models/MiniCPM-o-4_5-gguf/token2wav-gguf
```

Expected success marker:

```text
audio2audio-real: audit-only PASS files=9
audio2audio-real: inference_supported=false next=bind-official-tensors
```

## Required GGUF Files

The official bundle must include:

- `MiniCPM-o-4_5-Q4_K_M.gguf`
- `audio/MiniCPM-o-4_5-audio-F16.gguf`
- `tts/MiniCPM-o-4_5-tts-F16.gguf`
- `tts/MiniCPM-o-4_5-projector-F16.gguf`
- `token2wav-gguf/encoder.gguf`
- `token2wav-gguf/flow_matching.gguf`
- `token2wav-gguf/flow_extra.gguf`
- `token2wav-gguf/hifigan2.gguf`
- `token2wav-gguf/prompt_cache.gguf`

## Current Uya Status

`audio2audio-real --audit-only` is not a waveform generator yet. It is the model-package gate for the full implementation. The next implementation layer is to bind the official tensor families discovered by audit:

- Audio encoder: `encoder.*`
- TTS model: `emb_code.*` and model-specific decoder tensors
- Projector: `linear1.*`, `linear2.*`
- Token2wav encoder: `after_norm.*` and encoder blocks
- Flow matching: `estimator.*`
- Flow extra: `input_embedding.*`
- HiFiGAN2: `conv_pre.*`, `ups.*`, `resblocks.*`, `conv_post.*`
- Prompt cache: `prompt_cache.*`

## Acceptance

The real end-to-end path is complete only when Uya can run:

```sh
build/minicpm-o-uya audio2audio-real \
  --llm models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --audio models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
  --tts models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  --projector models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  --token2wav-dir models/MiniCPM-o-4_5-gguf/token2wav-gguf \
  --ref-audio outputs/complex_case2/complex2_0000.wav \
  --input-audio outputs/complex_case2/complex2_0001.wav \
  --out outputs/uya_audio2audio_answer.wav
```

Required metrics for comparison with `llama.cpp-omni`:

- First audio/token latency.
- Total wall time.
- Peak RSS or allocator high-watermark.
- Audio duration, sample rate, and generated text if available.
- Whether reference voice and user turn were interpreted with the same `_0000=ref`, `_0001=user` protocol.
