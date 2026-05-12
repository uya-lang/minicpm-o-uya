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

Each file audit also prints a compact inventory with tensor count, dtype distribution, name-prefix counts, and first tensor shape samples. For the official bundle this highlights the next binding groups, for example `encoder.*`, `emb_code.*`, `linear1.*`, `estimator.*`, `conv_post.*`, and `prompt_cache.*`.

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


## Audio Encoder Alias Inventory

The official `audio/MiniCPM-o-4_5-audio-F16.gguf` currently audits as:

- `encoder.conv1.{weight,bias}`: mel front-end conv, first tensor shape sample `[1,1024]` for bias and `[3,80,1024]` for weight.
- `encoder.conv2.{weight,bias}`: second conv, weight shape `[3,1024,1024]`.
- `encoder.positional_embedding`: positional table, shape `[1024,1500]`.
- `encoder.blocks.N.*`: 24 transformer blocks, 15 tensors per block, including `attn.{query,key,value,out}`, `attn_ln`, `mlp`, and `mlp_ln`.
- `encoder.ln_post.{weight,bias}`: final audio encoder norm.
- `audio_projector.linear1.{weight,bias}` and `audio_projector.linear2.{weight,bias}`: projection into LLM hidden space.

The audit summary should report `audio_encoder.encoder=367` and `audio_projector=4` for this file.


## TTS Alias Inventory

The official `tts/MiniCPM-o-4_5-tts-F16.gguf` currently audits as:

- `emb_code.0.weight`: codec/audio token embedding, shape `[768,6562]`.
- `emb_text.weight`: text token embedding for the TTS decoder, shape `[768,152064]`.
- `token_embd.weight`: combined decoder token embedding, shape `[768,158626]`.
- `blk.N.*`: 20 Qwen-style decoder blocks, 9 tensors per block.
- `output_norm.weight`: final decoder norm.
- `projector_semantic.linear{1,2}.{weight,bias}`: semantic conditioning projector.
- `projector_spk.linear{1,2}.{weight,bias}`: speaker conditioning projector.
- `head_code.0.weight`: audio-code output head, shape `[6562,768]`.

The audit summary should report `tts_core=11` with `emb_code=1`, `emb_text=1`, `projector_semantic=4`, `projector_spk=4`, and `head_code=1` for this file.


## Token2Wav Alias Inventory

The official `token2wav-gguf` directory currently audits as:

- `encoder.gguf`: `embed.*`, `encoders.*`, `up_embed.*`, `up_encoders.*`, `pre_lookahead_layer.*`, `up_layer.*`, and `after_norm.*`.
- `flow_matching.gguf`: `estimator.*`, including input projection, DiT blocks, timestep MLP, and final layer.
- `flow_extra.gguf`: `input_embedding.*`, `encoder_proj.*`, and `spk_embed_affine_layer.*`.
- `hifigan2.gguf`: `conv_pre.*`, `ups.*`, `resblocks.*`, `source_resblocks.*`, `source_downs.*`, `m_source.*`, `f0_predictor.*`, and `conv_post.*`.
- `prompt_cache.gguf`: `prompt_cache.spk_cb`, `prompt_cache.estimator_*`, and `prompt_cache.conformer_*`.

After alias classification, these token2wav files should have zero unknown tensors in audit output. Kernel support is still pending; this table only establishes names and shapes for binding.

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
