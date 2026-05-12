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

# Optional input-protocol audit. This validates that ref/user audio are true
# 16 kHz mono WAV PCM16/WAV F32/UYAP PCM, without auto-transcoding.
build/minicpm-o-uya audio2audio-real --audit-only \
  --llm models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --audio models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
  --tts models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  --projector models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  --token2wav-dir models/MiniCPM-o-4_5-gguf/token2wav-gguf \
  --ref-audio outputs/complex_case2/complex2_0000.wav \
  --input-audio outputs/complex_case2/complex2_0001.wav \
  --out outputs/uya_audio2audio_answer.wav

# llama.cpp-omni style multi-file protocol also works in audit mode:
# prefix_0000.wav is the reference voice, prefix_0001.wav.. are user turns.
build/minicpm-o-uya audio2audio-real --audit-only \
  --llm models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --audio models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
  --tts models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  --projector models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  --token2wav-dir models/MiniCPM-o-4_5-gguf/token2wav-gguf \
  --test-prefix outputs/complex_case2/complex2 \
  --count 2 \
  --out outputs/uya_audio2audio_answer.wav
```

Expected success marker:

```text
audio2audio-real: audit-only PASS files=9
audio2audio-real: inference_supported=false next=audio-encoder-forward-and-tts-bind
```

When audio inputs are supplied, the audit additionally prints `audio2audio-real input protocol`, per-file `audio input[ref]` / `audio input[user]` summaries, sample checksum, duration, peak, and RMS. Non-16 kHz mono files fail fast with an explicit transcode/downmix diagnostic. The same check can be run with:

```sh
MINICPM_O_REAL_BUNDLE=models/MiniCPM-o-4_5-gguf \
MINICPM_O_REAL_REF_AUDIO=outputs/complex_case2/complex2_0000.wav \
MINICPM_O_REAL_USER_AUDIO=outputs/complex_case2/complex2_0001.wav \
make audio2audio-real-input-audit
```

If you also want correctness-first audio encoder embeddings during the same input audit, add `MINICPM_O_REAL_ENCODE_PROBE=1` to the make command, or pass `--encode-probe` directly to `audio2audio-real --audit-only`.

Each file audit also prints a compact inventory with tensor count, dtype distribution, name-prefix counts, and first tensor shape samples. The audio file is additionally checked with `audio-bind`, which binds the official `encoder.conv*`, 24 `encoder.blocks.*` transformer layers, `encoder.ln_post.*`, and `audio_projector.linear{1,2}.*` tensors. The current official audio bind checks 371 tensors and validates the mel filter metadata (`n_mel=80`, `n_fft/filter_bins=201`, `filters=16080`). The TTS file is additionally checked with `tts-bind`, which binds `emb_code.*`, `emb_text.*`, the 20-layer decoder `blk.*`, `projector_semantic.*`, `projector_spk.*`, `head_code.*`, and checks 193 tensors. The remaining binding groups are token2wav families such as `estimator.*`, `conv_post.*`, and `prompt_cache.*`.

Audit diagnostics include candidate next actions for unknown dtype, unclassified tensor prefixes, and key alias shape mismatches. The current shape sanity checks cover representative official MiniCPM-o 4.5 audio, TTS, projector, token2wav, HiFiGAN2, and prompt-cache tensors. A shape mismatch is treated as unsupported until the alias table or expected shape is updated.

The Qwen3 text path now accepts the MiniCPM-o 4.5 text dimensions used by `MiniCPM-o-4_5-Q4_K_M.gguf`: `hidden=4096`, `layers=36`, `ffn=12288`, `ctx>=4096`, and `vocab=151748`. Large forward workspaces and logits buffers are heap-backed rather than fixed stack arrays.
Runtime text generation has real GGML-layout fused matvec support for Q4_K, Q5_K, and Q6_K, plus Q4_K/Q5_K/Q6_K embedding-row dequantization. The official Q4_K_M LLM now passes a text-only one-token generate smoke, and its audited dtype distribution is only `F32/Q4_K/Q6_K`; Q8_K/IQ runtime matvec remains guarded for other bundles. Use `generate ... --dump-hidden` to emit prompt/generated token hidden-state summaries; on the official MiniCPM-o 4.5 LLM this reports `n=4096`, which is the handoff shape needed by the later TTS projector path.
A manual text alignment target is available for the same text GGUF: `MINICPM_O_TEXT_GGUF=/path/to/MiniCPM-o-4_5-Q4_K_M.gguf LLAMA_COMPLETION_BIN=/path/to/llama-completion make text-real-align`. It compares Uya and llama.cpp greedy one-token text output for the same prompt before moving deeper into audio/TTS alignment.

A manual audio preprocessing probe is available as `build/minicpm-o-uya audio-real-preprocess-probe <audio.wav|audio.uyap.pcm>`, and a numeric mel probe is available as `MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf MINICPM_O_REAL_USER_AUDIO=user.wav make audio-real-mel-probe`. It validates 16 kHz mono PCM16/F32 WAV or UYAP input and prints the llama.cpp-omni MiniCPM-o preprocessing plan: `frame_size=400`, `filter_bins=201`, `hop_length=160`, `mel_bins=80`, 100ms sample alignment, 200-sample center padding on each side, conv2 downsampling, and pool(5,5) `encoder_positions`. The `audio2audio-real --audit-only` input path now prints this plan for both ref and user audio. The numeric mel probe runs periodic Hann, DFT/STFT power, the GGUF `filters` mel filterbank, and llama.cpp-omni style log10 clamp/normalization, then prints frames/elements/checksum/sample values.

A correctness-first encoder forward probe is now available as `MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf MINICPM_O_REAL_USER_AUDIO=user.wav make audio-real-encode-probe`. It runs the official `encoder.conv1/2`, 24-layer transformer, `encoder.ln_post`, `audio_projector.linear1/2`, and final pool(5,5), then prints `mel_frames`, `conv_tokens`, `pooled_tokens`, `n_pos`, `n_embd`, embedding checksum, and encode wall time. On the local 1-second sample `/tmp/minicpm-o-uya-real-probe.wav`, the current reference implementation reports `n_pos=10`, `n_embd=4096`, checksum `0xb95329ca`, and `encode_ms≈96870.831`, so this path is still intended for correctness/alignment work rather than speed.

`audio-real-mel-probe` also accepts `--dump-f32 out.uyml`, which writes a compact binary dump for full-array comparison:

- Header: `magic=UYML`, `version=1`, `frames(u32)`, `mel_bins(u32)`
- Payload: little-endian row-major `f32`, matching Uya's internal `mel.data[j * n_len + i]` layout

When you have a `llama.cpp-omni` `log_mel_spectrogram.json` dump, compare it with:

```sh
MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf \
MINICPM_O_REAL_USER_AUDIO=user.wav \
MINICPM_O_REAL_MEL_JSON=/path/to/log_mel_spectrogram.json \
make audio-real-mel-align
```

The helper runs Uya with `--dump-f32`, loads the llama.cpp-omni JSON dump, and reports shape, checksum, `L1`, `mean_abs`, and `max_abs`.

Current manual alignment thresholds default to `max_abs <= 2e-3` and `mean_abs <= 2e-5`. On the local `outputs/complex_case2` baseline, both the reference voice chunk and the user audio chunk pass under that bound:

- `complex2_0000.wav` vs `log_mel_spectrogram_0000.json`: `mean_abs=0.000012707`, `max_abs=0.001426596`
- `complex2_0001.wav` vs `log_mel_spectrogram_0001.json`: `mean_abs=0.000012695`, `max_abs=0.001400372`

Current upstream `llama.cpp-omni/tools/omni/audition.cpp` writes `log_mel_spectrogram.json` only when the local `debug` argument passed into `log_mel_spectrogram(...)` is switched from `false` to `true`; there is not yet a CLI flag for this. For Uya, `audio2audio-real --audit-only` also accepts an optional `--encode-probe` flag, which runs the same correctness-first encoder probe on the validated ref/user audio inputs and prints the resulting embedding diagnostics.

## TTS Condition Merge Probe

The next correctness gate after audio encoder alignment is the TTS conditioning merge:

```text
merged_embeds =
  emb_text(filtered_token_ids)
  + normalize(projector_semantic(filtered_hidden_states))
```

Uya now has a standalone probe for that stage:

```sh
build/minicpm-o-uya tts-condition-probe \
  models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  llama.cpp-omni/tools/omni/output/round_000/llm_debug/chunk_0/llm_token_ids.txt \
  llama.cpp-omni/tools/omni/output/round_000/llm_debug/chunk_0/llm_hidden_states.bin \
  --dump-merged /tmp/minicpm-o-uya-merged.bin \
  --dump-projected /tmp/minicpm-o-uya-projected.bin \
  --dump-filtered-token-ids /tmp/minicpm-o-uya-filtered-ids.txt
```

It reads the same `llama.cpp-omni` `llm_debug/chunk_*` token-id and hidden-state dumps, filters TTS-invalid tokens with the same rule (`special ids`, token `271`, and `>=150000`), runs the official `emb_text` and `linear1/linear2` projector weights, applies per-token L2 normalization, and writes binary dumps with a simple header:

- `u32 n_tokens`
- `u32 n_embd`
- row-major `f32` payload

`--dump-condition out.bin` is also available if you want the prefill condition with `audio_bos` appended. By default `audio_bos_id=151687`, and `--audio-bos-id` can override it for future bundles.

For direct alignment against the local `llama.cpp-omni` `merged_embeddings.bin`, use:

```sh
MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf \
MINICPM_O_TTS_TOKEN_IDS=/path/to/llm_token_ids.txt \
MINICPM_O_TTS_HIDDEN_BIN=/path/to/llm_hidden_states.bin \
MINICPM_O_TTS_MERGED_BIN=/path/to/merged_embeddings.bin \
make tts-merge-align
```

Current default thresholds are much tighter than the mel path because this stage is pure F32:

- `max_abs <= 1e-5`
- `mean_abs <= 1e-6`

On the local `llama.cpp-omni/tools/omni/output/round_000/llm_debug` baseline, all four saved chunks pass comfortably:

- `chunk_0`: `mean_abs=1.44496e-8`, `max_abs=2.38419e-7`
- `chunk_1`: `mean_abs=1.76361e-8`, `max_abs=2.38419e-7`
- `chunk_2`: `mean_abs=1.73070e-8`, `max_abs=2.38419e-7`
- `chunk_3`: `mean_abs=1.56946e-8`, `max_abs=2.38419e-7`

## TTS Simplex Probe

Uya now also has a standalone real TTS decoder probe that consumes a whole `llm_debug/chunk_*` directory and emits `audio_tokens_chunk_*.txt/bin` in the same relative-token format as `llama.cpp-omni`:

```sh
build/minicpm-o-uya tts-simplex-probe \
  models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  llama.cpp-omni/tools/omni/output/round_000/llm_debug \
  --count 4 \
  --out-dir /tmp/minicpm-o-uya-ttsprobe
```

Current Uya probe behavior:

- loads the official 20-layer Llama-style TTS decoder from `tts.gguf`
- pre-fills the same assistant-side prompt prefix before chunk audio generation
- reuses `tts-condition-probe` logic for `merged_embeds`
- keeps TTS KV cache and generated audio-token history across chunks
- writes relative audio token IDs, one file per chunk

This is the first real Uya implementation of:

- TTS prefill/decode cache
- chunked text/hidden-state queue consumption
- real audio token dump emission

Current limitation: the local saved `llama.cpp-omni` `audio_tokens_chunk_*.txt` files were generated from an unpinned sampler seed, so exact token-by-token replay is not yet expected to match. The Uya probe already supports `--compare-dir` for future seeded baselines, but today's `round_000/tts_wav` artifacts should be treated as a structural/debug reference rather than a strict deterministic oracle.

Even without a pinned baseline seed, the probe is now observability-friendly:

- `tts-simplex prompt: ... prefill_ms=...`
- `tts-simplex timing: chunk=N merge_ms=... decode_ms=... cache_tokens=... generated=...`
- `tts-simplex compare-summary: chunk=N ref=... uya=... prefix_match=... exact=...`

For repeated manual comparison runs, use:

```sh
MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf \
MINICPM_O_TTS_LLM_DEBUG_DIR=/path/to/llm_debug \
MINICPM_O_TTS_COMPARE_DIR=/path/to/tts_wav \
make tts-token-align
```

If you have a future seeded baseline and want strict pass/fail, add `MINICPM_O_TTS_REQUIRE_EXACT=1`.

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

Uya now also has a bind-only bundle command:

```sh
build/minicpm-o-uya token2wav-bind models/MiniCPM-o-4_5-gguf/token2wav-gguf
# or
MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf make token2wav-real-bind
```

Current bind coverage:

- `encoder.gguf`: `after_norm.*`, `embed.*`, `encoders.*`, `pre_lookahead_layer.*`, `up_embed.*`, `up_encoders.*`, `up_layer.*`
- `flow_matching.gguf`: `estimator.in_proj.*`, `estimator.t_embedder.*`, all detected `estimator.blocks.*`, and `estimator.final_layer.*`
- `flow_extra.gguf`: `input_embedding.*`, `encoder_proj.*`, `spk_embed_affine_layer.*`
- `hifigan2.gguf`: `conv_pre.*`, `conv_post.*`, `ups.*`, `source_downs.*`, `m_source.*`, `f0_predictor.*`
- `prompt_cache.gguf`: all 5 official cache tensors

This is still bind-only coverage: no token2wav / flow / HiFiGAN forward math is claimed yet.

Uya now also has two prompt-cache-side probes that prepare the next forward step:

```sh
build/minicpm-o-uya token2wav-prompt-cache-probe \
  models/MiniCPM-o-4_5-gguf/token2wav-gguf/prompt_cache.gguf

build/minicpm-o-uya token2wav-window-probe \
  models/MiniCPM-o-4_5-gguf/token2wav-gguf/prompt_cache.gguf \
  llama.cpp-omni/tools/omni/output/round_000/tts_wav/audio_tokens_chunk_0.txt
```

Current `prompt_cache` probe reports:

- `version`
- `n_timesteps`
- `temperature_bits`
- `pre_lookahead`
- `chunk_main`
- `chunk_total`
- `up_rate`
- cache tensor checksums for `spk_cb`, `conformer_*`, and `estimator_*`

Current `window` probe reports the actual `28/25` streaming calls, for example on the local `audio_tokens_chunk_0.txt` baseline:

- `window[0]`: `start=0 end=28`
- `window[1]`: `start=25 end=53`
- `window[2]`: final tail `start=50 end=61`

## Current Uya Status

`audio2audio-real --audit-only` is not a waveform generator yet. It is the model-package and input-protocol gate for the full implementation. It now accepts either explicit `--ref-audio` plus `--user-audio`/`--input-audio`, `--input-prefix prefix` for one user turn, or `--test-prefix prefix --count N` for the llama.cpp-omni convention where `0000` is reference voice and `0001..` are user turns. The next implementation layer is to run the official audio encoder forward and bind the remaining TTS/token2wav tensor families discovered by audit:

- Audio encoder: bind complete, and `audio-real-encode-probe` now runs correctness-first `conv + transformer + projector + pool` forward for short real clips; it is still too slow for practical long-turn inference.
- TTS model: bind-only complete for `emb_code.*`, `emb_text.*`, decoder `blk.*`, `projector_semantic.*`, `projector_spk.*`, and `head_code.*`; `tts-condition-probe` now aligns `emb_text + projector_semantic + normalize + merge` against `llama.cpp-omni`, while decode/cache forward still pending.
- Projector: `linear1.*`, `linear2.*`
- Token2wav encoder: bind-only complete for `after_norm.*`, `embed.*`, `encoders.*`, `pre_lookahead_layer.*`, `up_embed.*`, `up_encoders.*`, and `up_layer.*`
- Flow matching: bind-only complete for `estimator.in_proj.*`, `estimator.t_embedder.*`, `estimator.blocks.*`, and `estimator.final_layer.*`
- Flow extra: bind-only complete for `input_embedding.*`, `encoder_proj.*`, and `spk_embed_affine_layer.*`
- HiFiGAN2: bind-only complete for `conv_pre.*`, `conv_post.*`, `ups.*`, `source_downs.*`, `m_source.*`, and `f0_predictor.*`
- Prompt cache: bind-only complete for all 5 official `prompt_cache.*` tensors

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

# llama.cpp-omni style multi-file protocol also works in audit mode:
# prefix_0000.wav is the reference voice, prefix_0001.wav.. are user turns.
build/minicpm-o-uya audio2audio-real --audit-only \
  --llm models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --audio models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf \
  --tts models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-tts-F16.gguf \
  --projector models/MiniCPM-o-4_5-gguf/tts/MiniCPM-o-4_5-projector-F16.gguf \
  --token2wav-dir models/MiniCPM-o-4_5-gguf/token2wav-gguf \
  --test-prefix outputs/complex_case2/complex2 \
  --count 2 \
  --out outputs/uya_audio2audio_answer.wav
```

Required metrics for comparison with `llama.cpp-omni`:

- First audio/token latency.
- Total wall time.
- Peak RSS or allocator high-watermark.
- Audio duration, sample rate, and generated text if available.
- Whether reference voice and user turn were interpreted with the same `_0000=ref`, `_0001=user` protocol.
