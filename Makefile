UYA ?= /home/winger/uya/uya/bin/uya
SRC := src/main.uya
OUT := build/minicpm-o-uya
RELEASE_CFLAGS ?= -std=c99 -O3 -march=native -fno-builtin
UYA_GCC_JOBS ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: build build-debug build-release test fixtures inspect-fixture audit-fixture tokenizer-fixture tensor-fixture kernels-fixture quant-fixture qwen3-fixture generate-fixture vision-fixture audio-fixture audio-input-fixture speech-fixture audio2audio-fixture omni-fixture omni-chat-fixture stream-chat-fixture bench-fixture chat-fixture minicpmo-audit audio-real-bind tts-real-bind text-real-align audio2audio-real-audit audio2audio-real-input-audit clean FORCE

build: build-release

build-debug:
	mkdir -p build
	$(UYA) build $(SRC) -o $(OUT)

build-release:
	mkdir -p build
	$(UYA) build $(SRC) -o $(OUT)
	find .uyacache -name '*.o' -delete
	$(MAKE) -C .uyacache UYA_OUT="$(abspath $(OUT))" CC="$(CC)" CFLAGS="$(RELEASE_CFLAGS) -I." LDFLAGS="$(LDFLAGS)" -j$(UYA_GCC_JOBS)

test:
	$(UYA) test src/*.uya src/minicpmo/*.uya
	$(MAKE) inspect-fixture audit-fixture tokenizer-fixture tensor-fixture kernels-fixture quant-fixture qwen3-fixture generate-fixture vision-fixture audio-fixture audio-input-fixture speech-fixture audio2audio-fixture omni-fixture omni-chat-fixture stream-chat-fixture bench-fixture chat-fixture

fixtures:
	python3 tests/make_tiny_gguf.py

inspect-fixture: build fixtures
	$(OUT) inspect tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-inspect.out
	@if $(OUT) inspect tests/fixtures/tiny.gguf.part >/tmp/minicpm-o-uya-inspect-part.out 2>&1; then \
		echo "expected truncated fixture inspection to fail"; \
		exit 1; \
	else \
		grep -q "short file" /tmp/minicpm-o-uya-inspect-part.out; \
	fi

audit-fixture: build fixtures
	$(OUT) audit tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-audit.out
	grep -q "tensor dtype distribution" /tmp/minicpm-o-uya-audit.out
	grep -q "  vision: 15" /tmp/minicpm-o-uya-audit.out
	grep -q "  audio: 14" /tmp/minicpm-o-uya-audit.out
	grep -q "  speech: 2" /tmp/minicpm-o-uya-audit.out
	grep -q "Qwen3 text metadata: present" /tmp/minicpm-o-uya-audit.out
	grep -q "generation_supported=false" /tmp/minicpm-o-uya-audit.out
	$(OUT) audit tests/fixtures/bad_schema.gguf >/tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: missing tokenizer tokens/model metadata" /tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: missing root token embedding tensor" /tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: unknown tensor dtype" /tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: unclassified tensor branch" /tmp/minicpm-o-uya-audit-bad.out


tokenizer-fixture: build fixtures
	$(OUT) piece tests/fixtures/tiny.gguf 4 >/tmp/minicpm-o-uya-piece.out
	grep -qx "hello" /tmp/minicpm-o-uya-piece.out
	$(OUT) decode tests/fixtures/tiny.gguf 4 5 6 7 >/tmp/minicpm-o-uya-decode.out
	grep -qx "hello world!" /tmp/minicpm-o-uya-decode.out
	$(OUT) encode tests/fixtures/tiny.gguf "hello world!" >/tmp/minicpm-o-uya-encode-en.out
	grep -qx "4 5 6 7" /tmp/minicpm-o-uya-encode-en.out
	$(OUT) encode tests/fixtures/tiny.gguf "你好，世界" >/tmp/minicpm-o-uya-encode-zh.out
	grep -qx "8 9 10" /tmp/minicpm-o-uya-encode-zh.out
	$(OUT) encode tests/fixtures/tiny.gguf "<image> hello <audio>" >/tmp/minicpm-o-uya-encode-media.out
	grep -qx "12 5 4 5 13" /tmp/minicpm-o-uya-encode-media.out
	$(OUT) encode tests/fixtures/tiny_bpe.gguf "hello" >/tmp/minicpm-o-uya-encode-bpe-hello.out
	grep -qx "7" /tmp/minicpm-o-uya-encode-bpe-hello.out
	$(OUT) encode tests/fixtures/tiny_bpe.gguf " world" >/tmp/minicpm-o-uya-encode-bpe-space.out
	grep -qx "16" /tmp/minicpm-o-uya-encode-bpe-space.out
	$(OUT) encode tests/fixtures/tiny_bpe.gguf "你" >/tmp/minicpm-o-uya-encode-bpe-zh.out
	grep -qx "20" /tmp/minicpm-o-uya-encode-bpe-zh.out
	$(OUT) decode tests/fixtures/tiny_bpe.gguf 7 16 20 >/tmp/minicpm-o-uya-decode-bpe.out
	grep -qx "hello world你" /tmp/minicpm-o-uya-decode-bpe.out
	$(OUT) format-chat tests/fixtures/tiny.gguf "<image> hello" >/tmp/minicpm-o-uya-chat.out
	grep -q "<|im_start|>user" /tmp/minicpm-o-uya-chat.out
	grep -q "<image> hello<|im_end|>" /tmp/minicpm-o-uya-chat.out


tensor-fixture: build fixtures
	$(OUT) tensors tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-tensors.out
	grep -q "tensor table: count=45" /tmp/minicpm-o-uya-tensors.out
	grep -q "root token_embd=token_embd.weight root output=output.weight" /tmp/minicpm-o-uya-tensors.out
	grep -q "cache skeleton: llm_bytes=" /tmp/minicpm-o-uya-tensors.out
	$(OUT) tensors tests/fixtures/tiny.gguf --mmap >/tmp/minicpm-o-uya-tensors-mmap.out
	grep -q "mmap=yes" /tmp/minicpm-o-uya-tensors-mmap.out
	@if $(OUT) tensors tests/fixtures/bad_schema.gguf >/tmp/minicpm-o-uya-tensors-bad.out 2>&1; then 		echo "expected bad schema tensor table to fail"; 		exit 1; 	else 		grep -q "unknown ggml_type" /tmp/minicpm-o-uya-tensors-bad.out; 	fi
	@if $(OUT) tensors tests/fixtures/tiny_data_truncated.gguf >/tmp/minicpm-o-uya-tensors-trunc.out 2>&1; then 		echo "expected truncated tensor data to fail"; 		exit 1; 	else 		grep -q "data extends past file end" /tmp/minicpm-o-uya-tensors-trunc.out; 	fi


kernels-fixture: build
	$(OUT) kernels-smoke >/tmp/minicpm-o-uya-kernels.out
	grep -q "kernels smoke: PASS" /tmp/minicpm-o-uya-kernels.out

quant-fixture: build
	$(OUT) quant-smoke >/tmp/minicpm-o-uya-quant.out
	grep -q "quant smoke: PASS" /tmp/minicpm-o-uya-quant.out
	grep -q "unsupported quant dtype: tensor=blk.0.bad.weight dtype=IQ_UNKNOWN" /tmp/minicpm-o-uya-quant.out

qwen3-fixture: build fixtures
	$(OUT) qwen3-bind tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-qwen3.out
	grep -q "qwen3 bind: PASS" /tmp/minicpm-o-uya-qwen3.out
	grep -q "q_norm=yes k_norm=yes" /tmp/minicpm-o-uya-qwen3.out
	@if $(OUT) qwen3-bind tests/fixtures/tiny_qwen3_missing_q.gguf >/tmp/minicpm-o-uya-qwen3-missing.out 2>&1; then \
		echo "expected missing q projection binding to fail"; \
		exit 1; \
	else \
		grep -q "missing tensor blk.0.attn_q.weight" /tmp/minicpm-o-uya-qwen3-missing.out; \
	fi


generate-fixture: FORCE build fixtures
	$(OUT) generate tests/fixtures/tiny.gguf hello --max-new-tokens 1 --verbose >/tmp/minicpm-o-uya-generate.out
	grep -q "generate prompt_tokens: 4" /tmp/minicpm-o-uya-generate.out
	grep -q "sampled token\[0\]: 4 piece=hello" /tmp/minicpm-o-uya-generate.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --max-new-tokens 1 --temperature 1.0 --top-k 2 --seed 7 >/tmp/minicpm-o-uya-generate-seed-a.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --max-new-tokens 1 --temperature 1.0 --top-k 2 --seed 7 >/tmp/minicpm-o-uya-generate-seed-b.out
	cmp -s /tmp/minicpm-o-uya-generate-seed-a.out /tmp/minicpm-o-uya-generate-seed-b.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --max-new-tokens 3 >/tmp/minicpm-o-uya-generate-thread1.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --max-new-tokens 3 --threads 2 >/tmp/minicpm-o-uya-generate-thread2.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --max-new-tokens 1 --dump-hidden >/tmp/minicpm-o-uya-generate-hidden.out
	grep -q "prompt hidden: token_index=0 token=4 n=8" /tmp/minicpm-o-uya-generate-hidden.out
	grep -q "generated hidden: token_index=0 token=4 n=8" /tmp/minicpm-o-uya-generate-hidden.out
	cmp -s /tmp/minicpm-o-uya-generate-thread1.out /tmp/minicpm-o-uya-generate-thread2.out
	$(OUT) generate tests/fixtures/tiny.gguf hello >/tmp/minicpm-o-uya-generate-eos.out
	grep -q "stop: context_limit=32" /tmp/minicpm-o-uya-generate-eos.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --stop-token 4 >/tmp/minicpm-o-uya-generate-stop.out
	grep -q "stop: token=4" /tmp/minicpm-o-uya-generate-stop.out
	@if $(OUT) generate tests/fixtures/tiny.gguf hello --temperature nope >/tmp/minicpm-o-uya-generate-bad.out 2>&1; then \
		echo "expected invalid sampler args to fail"; \
		exit 1; \
	else \
		grep -q "error: invalid generate sampler args" /tmp/minicpm-o-uya-generate-bad.out; \
	fi

vision-fixture: FORCE build fixtures
	$(OUT) vision-preprocess-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_rgb.raw >/tmp/minicpm-o-uya-vision-pre.out
	grep -q "vision-preprocess: source=uyrg-u8 frames=1 output=2x2x3 tiles=1 placeholders=1 checksum=0xea7fa412" /tmp/minicpm-o-uya-vision-pre.out
	grep -q "vision-preprocess values\[0..5\]: -1.000000 -0.749019 -0.247058 0.003921 -0.498039 -0.247058" /tmp/minicpm-o-uya-vision-pre.out
	$(OUT) vision-preprocess-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_image.uyim >/tmp/minicpm-o-uya-vision-manifest.out
	grep -q "source=uyim-image-manifest frames=1 output=2x2x3 tiles=1 placeholders=1 checksum=0xea7fa412" /tmp/minicpm-o-uya-vision-manifest.out
	$(OUT) vision-preprocess-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_video.uyvm >/tmp/minicpm-o-uya-video-manifest.out
	grep -q "source=uyvm-video-manifest frames=2 output=2x2x3 tiles=2 placeholders=2 checksum=0xbc359fc5" /tmp/minicpm-o-uya-video-manifest.out
	$(OUT) vision-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_image.raw >/tmp/minicpm-o-uya-vision.out
	grep -q "vision-smoke: PASS" /tmp/minicpm-o-uya-vision.out
	grep -q "placeholders=1 span=1" /tmp/minicpm-o-uya-vision.out
	grep -q "vision embedding checksum: 0xb5a01b45" /tmp/minicpm-o-uya-vision.out
	grep -q "diff_l1=" /tmp/minicpm-o-uya-vision.out

audio-fixture: FORCE build fixtures
	$(OUT) audio-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_audio.raw >/tmp/minicpm-o-uya-audio.out
	$(OUT) audio-preprocess-smoke tests/fixtures/tiny_audio.pcm >/tmp/minicpm-o-uya-audio-pre.out
	grep -q "audio-smoke: PASS" /tmp/minicpm-o-uya-audio.out
	grep -q "audio-preprocess-smoke: PASS" /tmp/minicpm-o-uya-audio-pre.out
	grep -q "diff_l1=0.000000000" /tmp/minicpm-o-uya-audio-pre.out
	grep -q "audio raw: frames=1 mel_bins=4 dtype=f32 elements=4 checksum=0xbca0dcc" /tmp/minicpm-o-uya-audio.out
	grep -q "placeholders=1 span=1" /tmp/minicpm-o-uya-audio.out
	grep -q "audio embedding checksum: 0x625ac595" /tmp/minicpm-o-uya-audio.out
	grep -q "diff_l1=" /tmp/minicpm-o-uya-audio.out
	@if $(OUT) audio-smoke tests/fixtures/tiny_audio_missing.gguf tests/fixtures/tiny_audio.raw >/tmp/minicpm-o-uya-audio-missing.out 2>&1; then \
		echo "expected missing audio branch to fail"; \
		exit 1; \
	else \
		grep -q "missing tensor audio.conv1.weight" /tmp/minicpm-o-uya-audio-missing.out; \
	fi

speech-fixture: FORCE build fixtures
	$(OUT) speech-smoke tests/fixtures/tiny.gguf "hello world" /tmp/minicpm-o-uya-speech.wav >/tmp/minicpm-o-uya-speech.out
	grep -q "speech-smoke: PASS" /tmp/minicpm-o-uya-speech.out
	grep -q "speech schema: kind=cosyvoice2-codec hidden=4 codec_vocab=4 acoustic_vocab=4" /tmp/minicpm-o-uya-speech.out
	grep -q "vocoder schema: kind=vocos-like sample_rate=24000 samples_per_token=16 tensor=vocoder.proj.weight" /tmp/minicpm-o-uya-speech.out
	grep -q "speech tokens: 0 0 0 0 checksum=0x4b95f515" /tmp/minicpm-o-uya-speech.out
	grep -q "speech intermediate checksum: 0x4b95f515" /tmp/minicpm-o-uya-speech.out
	grep -q "speech waveform: samples=64 sample_rate=24000 rms=0.320025 peak=0.377256 range=\[0.203898,0.377256\] checksum=0x9f956c35" /tmp/minicpm-o-uya-speech.out
	grep -q "speech wav: path=/tmp/minicpm-o-uya-speech.wav bytes=172" /tmp/minicpm-o-uya-speech.out
	python3 -c 'from pathlib import Path; b=Path("/tmp/minicpm-o-uya-speech.wav").read_bytes(); assert len(b)==172 and b[:4]==b"RIFF" and b[8:12]==b"WAVE" and int.from_bytes(b[24:28],"little")==24000 and int.from_bytes(b[40:44],"little")==128'

audio-input-fixture: FORCE build
	python3 -c 'from pathlib import Path; import wave, struct, math; p=Path("/tmp/minicpm-o-uya-audio-probe.wav"); w=wave.open(str(p),"wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"".join(struct.pack("<h", int(12000*math.sin(2*math.pi*440*i/16000))) for i in range(32))); w.close()'
	$(OUT) audio-input-probe /tmp/minicpm-o-uya-audio-probe.wav >/tmp/minicpm-o-uya-audio-probe.out
	grep -q "audio input\[probe\]: path=/tmp/minicpm-o-uya-audio-probe.wav container=wav dtype=s16 sample_rate=16000 channels=1 samples=32" /tmp/minicpm-o-uya-audio-probe.out
	grep -q "audio-input-probe: PASS" /tmp/minicpm-o-uya-audio-probe.out
	python3 -c 'from pathlib import Path; import struct; p=Path("/tmp/minicpm-o-uya-audio-probe.uyap.pcm"); samples=[0, 1024, -2048, 4096]; p.write_bytes(struct.pack("<IIIIII", 0x50415955, 1, 16000, 1, len(samples), 0) + b"".join(struct.pack("<i", x) for x in samples))'
	$(OUT) audio-input-probe /tmp/minicpm-o-uya-audio-probe.uyap.pcm >/tmp/minicpm-o-uya-audio-probe-uyap.out
	grep -q "container=uyap dtype=s16-i32-container sample_rate=16000 channels=1 samples=4" /tmp/minicpm-o-uya-audio-probe-uyap.out
	python3 -c 'from pathlib import Path; import wave, struct; p=Path("/tmp/minicpm-o-uya-audio-probe-bad.wav"); w=wave.open(str(p),"wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(8000); w.writeframes(struct.pack("<hhhh", 1, 2, 3, 4)); w.close()'
	@if $(OUT) audio-input-probe /tmp/minicpm-o-uya-audio-probe-bad.wav >/tmp/minicpm-o-uya-audio-probe-bad.out 2>&1; then \
		echo "expected non-16k-mono wav probe to fail"; \
		exit 1; \
	else \
		grep -q "unsupported sample_rate=8000 expected=16000" /tmp/minicpm-o-uya-audio-probe-bad.out; \
	fi

audio2audio-fixture: FORCE build fixtures
	$(OUT) audio2audio-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_audio.pcm /tmp/minicpm-o-uya-audio2audio.wav >/tmp/minicpm-o-uya-audio2audio.out
	$(OUT) audio2audio-smoke tests/fixtures/tiny.gguf --audio-model tests/fixtures/tiny.gguf --speech-model tests/fixtures/tiny.gguf --vocoder-model tests/fixtures/tiny.gguf tests/fixtures/tiny_audio.pcm /tmp/minicpm-o-uya-audio2audio-multi.wav >/tmp/minicpm-o-uya-audio2audio-multi.out
	grep -q "audio2audio-smoke: PASS" /tmp/minicpm-o-uya-audio2audio.out
	grep -q "audio2audio-smoke: PASS" /tmp/minicpm-o-uya-audio2audio-multi.out
	grep -q "audio2audio models: text=tests/fixtures/tiny.gguf audio=tests/fixtures/tiny.gguf speech=tests/fixtures/tiny.gguf vocoder=tests/fixtures/tiny.gguf" /tmp/minicpm-o-uya-audio2audio-multi.out
	grep -q "audio2audio input: source=uyap-pcm input_frames=7 frames=1 mel_bins=4" /tmp/minicpm-o-uya-audio2audio.out
	grep -q "audio2audio audio: placeholders=1 span=1 embedding_checksum=" /tmp/minicpm-o-uya-audio2audio.out
	grep -q "diff_l1=" /tmp/minicpm-o-uya-audio2audio.out
	grep -q "audio2audio speech: prompt_tokens=4 generated=4" /tmp/minicpm-o-uya-audio2audio.out
	grep -q "audio2audio waveform: samples=64 sample_rate=24000" /tmp/minicpm-o-uya-audio2audio.out
	grep -q "audio2audio wav: path=/tmp/minicpm-o-uya-audio2audio.wav bytes=172" /tmp/minicpm-o-uya-audio2audio.out
	python3 -c 'from pathlib import Path; b=Path("/tmp/minicpm-o-uya-audio2audio.wav").read_bytes(); assert len(b)==172 and b[:4]==b"RIFF" and b[8:12]==b"WAVE" and int.from_bytes(b[24:28],"little")==24000 and int.from_bytes(b[40:44],"little")==128'

minicpmo-audit: build
	@if [ -z "$(MINICPM_O_GGUF)" ]; then \
		echo "usage: MINICPM_O_GGUF=/path/to/model.gguf make minicpmo-audit"; \
		exit 2; \
	fi
	$(OUT) audit "$(MINICPM_O_GGUF)"

audio-real-bind: build
	@if [ -z "$(MINICPM_O_REAL_BUNDLE)" ]; then \
		echo "usage: MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf make audio-real-bind"; \
		exit 2; \
	fi
	$(OUT) audio-bind "$(MINICPM_O_REAL_BUNDLE)/audio/MiniCPM-o-4_5-audio-F16.gguf"

tts-real-bind: build
	@if [ -z "$(MINICPM_O_REAL_BUNDLE)" ]; then \
		echo "usage: MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf make tts-real-bind"; \
		exit 2; \
	fi
	$(OUT) tts-bind "$(MINICPM_O_REAL_BUNDLE)/tts/MiniCPM-o-4_5-tts-F16.gguf"

text-real-align: build
	@if [ -z "$(MINICPM_O_TEXT_GGUF)" ] || [ -z "$(LLAMA_COMPLETION_BIN)" ]; then \
		echo "usage: MINICPM_O_TEXT_GGUF=/path/to/MiniCPM-o-4_5-Q4_K_M.gguf LLAMA_COMPLETION_BIN=/path/to/llama-completion [MINICPM_O_ALIGN_PROMPT=你好] [MINICPM_O_ALIGN_THREADS=4] make text-real-align"; \
		exit 2; \
	fi
	python3 tests/compare_text_alignment.py \
		--uya "$(OUT)" \
		--llama "$(LLAMA_COMPLETION_BIN)" \
		--model "$(MINICPM_O_TEXT_GGUF)" \
		--prompt "$${MINICPM_O_ALIGN_PROMPT:-你好}" \
		--threads "$${MINICPM_O_ALIGN_THREADS:-4}" \
		--ctx "$${MINICPM_O_ALIGN_CTX:-256}" \
		--max-new-tokens "$${MINICPM_O_ALIGN_TOKENS:-1}"

audio2audio-real-audit: build
	@if [ -z "$(MINICPM_O_REAL_BUNDLE)" ]; then \
		echo "usage: MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf make audio2audio-real-audit"; \
		exit 2; \
	fi
	$(OUT) audio2audio-real --audit-only \
		--llm "$(MINICPM_O_REAL_BUNDLE)/MiniCPM-o-4_5-Q4_K_M.gguf" \
		--audio "$(MINICPM_O_REAL_BUNDLE)/audio/MiniCPM-o-4_5-audio-F16.gguf" \
		--tts "$(MINICPM_O_REAL_BUNDLE)/tts/MiniCPM-o-4_5-tts-F16.gguf" \
		--projector "$(MINICPM_O_REAL_BUNDLE)/tts/MiniCPM-o-4_5-projector-F16.gguf" \
		--token2wav-dir "$(MINICPM_O_REAL_BUNDLE)/token2wav-gguf"

audio2audio-real-input-audit: build
	@if [ -z "$(MINICPM_O_REAL_BUNDLE)" ] || [ -z "$(MINICPM_O_REAL_REF_AUDIO)" ] || [ -z "$(MINICPM_O_REAL_USER_AUDIO)" ]; then \
		echo "usage: MINICPM_O_REAL_BUNDLE=/path/to/MiniCPM-o-4_5-gguf MINICPM_O_REAL_REF_AUDIO=ref.wav MINICPM_O_REAL_USER_AUDIO=user.wav [MINICPM_O_REAL_OUT=out.wav] make audio2audio-real-input-audit"; \
		exit 2; \
	fi
	$(OUT) audio2audio-real --audit-only \
		--llm "$(MINICPM_O_REAL_BUNDLE)/MiniCPM-o-4_5-Q4_K_M.gguf" \
		--audio "$(MINICPM_O_REAL_BUNDLE)/audio/MiniCPM-o-4_5-audio-F16.gguf" \
		--tts "$(MINICPM_O_REAL_BUNDLE)/tts/MiniCPM-o-4_5-tts-F16.gguf" \
		--projector "$(MINICPM_O_REAL_BUNDLE)/tts/MiniCPM-o-4_5-projector-F16.gguf" \
		--token2wav-dir "$(MINICPM_O_REAL_BUNDLE)/token2wav-gguf" \
		--ref-audio "$(MINICPM_O_REAL_REF_AUDIO)" \
		--input-audio "$(MINICPM_O_REAL_USER_AUDIO)" \
		--out "$${MINICPM_O_REAL_OUT:-/tmp/minicpm-o-uya-real-answer.wav}"

clean:
	rm -rf build

omni-fixture: FORCE build fixtures
	$(OUT) omni-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_omni.json >/tmp/minicpm-o-uya-omni.out
	grep -q "omni-smoke: PASS" /tmp/minicpm-o-uya-omni.out
	grep -q "omni manifest: events=7 text=2 image=1 video_frame=1 audio_chunk=1 speech_request=1 control=1" /tmp/minicpm-o-uya-omni.out
	grep -q "omni event order: text image text video_frame audio_chunk speech_request control" /tmp/minicpm-o-uya-omni.out
	grep -q "omni tokens: 4 5 12 6 14 13" /tmp/minicpm-o-uya-omni.out
	grep -q "omni spans: count=3 media_embeddings=3" /tmp/minicpm-o-uya-omni.out
	grep -q "span\[0\]: kind=image token_start=2 token_count=1 embed_start=0 embed_count=1" /tmp/minicpm-o-uya-omni.out
	grep -q "span\[1\]: kind=video_frame token_start=4 token_count=1 embed_start=1 embed_count=1" /tmp/minicpm-o-uya-omni.out
	grep -q "span\[2\]: kind=audio_chunk token_start=5 token_count=1 embed_start=2 embed_count=1" /tmp/minicpm-o-uya-omni.out
	grep -q "omni context: tokens=6 text_tokens=3 media_embeddings=3 effective=6 attention=causal+media-spans checksum=0x4e22ca" /tmp/minicpm-o-uya-omni.out

omni-chat-fixture: FORCE build fixtures
	printf "hello\nworld\n" | $(OUT) omni-chat tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-omni-chat-text.out
	grep -q "omni-chat: model loaded once" /tmp/minicpm-o-uya-omni-chat-text.out
	grep -q "turn\[0\]: input=text events=1 tokens=1 spans=0 media_embeddings=0 text_tokens=1 cache_tokens=1 resets=0" /tmp/minicpm-o-uya-omni-chat-text.out
	grep -q "assistant text: hello" /tmp/minicpm-o-uya-omni-chat-text.out
	grep -q "turn\[1\]: input=text events=1 tokens=1 spans=0 media_embeddings=0 text_tokens=1 cache_tokens=2 resets=0" /tmp/minicpm-o-uya-omni-chat-text.out
	grep -q "assistant text: world" /tmp/minicpm-o-uya-omni-chat-text.out
	grep -q "omni-chat: done turns=2 resets=0 cache_tokens=2" /tmp/minicpm-o-uya-omni-chat-text.out
	printf "manifest tests/fixtures/tiny_omni.json\n" | $(OUT) omni-chat tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-omni-chat-manifest.out
	grep -q "turn\[0\]: input=manifest events=7 tokens=6 spans=3 media_embeddings=3 text_tokens=3 cache_tokens=6 resets=0" /tmp/minicpm-o-uya-omni-chat-manifest.out
	grep -q "assistant text: hello" /tmp/minicpm-o-uya-omni-chat-manifest.out
	grep -q "speech output: unsupported in blocking omni-chat; queued_requests=1 text output preserved" /tmp/minicpm-o-uya-omni-chat-manifest.out
	grep -q "turn boundary: turn=0 cache_policy=preserve-until-overflow" /tmp/minicpm-o-uya-omni-chat-manifest.out

stream-chat-fixture: FORCE build fixtures
	$(OUT) stream-chat tests/fixtures/tiny.gguf tests/fixtures/tiny_stream.json >/tmp/minicpm-o-uya-stream-chat.out
	grep -q "stream-chat: PASS" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "stream queue: enqueued=6 high_watermark=6 dropped=0 backpressure=drop-oldest hits=0" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "audio ring: used=128 peak=128 dropped=0 chunks=1" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "audio ring: used=192 peak=192 dropped=0 chunks=2" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "partial output\[0\]: hello" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "partial output\[1\]: world" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "control interrupt: cleared_pending_speech=1 cancellations=1" /tmp/minicpm-o-uya-stream-chat.out
	grep -q "stream summary: events=6 audio_chunks=2 partial_callbacks=2 vision_encoded=0 audio_encoded=2 pending_speech=0 cancellations=1 encode_ms=10 decode_ms=8 vocoder_ms=4 ring_peak=192 dropped_audio=0" /tmp/minicpm-o-uya-stream-chat.out

bench-fixture: FORCE build fixtures
	$(OUT) bench tests/fixtures/tiny.gguf tests/fixtures/tiny_omni.json >/tmp/minicpm-o-uya-bench.out
	grep -q "bench: PASS" /tmp/minicpm-o-uya-bench.out
	grep -q "bench load: ms=3 mode=tiny-mmap-metadata" /tmp/minicpm-o-uya-bench.out
	grep -q "bench text_prompt: 500.000 tokens/s units=1 ms=2" /tmp/minicpm-o-uya-bench.out
	grep -q "bench text_decode: 333.333 tokens/s units=1 ms=3" /tmp/minicpm-o-uya-bench.out
	grep -q "bench vision_encode: ms=7 frames=1 tiles=1" /tmp/minicpm-o-uya-bench.out
	grep -q "bench audio_encode: ms=5 chunks=1" /tmp/minicpm-o-uya-bench.out
	grep -q "bench vocoder: 16000.000 samples/s units=64 ms=4" /tmp/minicpm-o-uya-bench.out
	grep -q "bench memory: peak_estimate_bytes=78336 llm=512 vision=4096 audio=4096 speech=4096 scratch=65536" /tmp/minicpm-o-uya-bench.out
	grep -q "bench optimize: hot_matvec=reference-scalar kv_cache=contiguous prefill=tiled-smoke media_scratch=reused" /tmp/minicpm-o-uya-bench.out
	grep -q "bench reference_error: text=0.000000 vision=0.000000 audio=0.000000 vocoder=0.000000" /tmp/minicpm-o-uya-bench.out
	grep -q "omni-smoke: PASS" /tmp/minicpm-o-uya-bench.out
	$(OUT) bench tests/fixtures/tiny.gguf --n-prompt 4 --n-gen 4 --repetitions 1 --no-warmup >/tmp/minicpm-o-uya-bench-real.out
	grep -q "bench config: mode=text-real prompt_tokens=4 gen_tokens=4 repetitions=1 warmup=0 threads=1" /tmp/minicpm-o-uya-bench-real.out
	grep -q "seed_prompt=hello seed_tokens=1 vocab=19 context=32 hidden=8 layers=1 kv_heads=1 head_dim=4" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench load: ms=" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench text_prompt:" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench text_decode:" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench memory: peak_estimate_bytes=" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench optimize: hot_matvec=q8_0_uya_parallel kv_cache=contiguous rope=precomputed sampler=fixed-seed" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench reference_error: n/a mode=real-timing" /tmp/minicpm-o-uya-bench-real.out
	grep -q "bench: PASS" /tmp/minicpm-o-uya-bench-real.out

chat-fixture: FORCE build fixtures
	printf "hello\n" | $(OUT) chat tests/fixtures/tiny.gguf --max-new-tokens 1 --verbose >/tmp/minicpm-o-uya-chat-repl.out
	grep -q "chat>" /tmp/minicpm-o-uya-chat-repl.out
	grep -q "sampled token\[0\]: 6 piece=world" /tmp/minicpm-o-uya-chat-repl.out
	@if printf "hello\nhello\nhello\n" | $(OUT) chat tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-chat-overflow.out 2>&1; then \
		echo "expected chat context overflow to fail"; \
		exit 1; \
	else \
		grep -q "error: context overflow" /tmp/minicpm-o-uya-chat-overflow.out; \
	fi

FORCE:
