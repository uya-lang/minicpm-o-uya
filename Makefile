UYA ?= /home/winger/uya/uya/bin/uya
SRC := src/main.uya
OUT := build/minicpm-o-uya

.PHONY: build test fixtures inspect-fixture audit-fixture tokenizer-fixture tensor-fixture kernels-fixture quant-fixture qwen3-fixture generate-fixture vision-fixture audio-fixture speech-fixture omni-fixture chat-fixture minicpmo-audit clean FORCE

build:
	mkdir -p build
	$(UYA) build $(SRC) -o $(OUT)

test:
	$(UYA) test src/*.uya src/minicpmo/*.uya
	$(MAKE) inspect-fixture audit-fixture tokenizer-fixture tensor-fixture kernels-fixture quant-fixture qwen3-fixture generate-fixture vision-fixture audio-fixture speech-fixture omni-fixture chat-fixture

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
	$(OUT) generate tests/fixtures/tiny.gguf hello >/tmp/minicpm-o-uya-generate.out
	grep -q "generate prompt_tokens: 4" /tmp/minicpm-o-uya-generate.out
	grep -q "sampled token\[0\]: 4 piece=hello" /tmp/minicpm-o-uya-generate.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --temperature 1.0 --top-k 2 --seed 7 >/tmp/minicpm-o-uya-generate-seed-a.out
	$(OUT) generate tests/fixtures/tiny.gguf hello --temperature 1.0 --top-k 2 --seed 7 >/tmp/minicpm-o-uya-generate-seed-b.out
	cmp -s /tmp/minicpm-o-uya-generate-seed-a.out /tmp/minicpm-o-uya-generate-seed-b.out
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

minicpmo-audit: build
	@if [ -z "$(MINICPM_O_GGUF)" ]; then \
		echo "usage: MINICPM_O_GGUF=/path/to/model.gguf make minicpmo-audit"; \
		exit 2; \
	fi
	$(OUT) audit "$(MINICPM_O_GGUF)"

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

chat-fixture: FORCE build fixtures
	printf "hello\n" | $(OUT) chat tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-chat-repl.out
	grep -q "chat>" /tmp/minicpm-o-uya-chat-repl.out
	grep -q "sampled token\[0\]: 4 piece=hello" /tmp/minicpm-o-uya-chat-repl.out
	@if printf "hello\nhello\nhello\n" | $(OUT) chat tests/fixtures/tiny.gguf >/tmp/minicpm-o-uya-chat-overflow.out 2>&1; then \
		echo "expected chat context overflow to fail"; \
		exit 1; \
	else \
		grep -q "error: context overflow" /tmp/minicpm-o-uya-chat-overflow.out; \
	fi

FORCE:
