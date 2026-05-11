UYA ?= /home/winger/uya/uya/bin/uya
SRC := src/main.uya
OUT := build/minicpm-o-uya

.PHONY: build test fixtures inspect-fixture audit-fixture tokenizer-fixture tensor-fixture kernels-fixture quant-fixture qwen3-fixture generate-fixture vision-fixture chat-fixture minicpmo-audit clean FORCE

build:
	mkdir -p build
	$(UYA) build $(SRC) -o $(OUT)

test:
	$(UYA) test src/*.uya src/minicpmo/*.uya
	$(MAKE) inspect-fixture audit-fixture tokenizer-fixture tensor-fixture kernels-fixture quant-fixture qwen3-fixture generate-fixture vision-fixture chat-fixture

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
	grep -q "  audio: 1" /tmp/minicpm-o-uya-audit.out
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
	grep -q "tensor table: count=32" /tmp/minicpm-o-uya-tensors.out
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
	$(OUT) vision-smoke tests/fixtures/tiny.gguf tests/fixtures/tiny_image.raw >/tmp/minicpm-o-uya-vision.out
	grep -q "vision-smoke: PASS" /tmp/minicpm-o-uya-vision.out
	grep -q "placeholders=1 span=1" /tmp/minicpm-o-uya-vision.out
	grep -q "vision embedding checksum: 0xbab3d2ad" /tmp/minicpm-o-uya-vision.out
	grep -q "diff_l1=" /tmp/minicpm-o-uya-vision.out

minicpmo-audit: build
	@if [ -z "$(MINICPM_O_GGUF)" ]; then \
		echo "usage: MINICPM_O_GGUF=/path/to/model.gguf make minicpmo-audit"; \
		exit 2; \
	fi
	$(OUT) audit "$(MINICPM_O_GGUF)"

clean:
	rm -rf build

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
