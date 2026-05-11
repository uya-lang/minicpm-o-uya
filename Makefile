UYA ?= /home/winger/uya/uya/bin/uya
SRC := src/main.uya
OUT := build/minicpm-o-uya

.PHONY: build test fixtures inspect-fixture audit-fixture minicpmo-audit clean

build:
	mkdir -p build
	$(UYA) build $(SRC) -o $(OUT)

test:
	$(UYA) test src/*.uya src/minicpmo/*.uya
	$(MAKE) inspect-fixture audit-fixture

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
	grep -q "  vision: 1" /tmp/minicpm-o-uya-audit.out
	grep -q "  audio: 1" /tmp/minicpm-o-uya-audit.out
	grep -q "  speech: 2" /tmp/minicpm-o-uya-audit.out
	grep -q "Qwen3 text metadata: present" /tmp/minicpm-o-uya-audit.out
	grep -q "generation_supported=false" /tmp/minicpm-o-uya-audit.out
	$(OUT) audit tests/fixtures/bad_schema.gguf >/tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: missing tokenizer tokens/model metadata" /tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: missing root token embedding tensor" /tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: unknown tensor dtype" /tmp/minicpm-o-uya-audit-bad.out
	grep -q "unsupported: unclassified tensor branch" /tmp/minicpm-o-uya-audit-bad.out

minicpmo-audit: build
	@if [ -z "$(MINICPM_O_GGUF)" ]; then \
		echo "usage: MINICPM_O_GGUF=/path/to/model.gguf make minicpmo-audit"; \
		exit 2; \
	fi
	$(OUT) audit "$(MINICPM_O_GGUF)"

clean:
	rm -rf build
