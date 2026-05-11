UYA ?= /home/winger/uya/uya/bin/uya
SRC := src/main.uya
OUT := build/minicpm-o-uya

.PHONY: build test fixtures inspect-fixture clean

build:
	mkdir -p build
	$(UYA) build $(SRC) -o $(OUT)

test:
	$(UYA) test src/*.uya src/minicpmo/*.uya
	$(MAKE) inspect-fixture

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

clean:
	rm -rf build
