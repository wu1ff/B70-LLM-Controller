GO ?= /usr/lib/go-1.25/bin/go
PREFIX ?= /usr/local
VERSION := $(shell tr -d '\n' < VERSION)
BINARY := dist/b70ctl

.PHONY: build test vet install package release-check release test-release clean

build:
	mkdir -p dist
	$(GO) build -ldflags "-X main.version=$(VERSION)" -o $(BINARY) ./cmd/b70ctl

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

install: build
	install -d $(PREFIX)/bin
	install -m 0755 $(BINARY) $(PREFIX)/bin/b70ctl

package: build
	./scripts/package_deb.sh

release-check:
	./scripts/release.sh

release:
	./scripts/release.sh --strict

test-release: package
	./scripts/release_test.sh

clean:
	rm -rf dist
