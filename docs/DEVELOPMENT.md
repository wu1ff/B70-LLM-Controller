# Development

Build and release instructions for B70 LLM Controller. See the [main README](../README.md) for installation and use, and the [model-pack guide](../model-packs/README.md) for pack development.

## Building from source

Run these commands from the repository root. Building requires Go 1.25:

```bash
make build
make test
make vet
sudo make install
```

`make build` writes `dist/b70ctl`; `sudo make install` installs it as `/usr/local/bin/b70ctl`. The [Makefile](../Makefile) expects Go at `/usr/lib/go-1.25/bin/go` — set `GO=/path/to/go` if yours is elsewhere, for example `make build GO=/path/to/go`. `PREFIX=/path` overrides the source-install prefix.

`make test` also uses Git and Python 3 with zlib to reconstruct the published Flash-Next archive from its pack release tag. Run the suite in a Git checkout containing that tag. The catalog test verifies the reconstructed release checksum offline, and the development archive test exercises the current pack tree.

## Building a release

Maintainer workflow for the release artifacts, from the repository root:

```bash
make package        # build dist/b70-llm-controller_<version>_amd64.deb
make release-check  # full validation; release tag verified if present
make release        # additionally requires tag v<VERSION> at HEAD
```

The build embeds [VERSION](../VERSION) for display and pack compatibility checks. Release tags prefix that version with `v`. Debian package versions replace a pre-release dash with `~`.

Both release commands require a clean tracked tree; untracked files are ignored. `make release-check` permits an absent release tag, but verifies that it points at HEAD if it exists. `make release` additionally requires the tag at HEAD. A tag pointing at another commit fails either command.

Both commands run `make clean build test vet package`, verify the built and packaged versions, check package metadata and contents, compare the packaged binary with the build, and write and verify `dist/SHA256SUMS`. The clean step removes the existing `dist/` directory. Outputs are the `.deb` and `SHA256SUMS` in `dist/`; `make package` alone does not generate the checksum file.

The Debian package installs `b70ctl` to `/usr/bin/b70ctl` and includes the root README and MIT license. The [packaging script](../scripts/package_deb.sh) uses `VERSION`, converts pre-release dashes to `~`, sets `amd64`, and defaults `SOURCE_DATE_EPOCH` to the current commit timestamp for reproducible package timestamps.

See the [release script](../scripts/release.sh) and [shared release helpers](../scripts/librelease.sh) for the exact validation and version rules. Nothing uploads automatically; publishing is a manual step.
