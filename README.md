# B70 LLM Controller

B70 LLM Controller (`b70ctl`) is a terminal UI for installing and serving qualified LLM model packs on Intel Arc Pro B70 GPUs.

A working B70 deployment requires the model revision, runtime image, quantization, card count, context limit, serving mode, and launch settings to agree. `b70ctl` packages those tested choices as model packs and exposes only qualified configurations, from model download through launch, status, logs, stop, and cleanup.

It manages the LLMs running on the B70s; it does not configure the GPUs or the host driver stack.

Current release: **1.0.1**.

## What it does

- Installs qualified model packs from the catalog or by local import, and downloads the exact Hugging Face model revisions each pack pins — interrupted downloads resume by reusing completed files.
- Acquires the pack's qualified Docker runtime, verified by digest.
- Offers only the model, card, context, and serving-mode combinations the pack actually qualified.
- Launches serving with read-only model mounts, Local or LAN binding, and a chosen port.
- Manages running servers: status, recent logs, stop.
- Cleans up: delete or replace models, remove orphaned models no installed pack references, uninstall packs, and remove runtimes `b70ctl` acquired that are no longer used.

## Requirements

The qualified host is **Ubuntu 26.04 amd64**. Other Linux distributions may work but have not been qualified by this project.

- One, two, or four Intel Arc Pro B70 GPUs, depending on the selected profile.
- A B70-capable Intel kernel driver and firmware — the qualified host uses `xe` and exposes card and render nodes under `/dev/dri`.
- Docker, with permission for the user running `b70ctl` to use the daemon.
- Access to the B70 render devices so Docker can pass `/dev/dri` into the container.

The serving-side Intel GPU userspace ships inside the pack's container runtime; the host supplies only the kernel driver, firmware, device nodes, and Docker. Driver installation is distribution-specific and not handled by this project.

## Installation

Release builds are distributed as a Debian package with a `SHA256SUMS` checksum file. Download both from the [1.0.1 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/v1.0.1), then verify and install:

```bash
sha256sum -c SHA256SUMS
sudo apt install ./b70-llm-controller_1.0.1_amd64.deb
```

The package installs `b70ctl` to `/usr/bin/b70ctl` and contains only `b70ctl` — not Docker, Intel drivers, model weights, or runtime images. Run `b70ctl --version` to check the installed version.

## Getting started

Run:

```bash
b70ctl
```

Then follow the normal flow:

```text
Model Packs → choose or import a pack → prepare the required checkpoints
Run Model   → choose model, cards, context, and mode
            → choose Local or LAN access and a port
            → Start
```

Run Model remembers the last combination you started and preselects it whenever the installed packs still offer it.

Models are stored in `~/b70ctl-models` by default. Under **Settings** you can change the model directory, the default access mode, and the default port.

## Model packs

A model pack freezes everything a deployment needs to agree:

- the exact model artifacts and revisions;
- the qualified runtime image digest;
- the card/context/mode profiles that passed qualification;
- the launch settings for each profile.

`b70ctl` offers only combinations contained in the pack; it does not invent unsupported ones or guess that a nearby configuration will work.

Each pack pins a qualified container runtime and the exact model revision. The runtime includes the Intel/XPU/vLLM userspace and any model-specific fixes that pack requires; model weights stay separate and are mounted read-only when serving. Checkpoints are already pre-quantized (FP8 or INT4) and are not converted at startup.

The current runtimes are derived from Intel's `llm-scaler` and XPU vLLM work but are modified and independently qualified by this project; they are not official Intel releases.

See [model-packs/README.md](model-packs/README.md) for the pack format and contributor guide.

## Ownership and cleanup

`b70ctl` deletes only models and Docker runtime images it can prove it acquired itself.

- **Runtimes:** images `b70ctl` pulled and digest-verified are recorded as owned and can be removed from the Unused Runtimes screen, but only while no installed pack references them. A Docker image that was already on the machine — even one identical to a `b70ctl` runtime — may be reused but is never recorded as owned and never deleted; remove such images yourself with `docker`. Docker data unrelated to model packs is entirely out of scope, and `b70ctl` never prunes Docker data or force-deletes images.
- **Models:** Controller-managed models can be deleted or replaced from the Models screen. Hugging Face cache contents `b70ctl` did not download are not cleanup targets.

## Building from source

Building requires Go 1.25:

```bash
make build
make test
make vet
sudo make install
```

`make build` writes `dist/b70ctl`; `sudo make install` installs it as `/usr/local/bin/b70ctl`. The Makefile expects Go at `/usr/lib/go-1.25/bin/go` — set `GO=/path/to/go` if yours is elsewhere.

## Building a release

Maintainer workflow for the release artifacts:

```bash
make package        # build dist/b70-llm-controller_<version>_amd64.deb
make release-check  # full validation; release tag verified if present
make release        # additionally requires tag v<VERSION> at HEAD
```

The `VERSION` file is the single version authority. Tags follow `v` + `VERSION` (`v1.0.0`); in Debian package versions a pre-release dash becomes `~`. Both release commands require a clean tracked tree, run the full test, vet, build, and packaging cycle, and write the `.deb` and `SHA256SUMS` to `dist/`. Nothing uploads automatically; publishing is a manual step.

## Included pack

[Qwen3.8-27B](model-packs/Qwen3.8-27B/README.md) is the currently included qualified pack, distributed with this repository. It covers standard and uncensored FP8 and INT4 checkpoints with Base, MTP1, and dFlash2 serving modes where each combination passed qualification.

## Status

The install, model preparation, serving, and uninstall flows are implemented and shipped in the [1.0.1 release](https://github.com/wu1ff/B70-LLM-Controller/releases/tag/v1.0.1).

## License

B70 LLM Controller is available under the [MIT License](LICENSE). Model weights, runtimes, Intel software, and other third-party components keep their own licenses.
