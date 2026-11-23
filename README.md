# B70 LLM Controller

B70 LLM Controller (`b70ctl`) is a terminal UI for installing and serving qualified LLM model packs on Intel Arc Pro B70 GPUs.

It manages tested model/runtime deployments, from checkpoint downloads through launch, status, logs, stop, and cleanup. It does not install or configure host GPU drivers.

Controller version: **1.0.2**.

[Model packs](#available-model-packs) · [Requirements](#requirements) · [Installation](#installation) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Documentation and development](#documentation-and-development) · [License](#license) · [What's cooking](https://github.com/wu1ff/B70-LLM-Controller/discussions/10)

## Available model packs

| Model pack | Checkpoints | Serving modes |
| --- | --- | --- |
| [Qwen3.8-27B](model-packs/Qwen3.8-27B/README.md) | Standard and uncensored FP8 / INT4 | Base, MTP1, dFlash2 |
| [Qwen3.6-35B-A3B](model-packs/Qwen3.6-35B-A3B/README.md) | FP8 / INT4 | Base, DFlash |
| [Qwen3.8 Flash-Next — vLLM DFlash2 (v6)](model-packs/Qwen3.8-Flash-Next-vLLM/) | Intel AutoRound W4A16 | DFlash2 — 7 proposals, 4× B70, 262,144-token context |
| [Qwen3.8 Flash-Next — SGLang DFlash2](model-packs/Qwen3.8-Flash-Next-SGLang/) | Intel AutoRound W4A16 | DFlash2 — 7 proposals, 4× B70, 262,144-token context |

Mode availability depends on the checkpoint, card count, and context. Each linked pack README describes its qualified profiles and any additional storage or runtime requirements.

## Requirements

The qualified host is **Ubuntu 26.04 amd64**. Other Linux distributions may work but have not been qualified by this project.

- One, two, or four Intel Arc Pro B70 GPUs, depending on the selected profile.
- A B70-capable Intel kernel driver and firmware — the qualified host uses `xe` and exposes card and render nodes under `/dev/dri`.
- Docker, with permission for the user running `b70ctl` to use the daemon.
- Access to the B70 render devices so Docker can pass `/dev/dri` into the container.

The serving engine and Intel XPU userspace ship inside the pack's runtime image. The host supplies the kernel driver, firmware, device access, and Docker; driver installation is distribution-specific and outside the Controller's scope.

## Installation

The [Controller releases](https://github.com/wu1ff/B70-LLM-Controller/releases) include an amd64 Debian package and a standalone `b70ctl` executable. Download the desired asset and the `SHA256SUMS` file from the same release, then verify the download before installation.

```bash
sha256sum -c --ignore-missing SHA256SUMS
sudo apt install ./b70-llm-controller_1.0.2_amd64.deb
```

The package installs `b70ctl` to `/usr/bin/b70ctl` and contains only `b70ctl` — not Docker, Intel drivers, model weights, or runtime images. Run `b70ctl --version` to check the installed version.

## Quick start

Launch the terminal UI:

```bash
b70ctl
```

1. Open **Model Packs** to choose a catalog pack or **Import Local Pack**.
2. Select and prepare the required checkpoints. Exact revisions and the qualified runtime are acquired as needed; interrupted model downloads reuse completed files.
3. Open **Run Model** and choose a supported model, card count, context, and serving mode.
4. Choose **Local** or **LAN** access, set a port, and select **Start**.
5. Use **Runtime** to check status, view recent logs, or stop the server.

Run Model remembers the last combination you started and preselects it whenever the installed packs still offer it.

Models are stored in `~/b70ctl-models` by default. Under **Settings** you can change the model directory, default access mode, and default port, and configure a Hugging Face token for gated or private models.

## How it works

A model pack pins exact checkpoint revisions, file inventories, and runtime image digests, together with qualified quantization, card, context, mode, and launch settings. `b70ctl` verifies the runtime's identity and offers only combinations recorded in the pack.

Checkpoints are pre-quantized and mounted read-only for serving; weights stay separate from the runtime image. The project independently qualifies each runtime and its pack-specific fixes. The pack READMEs describe the serving engines and their upstream sources.

Cleanup respects resource ownership:

- **Models:** delete or replace Controller-managed models from **Models**, including orphaned models no installed pack references. External Hugging Face cache contents are not cleanup targets; pack uninstall retains shared artifacts.
- **Runtimes:** only images `b70ctl` pulled, digest-verified, and recorded as owned are eligible for removal when no installed pack references them and they are not active. Pre-existing images may be reused but are never claimed or deleted. **Unused Runtimes** removes eligible images without forcing deletion.
- **Other resources:** server management applies to Controller-managed containers. Unrelated Docker data is outside its scope; `b70ctl` never prunes Docker data or force-deletes images.

## Documentation and development

- [Model-pack format and contributor guide](model-packs/README.md)
- [Building from source](docs/DEVELOPMENT.md#building-from-source)
- [Maintainer release instructions](docs/DEVELOPMENT.md#building-a-release)

## License

B70 LLM Controller is available under the [MIT License](LICENSE). Model weights, runtimes, Intel software, and other third-party components keep their own licenses.
