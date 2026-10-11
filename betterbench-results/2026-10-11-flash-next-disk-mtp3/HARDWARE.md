# Flash-Next Disk MTP3 benchmark system — 2026-10-11

This record summarizes the completed released-image campaign's inspected
hardware and runtime. It omits hostnames, addresses, serial numbers, UUIDs,
device paths and local filesystem locations. See the [methodology](METHODOLOGY.md)
and [full BetterBench report](results.html).

| Component | Inspected specification |
| --- | --- |
| CPU | AMD EPYC 7443; one socket; 24 physical cores / 48 threads |
| CPU policy / NUMA | `schedutil` governor; four NUMA nodes, six cores / twelve threads each |
| Motherboard | HUANANZHI H12D-8D, revision V2.0 (DMI) |
| Memory | Eight 32 GiB DDR4 DIMMs, 256 GiB installed; about 247.5 GiB usable by Linux |
| GPUs | Four Intel Arc Pro B70; 32 GiB advertised VRAM each; PyTorch reports 32656 MiB per device |
| PCIe | Every host-facing card bridge negotiated 16 GT/s ×16 (PCIe 4.0 ×16) |
| OS / kernel | Ubuntu 26.04 LTS; Linux 7.2.0-070200-generic; x86_64; `xe` GPU driver |
| Checkpoint storage | Samsung MZVLB1T0HBLR-000L2 NVMe, 1,024,209,543,168 bytes; ext4, `rw,relatime` |
| Storage policy | NVMe scheduler `none`; 128 KiB read-ahead |
| Host GPU userspace | Intel compute runtime 26.27.39122.14-1~26.04~ppa1; Level Zero loader 1.32.0-1~26.04~ppa1; IGC 2.38.5-1~26.04 |
| Container GPU userspace | Intel compute runtime 26.27.39122.11-0; Level Zero loader 1.32.0 |
| Serving packages | vLLM 0.30.0+xpu; PyTorch 2.13.0+xpu; Triton XPU 3.7.2; Transformers 5.16.1 |
| Collective / SYCL | oneCCL 2022.0.0; Intel SYCL/unified/compiler runtime and oneMKL SYCL packages 2026.0.0 |
| Containers | Docker 29.7.2; containerd 2.3.3; runc 1.4.3 |
| Benchmark environment | BetterBench v0.6.0; Python 3.14.4; NumPy 2.5.3 |

SMBIOS reports conflicting configured (3200 MT/s) and rated (2667 MT/s)
memory speeds. Actual electrical speed was not independently measured, so
neither is claimed as the operating frequency. PCIe figures refer to the
host-facing bridges; the GPUs' distinct internal endpoint link fields are not
the slot links. No clock, link, governor or firmware settings were changed.

## Runtime identity and configuration

- Released pack: **1.0.3**, image `ghcr.io/wu1ff/qwen38-flashnext-b70:disk`.
- Exact campaign image identity:
  `sha256:b13fed6dd54051395f603409ada440c74afcbd5331810eee3acd62315d7787e3`.
- Checkpoint: `devan-carlin/Qwen3.8-Flash-Next-W4A16`, revision
  `40b8f18df4d4a32cb6e687a51c78207e5e438522`; all 31 release file sizes matched,
  mounted read-only.
- TP4 with expert parallelism, BF16 activations, integrated MTP with exactly
  three speculative tokens and local argmax reduction; 262144 maximum context,
  four sequences, 1024 maximum batched tokens, GPU memory utilization 0.91.
- Original BF16 disk-mapped PLE, required peer-residency shim, native graphs,
  APC/ALIGN, eager/KV-only draft prefill, post-capture warmup, fused speculative
  metadata, device split plans, native tiled reducer, collective temporary
  buffers and 65536-byte small-message all-reduce threshold.

The image identity and configuration matched the released
[pack contract](../../model-packs/Qwen3.8-Flash-Next/pack.json). Implementation
and qualification details remain in the
[runtime recipe](../../model-packs/Qwen3.8-Flash-Next/RUNTIME_RECIPE.md).

## Observation limits

`xpu-smi` could not provide usable telemetry. Read-only `xe` sysfs samples at
15-second intervals observed at most 72 °C per GPU and no thermal-throttle
flags; they cannot exclude unsampled transients. Campaign observations found
no request failures, kernel GPU faults, restarts or preemptions. The existing
server and OS page cache were reused; checkpoint storage-cold behavior and
minimum required RAM were not measured.
