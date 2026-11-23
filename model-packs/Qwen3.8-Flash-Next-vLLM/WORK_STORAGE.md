# v0.33 storage and Controller compatibility

The exact accepted v6 image serves through installed, unchanged b70ctl **1.0.1**.
Production removes the inert external diagnostic extension and its bindings;
the frozen development/qualification profile and all source inputs remain
byte-for-byte intact. The [runtime recipe](RUNTIME_RECIPE.md) and
[compatibility receipt](CONTROLLER_COMPATIBILITY.json) identify every difference.

| Path | Required contents / writer | Production handling |
| --- | --- | --- |
| `/models` | Exact 29-file target snapshot, including original 102,400,512,256-byte PLE shard. | Controller-supported read-only target bind from original NVMe model storage. |
| `/draft` | All six files from pinned BF16 DFlash2 revision. | Controller-supported read-only assistant bind. |
| `/work/tmp` and cache roots | Temporary files and JIT/HF/vLLM caches. | Created in the ordinary writable container filesystem. No work bind, volume or tmpfs. |
| `/work/flashnext-serve.XXXXXXXX/view` | Fresh metadata/frontend files and weight symlinks into `/models`. | Created by accepted entrypoint and preparation code on each boot. No checkpoint or PLE-table copy. |
| `/dev/dri` | Linux Xe GPU render devices. | Original device passthrough; private IPC and 32 GiB shared memory. |

The image runs as root with `ReadonlyRootfs=false`. Cold startup created its own
work directories and reached HTTP 200 readiness in 240.6 seconds. A final live
snapshot held 843 regular work files totaling
74,634,571 bytes; the largest was the
23,778,698-byte serving weight index. These are disposable serving metadata and
caches. PLE stays mapped from the original read-only model shard; it is not
staged into the writable layer. Container recreation loses caches and can add
cold compilation time.

All four worker process maps showed the same original PLE file with read-only
private permissions; inode **134217883** matched the original qualification
checkpoint's PLE shard. The existing discoverable inventory already hardlinked
all 29 target files to that original checkpoint. No links or model files were
created or modified for this task. The first run's distinct Controller checkpoint
copy is retained as separate evidence and is not the final storage proof. Each rank reported
shape `(320001536, 160)`, capacity 1,024, pinned stage 5,242,880 bytes and zero
resident weight elements. Eight native gather threads were active per rank.
Each worker mapped the unchanged residency shim and held GPU render-device FDs;
the shim logged owner residency and peer suppression. No converted table,
compiler installation or campaign work directory is required for serving.

`parity_worker.py` inherits eight methods from `tuning_worker.py` and adds six.
They expose snapshots, A/B switches, synthetic parity probes, explicit timing,
tuning mutations and microbenchmarks through diagnostic collective RPC. Neither
class defines a constructor, startup/destructor hook or ordinary Worker method.
Importing them installs no serving hooks. The worker wrapper only injects these
additional RPC methods. Normal serving does not call them.

Production therefore omits both `/opt/dflash-parity/*.py` binds,
`--worker-extension-cls parity_worker.ParityWorker`, its diagnostic `PYTHONPATH`,
and `VLLM_SERVER_DEV_MODE=1`. Both modules and the original launcher/profile
remain in the preserved source subtree. Built-in GDN metadata, recurrent state,
DFlash2 proposal and STANDARD verification, target/query graph execution, PLE
transport and reasoning EOS guard are independent baked implementations. The
development HTTP RPC/server-info endpoints return 404 in production; ordinary
health, model listing and chat/tool requests succeed.

The installed Controller validated/imported the candidate through its normal
TUI, reused both complete checkpoint snapshots and the accepted local image,
launched all four ranks, showed Running and stopped the owned server cleanly.
Only the target and draft were mounted. No new mounts, Controller patch,
validator exception or binary modification was used. Isolated XDG configuration
pointed to the existing model inventory; production Controller state was preserved.

The [published v0.32 storage investigation](patches/archive/v032-source-delivery/WORK_STORAGE.md)
established the same writable-container principle. This v0.33 worker-dependency
decision was independently traced and served. Unlike v0.32, v0.33 retains
`TMPDIR=/work/tmp` because its accepted entrypoint already creates that directory.
No full-context qualification, MESTABench or BetterBench rerun was performed.
Local serving compatibility is resolved; registry/publication steps remain
separately authorized work.
