# Model packs

A model pack is the information `b70ctl` needs to install and run a particular family of models on B70 GPUs. It pins the model revisions and runtime image, defines the available modes, and records the card and context combinations that were tested.

The short version is:

```text
runtime + model + mode + profile = launch
```

Each part owns a different piece of that launch.

### Runtime

The Docker image and settings shared by the pack, including its exact image identity, container port, health path, environment, and common command.

### Model

An exact Hugging Face repository and revision, its expected file inventory and container mount, and any launch arguments specific to that checkpoint. A model can be a served target or a support artifact used by a mode.

### Mode

Base, MTP, dFlash-style, or other behavior that changes how a model is served. A mode can add launch settings and require support artifacts.

### Profile

One tested `model + card count + context + mode` combination. A profile selects its runtime and carries only the settings unique to that lane.

Profiles are qualification data, not standalone launch scripts. Settings shared by many profiles belong in the runtime, model, or mode blocks instead of being copied into dozens of commands.

## No inferred support

`b70ctl` only offers combinations present in the pack. It does not calculate context ceilings, assume tensor-parallel support, enable speculative decoding on its own, or expose a combination because a similar one worked.

If it was not tested and recorded as a profile, it does not appear in the UI.

## Repository directory layout

```text
model-packs/
├── README.md
├── index.json
└── <model>/
    ├── README.md
    └── pack.json
```

- `index.json` is the catalog read by **Browse Available Packs**. Each entry names a published versioned archive and its SHA-256. Entries are added only when an archive has actually been published somewhere it can be downloaded; the Qwen3.8-27B archive is distributed as a GitHub release asset.
- Pack archives (`archives/*.tar.gz`) are generated distribution artifacts built from a model directory with normalized tar metadata. They are not committed to this repository.
- Each model directory contains a human-readable description and the machine-readable pack.

Packs are qualified before they are copied into this repository's pack directory. The build or test setup used to produce one is not part of the pack.

## Making a pack

You can use any build and test workflow you like. What matters is that the finished pack is reproducible and describes only configurations you verified.

1. Build a runtime that works on B70.
2. Pin the exact model revisions.
3. Test the card, context, and mode combinations you intend to include.
4. Freeze each model's expected file inventory and the runtime identity.
5. Put shared launch behavior in the runtime, model, and mode blocks.
6. Add only tested combinations as profiles.
7. Check that every profile resolves to the command that passed your testing.
8. Import the finished pack with `b70ctl` and test it there.

The current [Qwen3.8-27B pack](Qwen3.8-27B/pack.json) is a useful schema version 1 example. `b70ctl` validates required fields, references, mount conflicts, duplicate settings, runtime identity fields, and profile uniqueness when it loads a pack.

At minimum, a finished pack should have:

- an exact repository and revision for every model or support artifact;
- the expected file inventory for each revision;
- an immutable runtime image ID;
- an immutable registry reference if `b70ctl` is expected to acquire the runtime;
- only qualified profiles, with no inferred combinations; and
- shared launch behavior factored out of individual profiles.

Keep a pack reproducible, document what you tested, and make sure `b70ctl` can validate it. Inclusion in this repository does not make the project author a guarantor of third-party model weights or runtimes.
