# Batch-fix dependency upgrade — 2026-09-27

The batch-fix versions below now resolve from crates.io with exact requirements
and verified registry checksums. All six sibling checkout overrides have been
removed. The [published resolution report](evidence/human_registry_upgrade/published_resolution.json)
records availability at 15:52 UTC, the resolved sources and the source comparison.
The earlier [14:09 UTC check](evidence/human_registry_upgrade/registry_visibility.json)
is retained as historical evidence from before publication completed.

| Crate | Required version |
| --- | --- |
| `burn_human` | 0.5.1 |
| `bevy_burn_human` | 0.6.1 |
| `burn_human_motion` | 0.1.1 |
| `burn_human_inference` | 0.1.4 |
| `burn_ardy` | 0.1.4 |
| `burn_llama` | 0.1.2 |

`burn_soma` 0.1.3, `burn_gemx` 0.1.3 and `burn_mhr` 0.1.2 were also verified
published and not yanked. Zeroverse does not use these model crates, so they are
not added to its dependency graph. The companion plugin's studio feature remains
disabled; default zeroverse features still exclude ARDY, Llama and the inference
library.

The lockfile change is limited to registry sources and checksums for the six
packages above. Their 71 source files match the previously validated sibling
checkout, apart from function-signature formatting in `burn_human/src/lib.rs`.
This registry transition does not change motion numerics or capture identity.
The wgpu optimizations reviewed separately are unchanged.

## Registry transition validation

- Motion-enabled library tests: **106 passed**, three existing long qualifications
  ignored ([log](evidence/human_registry_upgrade/published_tests.txt)).
- Default native viewer: compile check passed
  ([log](evidence/human_registry_upgrade/published_native.txt)).
- Wasm viewer with `web,human_motion`: compile check passed
  ([log](evidence/human_registry_upgrade/published_wasm.txt)); browser execution
  was not rerun. Cargo still reports the existing `burn-cubecl` future Rust
  compatibility notice.
- Metadata contains no dependency on the sibling `burn_human` checkout; all six
  lockfile checksums match crates.io. No unrelated lockfile packages changed.
- The source comparison found no behavioral changes, so the existing rendered
  motion review was not rerun for this dependency-source-only transition.

```sh
cargo test -p bevy_zeroverse --lib --features human_motion --locked -j12
cargo check -p bevy_zeroverse --bin viewer --locked -j12
cargo check -p bevy_zeroverse --target wasm32-unknown-unknown \
  --no-default-features --features web,human_motion --bin viewer --locked -j12
```

## Historical local preflight

[Results](evidence/human_registry_upgrade/preflight.json) use the local batch-fix
sources, not registry packages:

- Native library tests: 102 passed, three long qualifications ignored.
- Strict workspace Clippy for all targets and formatting checks passed.
- Motion-enabled Wasm viewer compilation passed. Browser runtime was not rerun.
- The default dependency graph excludes ARDY, Llama and the inference library.
- Native captures at seeds 1 and 13 compared batch sizes one and two. Seed 13
  admitted two actors; all 240 frames' root translations and local rotations
  matched exactly, with no foot-contact differences. Two serial batches became
  one two-actor batch. Each process loaded the model pair once.
- Seed 1 had no feasible motion plans in this generator revision and loaded no
  models. Admission/rejection decisions matched between the two configurations.
- RGB/depth/normal/semantic and pose checks passed across 50 views, including
  an explicit static seed-13 control that loaded no motion models.

This is a small integration check on NVIDIA RTX PRO 6000 Blackwell/Vulkan. It is
not a new photographic-quality, broad distribution or memory-stability claim.
The historical [motion qualification](human_motion_validation.md) used an earlier
local dependency snapshot; its measurements are not a new registry runtime run.

Commands, using the same cached model bundles:

```sh
cargo test -p bevy_zeroverse --lib --features human_motion --locked
cargo clippy --workspace --all-targets --features human_motion --locked -- -D warnings
cargo check -p bevy_zeroverse --target wasm32-unknown-unknown \
  --no-default-features --features web,human_motion --bin viewer --locked
cargo build -p bevy_zeroverse --features human_motion --bin motion_validate --locked
target/debug/motion_validate --seeds 128 --render-seed-list 1,13 \
  --policy '{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":1}' \
  --output out/human_registry_upgrade/preflight_serial
target/debug/motion_validate --seeds 0 --render-seed-list 1,13 \
  --policy '{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":2}' \
  --output out/human_registry_upgrade/preflight_batch2
target/debug/motion_validate --seeds 0 --render-seed-list 13 --static-only \
  --output out/human_registry_upgrade/preflight_static
python3 docs/evidence/human_registry_upgrade/compare.py out/human_registry_upgrade preflight
```

No commits, pushes, CI runs or publication were performed here.
