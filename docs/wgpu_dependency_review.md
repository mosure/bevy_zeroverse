# Registry GPU dependencies

Bevy 0.20 and Burn 0.22 now share registry `wgpu` 30.0.1. The workspace has no
wgpu path patches; local builds and published packages use the same GPU sources.
The 29.0.4 measurements below describe an earlier release and do not qualify
30.0.1 performance or unlimited-process memory stability. Keep the bounded worker
lifetime controls until long-run testing qualifies the new stack.

## Historical review: 2026-09-27


The earlier local wgpu patches had measured benefits and were retained at that time.
The crate archive already excludes `third_party` and Cargo removes root patches
from its normalized manifest. Publishing does not ship a private wgpu fork, but
consumers get different memory and upload behavior from this checkout.

The registry baseline is `wgpu`, `wgpu-core`, and `wgpu-hal` **29.0.4**. The root
dependency now requires at least that tested patch release. Bevy 0.19.1 requires
wgpu `^29.0.3`; Burn 0.21's CubeCL backend requires `^29`. Published wgpu 30.0.1
cannot replace this shared dependency without a Bevy/Burn migration. See the
[upstream releases](https://github.com/gfx-rs/wgpu/releases) and the dependency
versions in the packaged lockfile. Neither compatible registry source includes
the local encoder-cache or staging-allocation changes.

## Bounded comparison

The [standalone probe](evidence/wgpu_dependency_audit/probe/src/main.rs) uses a
separate Cargo workspace so its default build actually uses registry crates.
Both versions were compiled from the same probe with the same optimization
settings. Each run uploads four 32 MiB batches, checks all bytes of a 4 MiB GPU
readback, and completes command-encoder waves with peaks of 128, 512, and 1,024.
There are two runs per version/memory hint, with reversed order in the repeat.
All eight readbacks matched. GPU validation remained enabled.

Adapter: NVIDIA RTX PRO 6000 Blackwell, driver 610.43.02, Vulkan on Linux.

| Device memory hint | Registry retained encoders after each wave | Patched retained encoders |
| --- | --- | --- |
| `MemoryUsage` (capture) | 129 / 513 / 1,025 | 65 / 65 / 65 |
| `Performance` (viewer) | 129 / 513 / 1,025 | 129 / 513 / 1,025 |

These are HAL counters after completed submissions, including the queue encoder.
The patched capture cache holds at most 64 idle encoders; active concurrency is
unrestricted. The repeated run reproduced every count. Registry cache retention
still follows historical concurrency peaks, so removing the core patch would
undo this bound. The probe does not isolate the HAL command-pool retirement
change; its earlier profiling evidence remains in the
[patch notes](../third_party/wgpu-hal/ZEROVERSE_PATCH.md).

Ranges below are the two runs' medians, excluding each run's first upload batch:

| Hint | Registry CPU upload time / 32 MiB | Patched CPU upload time / 32 MiB |
| --- | --- | --- |
| `MemoryUsage` | 11.07–29.02 ms | 2.28–2.74 ms |
| `Performance` | 11.07–11.09 ms | 2.31–2.56 ms |

CPU time measures the eight `Queue::write_texture` calls only. Time through GPU
completion did **not** show a consistent improvement: registry per-run medians
were 11.50–30.97 ms and patched medians 12.25–34.07 ms. Host-cached staging reduces
the measured CPU stall; this is not evidence of higher total scene throughput.
Other GPU activity was present. The earlier second-scale viewer stall remains
historical evidence, not a result reproduced by this smaller probe.

Raw observations are the eight JSON files in
[`evidence/wgpu_dependency_audit`](evidence/wgpu_dependency_audit). This is a
bounded mechanism check, not a new rendered-scene, browser, or long-run memory
qualification. Keep the existing dataset-worker lifetime limits even with the
patches. Unlimited-process stability remains unproven.

## Package behavior

The two-crate packaging check succeeded:

```sh
cargo package --registry crates-io -p burn_siglip2 -p bevy_zeroverse \
  --allow-dirty --no-verify --locked
```

The explicit registry is required when packaging these crates together because
SigLIP restricts `package.publish` to crates.io and Zeroverse does not.
The inspected `bevy_zeroverse-0.20.0.crate` contained no vendored files, dependency
paths, `[patch]`, or `[replace]` entries in its normalized dependency manifest.
Its lockfile resolves all three wgpu crates from crates.io at 29.0.4. The optional
owned `burn_siglip2` dependency becomes a registry requirement for 0.1.1.

This was archive construction and dependency resolution with `--no-verify`, not a
new compilation/runtime qualification of the complete published generator. The
package [inspection record](evidence/wgpu_dependency_audit/package.json) records
the archive hash and exact dependency sources. Nothing was uploaded.

Release requirements remain explicit: publish the owned SigLIP crate before its
Zeroverse consumer, version and publish any required local motion-library fixes,
and compile/test the normalized package against registry dependencies. Local
`burn_human` path overrides must not hide unpublished fixes at that step. Do not
advertise the patched checkout's performance or memory behavior for the registry
package. Root patches were already absent from downstream consumers before this
review.

## Reproduce the mechanism check

Run from the repository root; the standalone project has `publish = false` and
uses no model downloads or Bevy build. Its Cargo.lock records registry sources.

```sh
cargo run --manifest-path docs/evidence/wgpu_dependency_audit/probe/Cargo.toml \
  --locked -- memory

cargo run --manifest-path docs/evidence/wgpu_dependency_audit/probe/Cargo.toml \
  --config 'patch.crates-io.wgpu-core.path="third_party/wgpu-core"' \
  --config 'patch.crates-io.wgpu-hal.path="third_party/wgpu-hal"' -- memory
```

Use `performance` in place of `memory` for the viewer's hint. The patched command
updates only the probe's lockfile; restore its registry lockfile before a later
`--locked` registry comparison. No root manifest edits are needed to run either
configuration. Vulkan and a working adapter are required.
