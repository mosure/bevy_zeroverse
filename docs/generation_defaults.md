# Efficient full-quality indoor capture

The native indoor path retains shadows, diffuse GI, SSAO, bloom, refraction,
procedural texture resolution, geometry detail and annotation precision. Use
`Auto` quality. `Portable` is an explicit reduction in rendering effects for
constrained adapters; it is not a throughput preset. Headless Portable capture
emits a warning describing those reductions.

## Library path

```rust,no_run
use bevy_zeroverse_burn::generator::{GenConfig, run_chunk_generation};

fn main() -> anyhow::Result<()> {
    let mut config = GenConfig::indoor("out/rooms", 128);
    config.seed = Some(200);
    run_chunk_generation(config)
}
```

`GenConfig::indoor` supplies three 512×512 views, one timestep, lossless color,
depth, normals, position, semantics and co-visibility, with four samples per
output chunk. It retains the default human population, 256 GI rays per probe,
and all native Auto effects. Content, camera and annotation requirements can be
changed independently of scheduling. Optional motion models are not loaded
unless requested. O-voxel remains opt-in.

For in-memory consumption, construct one `LiveDataset`, retain it across output
shards, and call `next_sample()`. The Burn `Dataset::get()` adapter uses the same
automatic bounded indoor lookahead. This is a consecutive live stream, not an
indexed dataset. A finite `num_samples` drains speculative work at each epoch's
end; zero denotes an unbounded stream. One-sample datasets do no lookahead.
Compatible handles share the persistent renderer and serialize request/response
transactions. Configure the renderer once; incompatible dimensions, modes or
scene settings require another process.

The public stream previously disabled lookahead, making ordinary library usage
slower than the CLI. It now uses the same bounded scheduling without requiring a
queue-depth, upload, thread-budget or process-environment setting.

## CLI path

```sh
cargo run --release -p bevy_zeroverse_burn --bin zeroverse_gen -- \
  --scene-type procedural-indoor --output out/rooms --samples 128 --seed 200 \
  --width 512 --height 512 --cameras 3 --playback-steps 1 \
  --render-modes color depth normal position semantic co-visibility --no-ui
```

Indoor CLI output uses lossless color and four-sample chunks by default. Existing
lookahead overrides remain accepted for diagnostic compatibility but are hidden
from ordinary help. There are no new scheduling controls to tune. Owned worker
lifetimes remain bounded by the existing process policy; this work does not
establish unlimited-process memory stability.

## Internal scheduling and uploads

- Geometry, texture construction and mesh realization share four native workers
  instead of starting three independent four-worker pools. Seed streams and
  scoped output order are unchanged. The recursively scoped GI builder remains
  separate.
- One completed future room may upload immutable images, materials and meshes
  during the current room's readback window. A 256 MiB upload-payload budget and
  one-room slot bound this speculation. Larger rooms use ordinary uploads with
  the same detail. Future entities, GI dispatches and motion inference are not
  activated. Current-room readiness excludes only the recorded future asset IDs;
  promotion restores the complete asset barrier.
- On Linux GNU x86-64 NVIDIA Vulkan capture, initial supported color textures use
  explicit staging copies through published Bevy/WGPU APIs. Fixed-size write-only
  copies avoid libc's large REP MOVSB path into uncached upload memory. The same
  bounded copy is used by Zeroverse's native transport/annotation buffer uploads.
  No HAL fork, unsafe pointer access, CPU-feature override or environment setting
  is needed for this path. Mip levels, layers, sampler settings and bytes are
  preserved. Image updates, resizing, unsupported formats and explicit Bevy
  upload budgets retain Bevy's ordinary preparation path.
- Other native platforms retain their existing image uploads. Browser
  construction remains cooperative and does not stage future GPU assets.

`indoor_bench` records ready versus unfinished prefetch hits, staged promotions,
and update durations classified by the state before each update. These are
bounded aggregate diagnostics, not exclusive CPU/GPU stage attribution.

## Validation scope

The [qualification receipt](evidence/throughput_defaults/receipt.json) records
two counterbalanced comparisons, each with four 32-room runs and four warmup
rooms per run. All captures use three 512×512 views and the six modes above.
Rates pool completed views over measured capture time.

| Scheduling comparison | Control views/s | Current views/s | Gain |
| --- | ---: | ---: | ---: |
| Existing prefetched capture path | 9.12 | 10.07 | 10.3% |
| Former public API's unprefetched scheduling | 4.46 | 10.18 | 2.28× |

The second control uses `indoor_bench --no-prefetch` to reproduce the former
`Dataset::get()` scheduling policy. It is a capture benchmark, not an end-to-end
Burn training or archive-writing benchmark. Separate channel tests verify that
both public entry points now request automatic lookahead and retain response
ownership across concurrent callers.

Against the already-prefetched control, full-command CPU time is 1.56 versus
1.50 CPU-seconds/room, and peak RSS is 3,614 versus 3,644 MiB. Against the
unprefetched control, CPU time falls from 2.46 to 1.52 CPU-seconds/room; peak RSS
rises from 3,314 to 3,495 MiB because future rooms remain resident. CPU timings
include startup, warmup and shutdown. These finite runs do not qualify long-run
memory growth or isolate GPU utilization from other processes.

For the scheduling change, all **240 saved files across 12 rooms / 36 views /
9,437,184 view pixels** were byte-identical to its original renderer: RGB, depth, normals, position,
semantics, co-visibility, manifests and camera/scene metadata. Across 32 distinct
seeds, human counts, mesh triangles/vertices/instances, GI probes and relocated
probes, rays, bounces and transport/texture bytes also match. Future work drains
at the end of each finite run; no seed is skipped.

The subsequent [material quality pass](material_quality.md) deliberately updates
RGB under a new capture identity. It separately verifies unchanged annotations
and reports current full-quality throughput; the RGB replay claim above applies
to the scheduling qualification, not to the later appearance change.

Validation passed: 192 core unit tests (three existing ignored tests), 15 Burn
unit tests, two final upload packing/copy tests, 10 CLI tests, the native GPU
capture-coherence test, strict Clippy for core/Burn across all targets, and the
WebGPU viewer build with human motion enabled. The Wasm compiler reports an
existing upstream `burn-cubecl` future-compatibility advisory. Browser runtime,
motion-inference throughput and unlimited-process stability were not measured.

Performance and replay measurements for this change use published WGPU 29.0.4
in an external consumer workspace. The root checkout's pre-existing WGPU patches
are not inherited. The active downstream production pool shares this machine;
repeated timings describe that concurrent-load environment and do not establish
isolated adapter throughput or a gain for the differently configured downstream
Portable workload. No production job or downstream quality setting was changed.
