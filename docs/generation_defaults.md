# Efficient full-quality indoor capture

The native indoor path retains shadows, diffuse GI, SSAO, bloom, refraction,
procedural texture resolution, geometry detail and annotation precision. Use
`Auto` quality. `Portable` is an explicit reduction in rendering effects for
constrained adapters; it is not a throughput preset. Headless Portable capture
emits a warning describing those reductions.

Preparation, polling and capture-buffer optimizations are automatic. The current
source-bound [paper timing receipt](../tex/generated/pipeline_receipt.json) records
the qualified workload and measured cost. No additional scheduling controls are
required.

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

SSAA defaults to **Auto** (`ssaa_factor = 0`): 2× below 1280×720 pixels, native
resolution at or above that pixel count. This rule treats portrait and landscape
images consistently. Thus the 256×256 CLI and 512×512 training presets use 2×.
An explicit `config.ssaa_factor = 2` renders capture RGB at twice the width and height,
then area-averages each 2×2 footprint on the GPU into the original output size.
Explicit factors 1 (off), 2, 3 and 4 override Auto. The equivalent CLI flag is
`--ssaa-factor 2`; use `--ssaa-factor 1` to disable it. This increases raster/shading work and GPU attachment memory;
it does not increase final readback bytes or add a CPU filtering/upload step.
The same-device WGPU and optional CUDA tensor paths consume the resolved texture.

RGB is filtered in linear light after tone mapping. Numeric depth, normals,
position, semantic IDs, optical flow and co-visibility retain their exact final
pixel-center rays: they are **not averaged across surfaces**. In particular,
semantic colors cannot become invented classes and depth cannot become a
nonexistent surface at an edge. RGB edge pixels may contain mixed coverage while
their aligned geometric target describes the central ray. Image dimensions,
intrinsics and flow units always refer to the final grid. `Sample::sampling`
records this versioned policy for SSAA captures, including archive round trips;
its absence in older/non-supersampled captures means factor 1. Metadata stores
the actual resolved factor, rather than the Auto sentinel.

For in-memory consumption, construct one `LiveDataset`, retain it across output
shards, and call `next_sample()`. The Burn `Dataset::get()` adapter uses the same
automatic bounded indoor lookahead. This is a consecutive live stream, not an
indexed dataset. A finite `num_samples` drains speculative work at each epoch's
end; zero denotes an unbounded stream. One-sample datasets do no lookahead.
Compatible handles share the persistent renderer and serialize request/response
transactions. Configure the renderer once; incompatible dimensions, modes or
scene settings require another process.

Both public entry points use the same bounded scheduling without requiring a
queue-depth, upload, thread-budget or process-environment setting.

## CLI path

`--scene-type procedural-indoor` resolves to four 256×256 views, one captured
timestep, furnishing density 0.65 and human density 0.25. `--indoor-layout mixed`
(alias `all`) samples all 20 supported room programs uniformly, including kitchens,
teaching, dining, transit and exhibition spaces. `GenConfig::indoor` is the
separate three-view, 512×512 convenience preset described above.

One timestep captures only progress 0. Longer paths and enabled camera trajectories
do not add temporal observations to that sample. Request `--playback-steps` greater
than one to capture more times; human and object motion still require their opt-in
policies. Four cameras at one time are one multi-view room sample with four images,
not four independent rooms. Capacity reports must bind the resolved capture plan.

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

- On Linux GNU x86-64 NVIDIA Vulkan, unchanged CPU light-cluster buffers are
  retained across frames. Exact ordered membership, object classification,
  render-entity mappings, resolved light/probe indices and buffer layout must
  match; no probabilistic hash is used. Changed storage inputs use Bevy's exact
  packing and bounded queue copies, retaining buffers across camera motion.
  A pool retains at most 64 discarded view allocations and 32 MiB of buffer
  capacity across rooms. Every upload clears unused capacity, and recycling
  uses a component hook so render-entity allocation and draw ordering do not
  change. Incomplete light mappings fall back to Bevy's original preparation.
  Light uniforms and probe textures continue updating each frame. The original
  cluster resolution, light ordering, shadows and GI remain unchanged; other
  adapters retain Bevy's normal preparation.
- Geometry, texture construction, mesh realization and both transport trees share
  one bounded native pool, defaulting to at most eight workers. An explicit
  `configure_preparation_threads` budget now covers transport-tree construction
  too; Bevy's other task pools and driver threads remain separate. Ordered
  triangle emission releases the geometry borrow before tree and
  tangent work proceed concurrently. Scoped output order and seed streams are
  unchanged; no full-scene clone or per-room worker pool is introduced.
- Camera coverage queries use a conservative spatial index followed by the same
  exact oriented-box intersection. Probe-tree construction caches exact bounds
  and centroids in compact temporary records, preserves median splits and ties,
  and applies the final triangle permutation in place.
- Reflection probes share immutable light inputs and schedule their cube faces
  together. Transport albedo proxies reuse prepared material evaluators; visual
  GPU material maps, normal/roughness mips and texture sampling remain unchanged.
- On bindless adapters, render-only aliases give each PBR image slot a distinct
  Bevy resource identity. This avoids stranded capacity in Bevy 0.20's material
  allocator when packed maps or emissive screens share an image across slots.
  Aliases reference existing WGPU texture views and samplers with identical
  filtering: they create no texture uploads, copies, shader changes or changes
  to CPU materials. Their
  lifetime follows the source image, including render-world-only images and
  pending uploads. Samplers with identical complete filtering descriptions are shared per slot,
  preserving compact bindless groups. Metadata follows actual render uploads;
  pending and directly replaced GPU images cannot reuse a stale description. Non-bindless adapters use ordinary
  materials.
- Supported headless indoor captures pin scene, time, calibration, target and
  exposure identity before rendering. Current-pose cluster depth is computed
  before light assignment. The render thread copies the first qualified result
  only after current assets, pipelines and every camera attachment are ready.
  The copy group is atomic across cameras. Temporal history effects and unsupported
  policies retain conservative settling. No quality setting is lowered.
- One completed future room may upload immutable images, materials and meshes
  during the current room's readback window. A 256 MiB upload-payload budget and
  one-room slot bound this speculation. Larger rooms use ordinary uploads with
  the same detail. After current copies submit, owned future GI/material work can
  run in separate render slots; a second consecutive successor may occupy a
  GI-only slot within the same byte budget. Future entities, cameras and motion
  inference do not enter the active room. Current-room readiness excludes only
  recorded future asset IDs; promotion restores the complete asset barrier.
  Explicit CPU-only admission holds do not advance the render app or authorize
  additional GPU work.
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
  Bevy's WASM task pool runs on the browser main thread; spawning an async task
  does not create CPU workers. Preparation checks a 4 ms work budget between
  units instead of scheduling a clamped timer for every tiny mesh. Individual
  texture, architecture and human builders remain synchronous and can exceed
  that budget. The exact reflection-filter stencil is reused between rooms
  with bounded process residency (under 24 MiB); its first construction is
  still synchronous. These changes do not establish multithreaded WASM generation.

`indoor_bench` records ready versus unfinished prefetch hits, staged promotions,
and update durations classified by the state before each update. These are
bounded aggregate diagnostics, not exclusive CPU/GPU stage attribution.

## Validation scope

The current source-bound [paper timing receipt](../tex/generated/pipeline_receipt.json)
records the measured preparation and completed-capture workloads, environment,
latencies and memory. Preparation timers distinguish tree emission, probe-tree
construction and shadow-tree construction. Parallel branch durations overlap;
they must not be summed into a fictional total CPU time.

Qualification checks exact seeded manifests, mesh attributes and indices,
material bytes, camera choices, RGB and numeric planes. WGPU/CUDA tests cover
completed consumers, padding, SSAA, glass policies, retained samples, indexed
captures and shutdown. Temporal renders check first-frame lighting and all-view
co-visibility dependencies. Browser checks use actual Chrome WebGPU; native
results do not establish Metal performance.

Matched timing studies and replay captures remain local under ignored `out/`.
Only current-state evidence is installed in the paper and project page. Finite
soaks do not prove unlimited process lifetime, and capture rates do not establish
training speed or downstream learning utility. Keep the renderer alive across
shards and preserve the demand controller's holds, quotas and rejection policy.
