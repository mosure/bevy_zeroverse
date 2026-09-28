# Procedural interiors

`procedural-indoor` (also accepted as `procedural_indoor`) is a separate scene mode.
It creates geometry and textures in memory; it needs no MatSynth catalog, GLTF
furniture or environment photographs. Occupied scenes use the bundled AnnyBody
reference (`assets/burn_human`); zero human density omits that model. Cornell, object, room,
semantic room and human modes remain available.

This is a substantial procedural/PBR baseline, **not a demonstrated state-of-the-art
photorealistic renderer**. The implementation and validation below separate what
works from what has not been qualified.

The [v17 exterior/window review](facade_v17.md) adds independently sampled
exterior walls, larger near-floor-to-ceiling glazing, inset frames and shared
aperture checks for finishes and wall-mounted objects, with distribution exports.

The [v16 seating/activity review](seating_v16.md) documents modular upholstery,
chair and bookshelf programs, wall hardware, generated board/TV content, continuous
activity mixtures, playback/grid fixes and bounded native/browser validation.

The [v15 tabletop/material review](tabletop_v15.md) documents new drink vessels,
stationery, phones and microphones, device/clock programs, shared PBR finishes,
and the preceding distribution and rendered-output checks.

## Run and inspect

```sh
cargo run --bin viewer -- --scene-type procedural-indoor --indoor-seed 6
cargo run --bin viewer -- --scene-type procedural-indoor --indoor-seed 3 \
  --indoor-layout conference --indoor-density 0.8

# CPU-only geometry/distribution audit: does not initialize a GPU.
cargo run --no-default-features --features multi_threaded --bin indoor_validate -- \
  --seed 0 --audit-seeds 10000 --output out/indoor_audit

# Actual GPU RGB/depth/normal/semantic/position capture, including validation.
cargo run --no-default-features --features multi_threaded --bin indoor_validate -- \
  --seed 0 --audit-seeds 10000 --renders 8 --cameras 4 --width 800 --height 600 \
  --labels --output out/indoor_final
```

Use `--human-density 0 --asset-root` pointing to an existing empty directory to test
architecture/furniture independence from external assets, and `--rotation-augmentation` to rotate the entire scene and cameras.
Native headless capture uses image targets and needs a working graphics adapter,
but no window or display server. The CPU audit does not require a graphics adapter.

The viewer starts inside the generated room. The existing regeneration control
advances the scene seed. `--indoor-layout` accepts `mixed`, `conference`, `open-office`,
`lounge`, or `training`; density accepts `[0, 1]`. `--indoor-human-density` controls occupancy independently (default 0.25, zero disables people). Invalid density values fail generation.

## Generation contract

The review found three central constraints in the older path: imported objects
were fitted as coarse mesh units, material selection lacked part/surface roles,
and export applied tone mapping plus channel normalization after rendered color
was already tone mapped. Legacy semantic-room dimensions/cameras also covered
spaces much larger than ordinary offices. The new mode addresses those constraints
with metric construction, explicit surface roles, calibrated cameras and separate
color-encoding metadata; it preserves the existing generators for compatibility.

`src/scene/procedural_indoor/` separates layout, geometry, surfaces, object builders,
architecture, runtime integration and validation. `IndoorManifest` is a versioned,
serializable record of the seed, dimensions, layout, styles, lighting, instances,
support relationships, camera paths and optional world rotation.

- Furniture placement uses four activity priors over continuously sampled room
  partitions, spacing, object proportions and coordinated finishes.
  Room dimensions are 5.6–21 m, with logarithmic area/aspect sampling; ceiling height is 2.65–4.8 m.
  The adjoining glazed room is 3.2 m deep.
- Tables have separate tops, aprons, legs and cable details. Chairs have seats,
  thick curved backs, frames, arms and wheeled, cantilever or wood bases. Other
  builders cover sofas, cabinets, bookcases/books, leafy plants and pots, bins,
  whiteboards, displays, monitors, laptops, mugs, notebooks, keyboards, mice,
  bottles, pen holders, clocks, lamps, rugs, printers, storage boxes, coat racks and bags.
- Exterior windows sample one, two or three walls (all seven combinations),
  with continuous bay widths, sill/head heights, pier spacing, frame dimensions
  and inset depths. Tall glazing, horizontal ribbons and stacked bands use real
  wall cutouts. The fourth wall retains the glazed neighboring room.
- Architecture includes real window openings, inset glazing, sills, mullions,
  optional blinds, pilasters, skirting, ceiling trim, acoustic treatments,
  ventilation grilles, light housings and an open doorway. Adjacent furniture and
  exterior buildings are actual geometry.
- Wood, flooring, fabric, paint, foliage, concrete and other surfaces have distinct
  material roles. Generated 256² maps use mipmaps, repeating metric UVs, linear
  normal/roughness data, and sRGB albedo. Wood uses periodic anisotropic noise;
  flooring includes staggered boards, carpet and tile. Mesh parts are batched by
  material and semantic class within each instance.
- Daylight, overcast and evening vary sun direction/intensity, luminaire output,
  exposure and color temperature. Shadowed lights, transmission, a generated
  environment reflection proxy, native SSAO, FXAA and restrained bloom serve RGB.
  RGB postprocessing is removed from annotation passes.
- Solid furniture uses rotated footprint rejection, wall margins and a clear
  doorway approach. Props have explicit supporting surfaces. Cameras use continuous
  swept collision tests against furniture/columns, not just clear endpoints.
  Paths maintain a 0.28 m clearance around nominal object envelopes, including supported props, columns and suspended lights; viewing directions are checked throughout motion.
- Cameras cover seated, low, standing and elevated viewpoints, 0.78–3.25 m heights,
  roughly 28–107° vertical FOV and 0.03–3.0 m requested translation baselines.
  Cubic Bezier paths vary bend, target and roll; the full curve is clearance checked.
  Even single-camera streams span
  height strata across seeds. Camera count does not change furniture or earlier views.

An explicit seed produces a reproducible manifest with the same generator version
and build. Regeneration uses `base_seed + scene_index` with wrapping u64 arithmetic;
changing the configured seed resets that sequence. Independent ChaCha8 streams
separate layout, cameras and surface/object detail. Pixel-identical rendering across
GPUs/drivers is not promised. Keep the manifest, generator version, code revision,
configuration and renderer identity with datasets; split by **scene seed**, not view.

## Dataset use

```python
from torch.utils.data import DataLoader
from bevy_zeroverse_dataloader import BevyZeroverseDataset, chunk_collate

dataset = BevyZeroverseDataset(
    editor=False, headless=True, num_cameras=4, width=640, height=480,
    num_samples=1000, scene_type="procedural_indoor",
    indoor_seed=0, indoor_layout="mixed", indoor_density=0.65, indoor_quality="auto",
    render_modes=["color", "depth", "normal", "semantic", "position"],
    depth_format="linear", playback_steps=1, ovoxel_mode="disabled",
)
loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=chunk_collate)
batch = next(iter(loader))
```

`indoor_manifest` is UTF-8 JSON carried as a ragged uint8 tensor. Use the provided
collator for manifests and variable-length annotations. Indoor Python indexing is
now reproducible: `dataset[i]` requests `base_seed + i`, including repeated indices,
shuffled access and worker changes. Camera count does not change furniture or
previous camera samples. Split datasets by scene seed, not camera or timestep.

The canonical Rust `zeroverse_gen` CLI accepts indoor layout, density, quality,
rotation and codec settings. Indoor RGB defaults to lossless sRGB float32; depth,
normal, position and semantic planes are lossless. Worker count does not change the
scene assigned to an index. Resume validates the generation contract and actual
sample counts, including partial chunks; capture failures stop the export.
Completed finite indoor exports include per-scene counts, CSV records, calibrated
camera distributions and placement heatmaps in `metrics/`.

See [dataset configuration and export contracts](procedural_indoor_dataset.md) for
complete commands, resume semantics and tested Rust/Python interoperability.

For multi-view reconstruction, use the optional [shared-surface camera policy](multiview_cameras.md), with overlap and baseline controls plus rendered-depth qualification.

## Capture and color contracts

The validation CLI writes `distribution.json`, `metrics.json`, `scenes.csv`,
`objects.csv`, `humans.csv`, `cameras.csv`, `placement_heatmaps.svg`, geometry counters and, per
seed, `manifest.json`, `capture.json`, PNG previews and little-endian `.rgba32f`
buffers. `--no-raw` omits the large raw files. Count histograms include zero-count
scenes and separate the main room from its neighbor. Heatmaps measure normalized
object centers and camera path occupancy, not visible instance counts.

`--playback-steps 3` validates the start, midpoint and endpoint of every camera
trajectory against captured transforms/FOV/near/far. Each view records actual
intrinsics, time, semantic pixel counts, RGB signal statistics and annotation
alignment. Run IDs and completion markers prevent mixing stale or partial captures
into an accepted report. `--stratified` selects the first seed in each observed
layout × lighting × floor × furniture cell, with balanced ordering for smaller
budgets; it does not filter by image appearance. It rejects absent/nonfinite buffers, degenerate RGB signal,
invalid semantic palette sizes and inconsistent geometric/camera annotations.

| Output | Meaning |
| --- | --- |
| RGB raw | Bevy tone-mapped **linear** RGB, not scene radiance |
| RGB PNG / new indoor dataset exports | Fixed linear-to-sRGB transfer; no second tone map or image/channel min/max scaling |
| Linear depth | Positive camera z depth in metres; depth PNG is only a `/15 m` visualization |
| Normal | View-space geometric normal encoded as `0.5 * normal + 0.5` |
| Position | World position affine-normalized to the primary-room AABB; decode with `min + value * (max - min)`. Visible context can lie outside `[0, 1]`. |
| Semantic | Existing semantic palette; PNG applies sRGB encoding, raw/tensor colors are linear |
| OBB | Bounds of constructed object geometry with object class; architectural surfaces are semantic meshes |

Glass is treated as the first opaque geometric surface in depth/normal/position/
semantic passes even though RGB sees through it. This policy is intentional and
must be considered when training cross-modal tasks. Labels are not transparent-layer
ground truth or pixel-level object-instance IDs.

The scene AABB annotation and viewer gizmo enclose the primary reconstruction
room with its structural shell, matching the default [O-voxel region](ovoxel_indoor.md).
Exterior backdrops and neighboring rooms remain visible but do not expand this
box. This scope also applies when voxel export is disabled. Position values are
not clamped to the box, so the visible context still decodes correctly.

**Precision:** native indoor dataset capture uses a dedicated geometry pass with
its own depth test and two RGBA32F attachments. World position/linear depth and
view normals/semantic IDs bypass the RGB HDR pipeline. The established depth,
normal, normalized-position and semantic dataset planes are expanded on the CPU
without float16 quantization. `annotation_precision=float32_geometry` records
this path. Interactive annotation display and legacy modes retain the HDR16
path (`float16_hdr`). RGB still uses Bevy's HDR/tonemapping pipeline.

An independent f64 ray/triangle test measured maximum position error of 66 µm
and reprojection error of 0.00218 pixels across 518 samples, including transformed
curved surfaces, motion, visibility and opaque-glass labels. Raster subpixel
precision is distinguished from storage precision. PNG previews remain visual
previews, not the authoritative geometric tensors.

The new `ColorEncoding` metadata prevents repeated RGB conversion when chunks are
loaded and saved again. Missing metadata retains the historical legacy behavior.
Existing scene modes keep their previous color-export convention.

## Validation evidence

The implementation is generator v8; see the [continuous domain review](procedural_domain_v8.md).
The [generator-v7 review](scene_quality_v7.md) records earlier annotation and viewer fixes.
Historical evidence includes the [v3 evaluation](procedural_indoor_review_v3.md),
[diffuse transport](procedural_indoor_gi.md), [human generation](procedural_indoor_humans.md),
and [dataset contract](procedural_indoor_dataset.md). The [v2 evaluation](procedural_indoor_review.md)
and its [qualification](procedural_indoor/qualification.json) are retained as historical evidence.
The checked-in figures and pictures are actual engine captures. The review covers
geometry, materials, shadows, camera motion, dataset export, and a real WebGPU
browser runtime; it explicitly separates correctness from photographic realism.

![Office](procedural_indoor/office_v3.png)
![Conference room](procedural_indoor/conference_v3.png)
![Lounge](procedural_indoor/lounge_v3.png)

Reproduce the broad native audit and measured report:

```sh
cargo run --no-default-features --features multi_threaded --bin indoor_validate -- \
  --seed 0 --audit-seeds 10000 --stratified --cameras 2 \
  --width 320 --height 240 --playback-steps 3 --labels --no-raw \
  --output out/indoor_review
python scripts/indoor_report.py out/indoor_review --figures
```

The report script needs Pillow and matplotlib for `--figures`; JSON summaries use
only Python's standard library. It exports complete first-view contact sheets by
layout, exposure extremes and count/intrinsic/trajectory charts. CPU audit exports
remain available without Python or a GPU. Large stratified render runs use all
observed cells when `--renders` is omitted; `--renders N` caps the sample count.

Regression commands:

```sh
cargo test --lib --no-default-features --features multi_threaded
cargo test --lib --no-default-features --features multi_threaded \
  broad_density_sweep_is_valid -- --ignored --nocapture
cargo test -p bevy_zeroverse_burn --lib --test indoor_roundtrip
cargo test --test procedural_indoor_render --no-default-features \
  --features multi_threaded -- --ignored --nocapture
cargo test --test procedural_indoor_lighting --no-default-features \
  --features multi_threaded -- --ignored --nocapture
cargo test --test procedural_indoor_viewer -- --ignored --nocapture
cargo clippy --workspace --all-targets -- -D warnings
python crates/ffi/python/test_indoor.py
python scripts/test_indoor_report.py
```

Native GPU tests need a working display/adapter. The capture test exercises an
empty asset root, odd image dimensions, rotated scenes, all five modalities, real
OBBs, asset residency and switches back to Cornell/simple rooms. Legacy semantic
room/human modes still use their existing catalog assets.

## WebGPU / Wasm

```sh
cargo build --bin viewer --target wasm32-unknown-unknown --no-default-features --features web
wasm-bindgen --out-dir www/out --target web target/wasm32-unknown-unknown/debug/viewer.wasm
python -m http.server 8765 --directory www
```

Open `http://localhost:8765/?scene_type=procedural-indoor&indoor_seed=6&indoor_quality=auto`.
Use `indoor_quality=portable` for fewer effects. The page checks WebGPU availability
and reports startup failures. Typed URL parameters accept reproducible seeds.

| Profile | PBR / procedural maps | Shadow maps | SSAO | Bloom | Glazing |
| --- | --- | --- | --- | --- | --- |
| Native Auto | yes | sun/spot 2048; point 1024 | yes | yes | refraction/transmission |
| WebGPU Auto | yes | yes, 1024 | omitted | yes | refraction/transmission |
| Portable | yes | disabled; direct sun disabled | disabled | disabled | alpha blend |

These profiles are explicit settings, not automatic quality or frame-rate promises.
Portable still requires the engine's WebGPU/PBR binding limits. Browser support is
for the scene viewer and displayed modes. `image_copiers=true` is rejected on Wasm:
browser dataset capture remains disabled and unqualified. The native asynchronous
transport and float32 geometry capture are not implemented for browser dataset
readback. Native Rust/Python capture remains supported.

See [WebGPU build and qualification](procedural_indoor_web.md) for the tested browser,
launch flags, adapter limits and reproducible runtime test. A successful Wasm build
alone is not runtime evidence.

## Remaining qualification and scope

The architecture grammar has a rectangular building envelope and one adjoining
office, with open, service-core, gallery and divided-suite floor plans.
Material/palette and object variation are broader than the old fitted
mesh path, but this is not yet unrestricted building topology or a complete model
of real indoor clutter. Generator v4 adds four architectural finish families,
six potted botanical forms and support-aware desk accessories; see the
[local generator-v5 scene review](local_scene_quality_review.md). People now use AnnyBody
with procedural garment surfaces; their appearance remains synthetic. Outdoor
trees are not implemented in this scene. The existing human mode remains.

Native Auto uses multi-bounce diffuse transport computed from the generated
triangles on the GPU, with an independent CPU oracle. The filtered probe volume
still approximates spatial/angular variation; it is not full-image path tracing.
Reflections use a proxy environment, and area-light visibility, specular transport,
cloth/skin appearance, complex architecture and sensor models remain limited.
See the [GI contract and controls](procedural_indoor_gi.md).

Layout tests establish sampled validity and coverage. They do not establish a match
to real office distributions or downstream ML benefit. Procedural humans remain
stylized. No comparison against leading synthetic engines or a matched real-image
benchmark establishes a state-of-the-art photographic-realism claim.

Native Vulkan and headed Chromium WebGPU are exercised on the reported adapter.
Mobile GPUs, baseline-limit WebGPU devices, Safari, Windows and macOS remain
unqualified. Browser headless screenshots are not accepted when the GPU canvas is
blank. Native asynchronous dataset capture and native diffuse GI are separate
capabilities from the browser viewer.

## Sustained generation

Readbacks occur only for requested captures. RGB and the packed float32 geometry
attachments share queue-ordered copies, asynchronous mapping and one bounded
staging allocation per attachment. No per-camera blocking GPU wait is used.
Geometry annotations are rendered once per requested camera/time, and immutable
texture generation uses a bounded native CPU pool. The dataset CLI overlaps one
CPU writer batch with the next capture; it has no unbounded work queue.

```sh
cargo run --bin indoor_bench -- --scenes 1000 --warmup-scenes 64 \
  --cameras 3 --width 320 --height 240 --gpu-timings --output out/indoor_bench
# Fixed-scene control separates regeneration behavior from repeated capture.
cargo run --bin indoor_bench -- --scenes 1000 --warmup-scenes 64 \
  --fixed-scene --output out/indoor_bench_fixed
```

The benchmark measures complete dataset views, CPU preparation/capture time,
GPU timestamp spans, RSS and allocator residency, asset/pipeline/entity counts,
and staging bytes. `scripts/indoor_telemetry.py` adds matching-PID GPU memory and
utilization; device-wide activity is never attributed to the generator.
`scripts/indoor_bench_report.py` validates completion and exports JSON plus memory
and throughput plots. PNG/file encoding is excluded from this engine benchmark;
canonical CLI writer tests measure that separate path.

The [continuous capture and physical review](procedural_indoor_review_v6.md)
records the Bevy 0.19 measurements and the hardware-specific streaming-copy
optimization. The direct benchmark commands above inherit the caller's system
copy policy; use the exact environment recorded with a report when reproducing
its timing. Finite indoor CLI jobs additionally bound worker lifetimes to 256
scenes by default; see the [dataset guide](procedural_indoor_dataset.md).

The [v14 appearance qualification](appearance_v14.md) adds fitted grooms, material microstructure, deterministic glass quadrature and matched Cycles evidence. It records measured glass sampling improvements and the remaining photographic-realism limitations.
