# Generator 8: broader continuous indoor domains

Generator 8 / `capture-v10` changes scene sampling and pixels. Keep the generator
version, manifest and capture identity with every dataset shard. Generator 7
renders and physical comparisons describe that earlier distribution.

## Sampled programs

Room area and aspect ratio are sampled logarithmically, with metric dimensions
bounded to 5.6–21 m and heights 2.65–4.8 m. Recursive partitions target 14–140 m²
zones with at most twelve leaves, constrained by minimum usable dimensions and
unblocked portals. Occupancy, aisle width, work-surface dimensions, seating pitch
and alignment vary within zones. The enclosing shell and partitions are still
orthogonal; this is not an arbitrary building generator.

Façade bay spacing, opaque pier fraction, sill/head heights, blind coverage and
slat tilt vary. Suspended ceiling rafts follow functional zones with continuously
sampled coverage and drop. Fixtures hang below those rafts. Fixture dimensions
now affect the actual rendered geometry; previously the manifest sampled dimensions
while the builder still selected one of three fixed sizes.

Wall displays, artwork, clocks and whiteboards have sampled sizes and positions
with overlap rejection. Clutter ranges from sparse to occupied surfaces, retaining
support and footprint constraints. Printers, storage boxes, coat racks and bags
join the existing furniture, electronics, books, plants and utensils. The broader
room range also required collision checks in neighboring rooms, dimensioned
whiteboard writing and explicit floor contacts for coat racks.

Material recipes now compose tileable stripes, mineral flecks, veins, grain and
weave, with independently sampled scales, phase, strength, pigment and finish.
Some painted walls use coordinated chromatic finishes. Material and texture
budgets remain bounded per scene; this adds procedural signals, not asset downloads.

## Lighting and cameras

Solar illuminance spans **0.1–100,000 lux** and the electric lighting design target
spans **3–1,000 lux**. Each uses a continuous mixture: solar intensity draws 65%
from log-uniform 3,000–100,000 lux and 35% from the full range; electric intensity
draws 75% from log-uniform 150–900 lux and 25% from the full range. The two draws
are independent. This concentrates probability on occupied/daylit conditions
while retaining low-light tails and their combinations. The electric value is
a lumen-method design target, not a measured work-plane illuminance. Fixture
circuits vary activation and dimming; individual CCTs vary around a continuously
sampled room temperature. The old 900-lumen fixture minimum has been removed.
Sky radiance, environment strength and exposure are serialized, shared by runtime
and reference tools. Exposure partially adapts to illumination and has an
independent offset, preserving useful dark-to-bright differences. Portable ambient
lighting now follows the sampled illumination instead of imposing a constant floor.

Requested camera baselines span 0.03–3 m, vertical FOV roughly 28–107 degrees,
and heights 0.78–3.25 m subject to ceiling and collision clearance. Curve bend,
vertical displacement, aim and roll vary. Accepted distributions are conditioned
on full-path clearance; requested ranges are not a claim of uniform output.
Projection remains a centered pinhole, without sensor noise or lens distortion.

The first rendered review used log-uniform intensity over each full range. It
produced too many dim scenes. The final mixture above changes the population
balance; the rendered validation cohort deliberately balances strata and therefore
overrepresents rare lighting conditions compared with unselected seeds.

RGB validation retains low-light samples instead of rejecting them by the old
normally-lit-office brightness threshold. It still rejects lost signal and nearly
fully clipped images, and exports darkness, contrast and clipping statistics.
Depth, normals and semantics retain their independent geometric checks.

## Coverage and scale

Metrics schema 5 includes the four additional object kinds with zero-count bins,
room area/aspect, photometry/exposure, circuit activation, façade and blind settings,
ceiling drop, clutter, and texture-layer strengths. Joint histograms cover solar
versus electric intensity and room area versus clutter, alongside the existing
layout/furniture and material checks. Render selection balances lighting decades
as well as activity and architecture strata.

Ten million different seeds do not prove ten million perceptually different or
useful training examples. Geometry occupancy sketches, joint histograms, rendered
tails and downstream learning evaluation are separate checks. This upgrade does
not establish 10M-sample learning utility, photographic realism or unlimited-process
memory stability. Matched generator-8 Cycles comparisons remain unperformed.

## Local validation, 2026-09-26

This evidence was collected locally before release, based on
`f81f1630e79c825fbe67b13850779b3c2184f76e` plus the then-uncommitted changes.
Release preparation and remote CI are separate from this recorded evaluation;
see the [0.20 release notes](release_0_20.md).

- 85 library tests passed, with three explicitly ignored tests. The active tests
  include GPU/CPU voxel agreement, topology, people support, texture continuity,
  light dimming and joint domain coverage. Twelve Python reference-tool tests passed.
- Native build, strict Clippy and the Wasm viewer build passed. Formatting and
  whitespace checks passed.
- Seeds 0–99,999, furniture density 0.65, human density 0.25 and two cameras per
  scene: **zero invalid manifests/placements/camera sweeps**. The initial CPU audit
  phase took 22.44 seconds; metric export and mesh diagnostics are additional work.
- Twelve scenes balanced across strata, two cameras and two times per scene:
  **48 native 800×600 views**, all capture/annotation gates passed. This is a small
  rendered cohort, not a 100,000-scene render qualification.
- Eight Chrome WebGPU cases passed: RGB/depth/normal/semantic in Auto and Portable,
  with the editor enabled and regeneration tested. No WebGPU validation errors.
  The tested Wasm SHA-256 is
  `b894769a4b1d23d3dd23fcdbefb224a915efbf23808ff2ec04533f90137ba360`.

Observed population ranges (the percentiles describe accepted scenes, not requested
sampler bounds):

| Quantity | Minimum–maximum | Median | 5th–95th percentile |
| --- | ---: | ---: | ---: |
| Room area, m² | 38.00–269.99 | 100.67 | 43.34–241.52 |
| Room height, m | 2.65–4.80 | 3.73 | 2.76–4.69 |
| Functional zones | 1–12 | 2 | 1–6 |
| Main-room instances | 5–173 | 29 | 14–71 |
| Main-room chairs | 0–26 | 4 | 1–11 |
| Vertical FOV, degrees | 28.07–107.00 | 60.32 | 30.43–102.32 |
| Camera height, m | 0.78–3.25 | 1.52 | 0.85–2.63 |
| Camera path length, m | 0.03–4.09 | 0.25 | 0.07–1.60 |
| Direct sun, lux | 0.10–99,991.10 | 9,380.31 | 0.73–78,699.25 |
| Electric design target, lux | 3.00–999.90 | 305.89 | 9.38–814.31 |
| Exposure, EV100 | 2.19–8.99 | 6.73 | 5.13–8.15 |

All 224 reachable cells of the exported 16×16 solar/electric log-intensity
histogram are populated; its bottom two rows lie below the 3-lux electric bound.
The coarse occupancy sketch estimates 99,716 distinct signatures from 100,000
scenes. This is an approximate layout diagnostic, not an exact uniqueness count
or a measure of perceptual novelty. All 58 main/neighbor count distributions
include zero-count scenes. For example, main-room printers occur in 3.89% of
scenes and storage boxes in 58.99%; added kinds are not uniformly frequent.

In the rendered cohort, mean linear luminance spans 0.000614–0.2184 (median
0.0888), and the darkest view has 99.8% of pixels below the dark threshold.
These deliberately selected tails should not be interpreted as the population's
brightness distribution. Brightness gates establish retained signal, not learning
utility; consumers can filter/reweight using the exported values. Maximum clipped
pixel fraction is 0.391%. Worst per-view p99 depth/position disagreement is
0.00000334 m and p99 reprojection disagreement is 0.002784 pixels; captured camera
poses match exactly. Native geometric annotations use direct RGBA32Float.

Capture wall time was 1.01–3.08 seconds per four-view scene (mean 1.72), including
generation, warmup and IO on an NVIDIA RTX PRO 6000 Blackwell / Vulkan. This is
not a steady-state throughput or long-run memory benchmark.

The contact sheet and exposure extremes were visually inspected, as were browser
RGB outputs. Architecture and illumination vary substantially, but people/hair
and some surface finishes remain visibly synthetic. Indirect illumination and
screen-space glass still have approximations. These images do not establish
photographic realism.

Retained evidence:

- [All 48 rendered views](procedural_indoor/contact_generator8.jpg)
- [Population audit](procedural_indoor/distribution_program8.json)
- [Counts, camera/material distributions and joint histograms](procedural_indoor/metrics_program8.json)
- [Placement and camera-path heatmaps](procedural_indoor/placement_program8.svg)
- [Native render and annotation metrics](procedural_indoor/render_summary_program8.json)
- [Browser cases and tested artifact](procedural_indoor/web_program8.json)

Raw per-instance/per-camera CSVs and individual rendered images remain in
`out/domain8_population_final` and `out/domain8_final` (ignored local outputs).

## Reproduction

Run the population audit separately from the stratified rendered cohort. The audit
checks manifests, placements and swept camera clearance without building every
scene mesh or rendering every seed.

```sh
cargo test --lib -j2
cargo clippy --lib --bins --tests -j2 -- -D warnings
cargo build --bin indoor_validate --bin viewer -j2
target/debug/indoor_validate --audit-seeds 100000 --renders 0 \
  --cameras 2 --human-density 0.25 --output out/domain8_population_final
env -u DISPLAY -u WAYLAND_DISPLAY WGPU_BACKEND=vulkan \
  BEVY_ASSET_ROOT="$PWD" target/debug/indoor_validate \
  --audit-seeds 4096 --renders 12 --stratified --cameras 2 \
  --playback-steps 2 --width 800 --height 600 --human-density 0.25 \
  --labels --no-raw --output out/domain8_final
python scripts/indoor_report.py out/domain8_final --figures
cargo build --target wasm32-unknown-unknown --no-default-features \
  --features web --bin viewer -j2
```

The report tool needs NumPy, Pillow and Matplotlib. Browser validation uses
`scripts/validate_indoor_web.py`, Playwright and Chrome against a local
wasm-bindgen site, with `--profiles auto portable --modes Color Depth Normal
Semantic --generator-version 8 --human-density 0.25 --editor --regenerate`.
Browser checks cover the displayed modalities and regeneration, not browser
dataset readback or a multi-device compatibility matrix.
