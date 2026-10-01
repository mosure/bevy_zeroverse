# Indoor engine qualification: generator version 3

> **Archived evaluation.** This report describes its recorded build. Use the
> [documentation index](README.md) for current capabilities and defaults.

> **Historical Bevy 0.17 evidence.** The current source uses Bevy 0.19.1. See the
> [migration review](procedural_indoor_review_bevy019.md) for its recorded qualification
> status and separate runtime artifacts. Generator version 3 is retained, but the
> renderer identity, performance and image evidence are not interchangeable.

This revision adds native float32 geometry annotations, static multi-bounce diffuse
GI, asset-free articulated people, asynchronous capture and bounded export overlap.
It is a validated procedural dataset engine upgrade. The evidence does not establish
photographic realism, a state-of-the-art renderer, or downstream real-data ML benefit.
The [machine-readable qualification](procedural_indoor/qualification_v3.json)
links the individual accuracy, distribution, runtime and export artifacts.

## Geometry, occupancy and scene diversity

The generator preserves the Cornell and simple-room modes. Indoor rooms combine
separate architectural, furniture and human components and materials: inset glazing,
neighboring rooms, exterior views, structural columns, trim, ceiling fixtures,
segmented tables and chairs, shelving, whiteboards, displays, plants and small props.
Four room families, three lighting families and independent finish/furniture choices
provide 108 categorical audit cells. Seeded continuous variation covers dimensions,
furnishing density, placement, materials, camera intrinsics and paths.

The [CPU geometry qualification](procedural_indoor/geometry_qualification_v3.json)
checked 40,000 layouts across four furnishing densities and 160,000 camera paths
without an invalid layout. A separate 4,096-scene full-occupancy sweep checked
42,496 people, 16,384 paths and 474 million actual human mesh vertices. It covered
all five pose families, three outfits, eight skin tones and six hairstyles.
People have stable instance IDs, 21-joint world-space annotations, chair references,
bounded collision envelopes and independent sampling from the furniture RNG.
The [human guide](procedural_indoor_humans.md) describes the representation and limits.

Human content remains visibly procedural. The meshes improve occupancy and pose
diversity; they do not establish realistic skin, hair, clothing or facial appearance.
The topology grammar and asset families also remain much narrower than real offices.
These finite tests are not a proof for every possible seed or configuration.

The [native render sweep](procedural_indoor/render_summary_v3.json) selected the
first observed seed in each of all 108 cells from a 10,000-seed audit, without
image-quality filtering. All 648 views passed at 320×240: two cameras per scene,
each at trajectory start, midpoint and endpoint. Captured poses exactly matched
the generated paths. Worst per-view 99th-percentile depth/position disagreement
was 3.34 micrometres and reprojection disagreement 0.00276 pixels.
People were visible in 87.8% of these views; that is pixel visibility rather
than a scene-level person count. The largest fraction of clipped pixels in a view
was 0.90%; mean linear luminance ranged from 0.0429 to 0.3246.

[Count/intrinsic distributions](procedural_indoor/distributions_v3.svg),
[placement heatmaps](procedural_indoor/placement_heatmaps_v3.svg) and
[raw metric histograms](procedural_indoor/metrics_v3.json) retain population-level
evidence, including zero-count scenes. Complete first-view contact sheets show
[conference](procedural_indoor/contact_conference_v3.png),
[office](procedural_indoor/contact_openoffice_v3.png),
[training](procedural_indoor/contact_training_v3.png) and
[lounge](procedural_indoor/contact_lounge_v3.png) scenes.
The [exposure extremes](procedural_indoor/contact_exposure_extremes_v3.png) are also
included. Visual inspection shows repeated furniture families, recognizable
procedural people and dark ceiling regions; passing signal gates does not remove
these realism limitations.

An additional [odd-size, rotated-camera qualification](procedural_indoor/highres_summary_v3.json)
passed eight scenes and 72 views at 801×601 with three cameras, three trajectory
samples and human density 0.6. Its worst per-view 99th-percentile depth/position
disagreement was also 3.34 micrometres, and reprojection disagreement 0.00272 pixels.

## Lighting and photographic realism

Native Auto uses generated triangle geometry and material albedo to bake a three-bounce
diffuse irradiance volume. The GPU dispatch occurs once per scene before rendering;
the production path never reads the volume back. Directional sunlight, a room-size
photometric fixture grid, shadow maps, procedural PBR textures, exposure-aware
emissive screens and fixture diffusers contribute to the rendered result.

The [GI qualification](procedural_indoor_gi.md) includes analytic sky/enclosure
controls, BVH versus brute-force visibility, volume packing, world rotation, and
actual GPU readback versus a separate CPU transport implementation. At 54 selected
probe/lobe values, relative mean absolute error against a 4,096-ray CPU reference
was 17.01% at the default 256 rays and 8.76% at 1,024 rays. The higher budget is
configurable through native CLI, Burn and Python, and costs approximately twice
the completed generation time in the matched 64-scene comparison.

Fixed-camera intervention tests now assert identical camera matrices. Removing
direct lights produced mean absolute per-pixel linear-luminance differences of
0.04545/0.04330 in the two seed-6 views; removing local shadows produced
0.00774/0.00658. GI alone produced smaller differences of 0.000574/0.001219.
The ceiling remains dark in these examples.
Earlier comparisons with interactive camera motion are explicitly superseded.

GI remains an approximation: finite bounces and rays, coarse spatial probes,
six angular lobes, interpolation leaks, static geometry and diffuse transport.
Reflections use a proxy environment; glossy interreflection, caustics and full
refractive transport are absent. No comparison against a matched full path-traced
image or real photograph has been accepted. Signal thresholds and image variety
must not be presented as photographic realism scores.

## Annotation precision and capture consistency

Native indoor capture renders geometry directly into two RGBA32Float targets:
world XYZ plus camera-Z depth, and encoded view normal plus semantic class ID.
An independent depth attachment selects the first geometric surface. CPU export
preserves float32; it does not recover precision from a float16 color target.
RGB remains the normal HDR raster render. Glazing is opaque for geometric labels.

The [independent ray/triangle test](procedural_indoor/ground_truth_precision_v3.json)
checked 518 reference intersections at odd resolution, with translation,
nonuniform scaling, curved normals, hidden geometry and despawn controls. Maximum
world-position error was 66.1 micrometres, depth error 57.7 micrometres and
reprojection error 0.00218 pixels. None of the sampled depth values were exactly
representable in float16. These errors include raster subpixel discretization; float32
storage does not imply exact analytic ray intersections.

Sampling controls explicit normalized trajectory time, freezes automatic scene
motion, snapshots metadata at request time and restores interactive playback.
Scene identity, camera pose and projection changes during readback fail closed.
This prevents metadata from silently describing a later interactive frame.
The [capture coherence regression](procedural_indoor/capture_coherence_v3.json)
exercises interactive sine playback, scene yaw, three explicit trajectory steps,
native and legacy modes, failure recovery and regeneration. Unsupported mixed
native flow/motion-vector requests fail explicitly rather than emit partial samples.
Legacy sequential render modes retain their original format and functionality.
The dedicated native indoor path rejects unsupported skin/morph geometry.

## Generation scheduling and memory

Capture copies are encoded in the same render graph command stream as their source
passes. GPU map callbacks are polled without a blocking device wait. Each camera
has one bounded staging buffer per attachment and at most one outstanding packet;
packet IDs synchronize the whole requested view set. Native indoor annotations share
one geometry pass per camera rather than cycling four independent label renders.
Geometry uploads are cached only for the current scene and evicted after regeneration.
Generated headless cameras stop redundant RGB rendering after their copies are
submitted. The [native headless polling window](procedural_indoor/capture_poll_window_v3.json)
checks readiness with nonblocking device polls and one-millisecond sleeps, returning
to the normal application schedule on completion or a ten-millisecond deadline.
It applies only after every current copy is encoded and some packet remains pending.
This avoids repeatedly running the full ECS and submitting empty render frames while
the GPU finishes. It does not delay initial submission or affect interactive/Wasm
rendering. `CapturePollBackoff` can disable pacing for matched controls. The deadline
bounds intentional polling/sleep scheduling; OS scheduling or servicing a callback
can extend wall-clock time.

In the historical [single-sleep 128-scene polling control](procedural_indoor/polling_comparison_v3.json),
the one-millisecond backoff reduced whole-command CPU time from 147.51 to 107.98
seconds (26.8%), mean measured-scene updates from 202.95 to 121.21 (40.3%), and
peak RSS from 3,720.8 to 3,441.4 MiB (7.5%). Completed throughput was 5.884 versus
5.856 views/s, a 0.48% decrease in this single pair. Both runs used the same seeds,
16 warmup scenes and GPU timestamp instrumentation. This earlier implementation yielded once per update; the final window above
performs several readiness polls before another ECS update. These historical numbers
are retained separately from the final measurements below.

The final [bounded-window comparison](procedural_indoor/poll_window_comparison_v3.json)
used 128 identical seeds, 16 warmup scenes, cache cleanup enabled in both runs and
GPU timestamp instrumentation disabled. CPU time fell from 144.32 to 66.60 seconds
(53.9%), mean measured-scene updates from 192.13 to 26.53 (86.2%), and peak RSS from
3,275.3 to 2,824.0 MiB (13.8%). Throughput was 5.983 versus 6.002 completed views/s
(+0.32%). This is one matched pair, not a statistical equivalence result. The
[memory and timing curves](procedural_indoor/poll_window_comparison_v3.svg) retain
the transient allocation behavior rather than treating a short run as a plateau.

The [renderer residency regression](procedural_indoor/render_residency_v3.json)
identified seven retained Bevy view/mesh key maps. Cleanup keeps currently extracted
view keys and live main-world Mesh3d entities, including hidden meshes, while retaining
compiled pipelines. Inactive views can re-specialize when activated. Across 64
regenerated scenes and eight fixed-scene captures, cleanup preserved RGB and all four
annotation planes bit-for-bit. Map capacities stayed bounded and obsolete entries
were evicted. These small maps do not explain or bound every process allocation;
whole-process residency is measured separately.

The dataset writer overlaps one encoding batch with one capture batch through a
rendezvous channel. There is no backlog of queued batches. A large batch still
requires substantial RAM; the [dataset guide](procedural_indoor_dataset.md) gives
the memory calculation. Scene preparation and image conversion still require CPU
work. Asynchronous readback is not a claim of full GPU occupancy.

The reproducible `indoor_bench`, `indoor_telemetry.py` and `indoor_bench_report.py`
measure completed RGB plus annotation samples, separate preparation/capture costs,
live glibc allocations, resident memory, bounded staging, render assets, actual GPU
timestamp spans and matching-process NVML memory/utilization. Device-wide activity
from desktop applications is recorded separately. GPU diagnostic frames can arrive
asynchronously; each reported span includes its measured sample count.
NVML process utilization is a mean over reported samples, which can omit idle
intervals; it is not wall-time utilization or occupancy. Memory peaks are observed
polling peaks. Wgpu registry counts measure resource handles and vacant slots,
not driver pool bytes; they complement memory measurements rather than prove a
complete allocation bound.
Measurements use this checkout's optimized development profile (workspace
optimization level 1, dependencies level 3), native Vulkan and the RTX PRO 6000
Blackwell/610.43.02 driver. They are implementation qualification rather than a
release-build throughput ceiling or a cross-GPU benchmark.

Sustained final measurements and interpretation are recorded in the linked
machine-readable qualification. Short runs are not evidence of a memory plateau.
The canonical 128-scene writer test peaked at 1,737 MiB RSS and grew 192 MiB from
warm to tail; this is export/schema evidence, not a long-run memory acceptance.

## Dataset and browser contracts

Portable, native Auto256 and native Auto1024 CLI/Python tests passed worker-count
invariance, partial-chunk resume, incompatible-resume rejection, repeated indices,
two spawned DataLoader workers and exact lossless RGB roundtrips. Odd 161×119
captures used two cameras and three time samples with all five planes. Stable
human/OBB IDs, ragged person counts, class remapping, annotation precision and actual
GI settings survive filesystem/chunk/FFI/Python export. The canonical CLI exports
layout, object, person, camera and placement-distribution metrics.
See the [final-source Auto256 dataset qualification](procedural_indoor/dataset_qualification_v3_final.json)
and the [higher-ray-budget qualification](procedural_indoor/dataset_qualification_v3_auto_1024.json).

The [WebGPU qualification](procedural_indoor_web.md) tested 20 real-browser
mode/profile/seed cases plus regeneration, a camera grid and explicit native-only
dataset-readback rejection. A [final-source browser regression](procedural_indoor/web_residency_v3.json)
repeated Auto/Portable RGB startup and regeneration after the residency changes.
Wasm supports the procedural viewer with deliberate
capability reductions: environment-light fallback in place of the native GI volume,
profile-dependent shadows and legacy HDR annotation visualization. Browser viewing
does not imply native float32 dataset readback support. The tested NVIDIA/Chrome
configuration does not qualify mobile devices, Safari or baseline-limit adapters.

The legacy GPU voxel equality test remains outside this acceptance: it reported
13 versus 43 voxels and skipped strict equality. A passing library suite must not
be interpreted as GPU voxelization parity.

## Reproducing the final checks

The native benchmark excludes PNG/chunk encoding and validates completed RGB and
float32 annotation captures. Its default 1,000-scene population uses seeds 400–1399,
three 320×240 cameras, all five modalities, human density 0.25, Auto256 lighting,
and one normalized trajectory step. The first 64 scenes are excluded from the
following example's throughput statistics; telemetry includes startup and cleanup.
Run controls sequentially with fresh output directories and no other GPU workload:

```sh
cargo build --bin indoor_bench --no-default-features --features multi_threaded
python scripts/indoor_telemetry.py --output out/final_regeneration -- \
  target/debug/indoor_bench --scenes 1000 --warmup-scenes 64 \
  --output out/final_regeneration
python scripts/indoor_telemetry.py --output out/final_fixed -- \
  target/debug/indoor_bench --scenes 1000 --warmup-scenes 64 --fixed-scene \
  --output out/final_fixed
python scripts/indoor_bench_report.py out/final_regeneration out/final_fixed \
  --output out/final_report
```

`--poll-backoff-ms 0` disables the polling window for a matched control;
`--no-cache-pruning` disables the seven-map cleanup. `--gpu-timings` adds optional
GPU query instrumentation and its allocation overhead. Keep instrumentation equal
when comparing runs. Python reporting needs NumPy, Matplotlib and nvidia-ml-py.
The final strict workspace/all-target Clippy check passed. The library suite passed
67 tests with three ignored; targeted ignored GPU tests were run separately as linked
above. The preserved voxel-equality limitation still applies.
