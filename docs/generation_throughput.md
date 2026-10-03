# Procedural capture throughput

For the automatic library/CLI path and the published-WGPU upload improvements,
see [efficient full-quality capture](generation_defaults.md). The measurements
below are the October 2 checkout benchmark, including its stated local WGPU
patches; they are retained as a separate workload and implementation snapshot.

Measured October 2, 2026 on an NVIDIA RTX PRO 6000 Blackwell workstation
(Vulkan, driver 610.43.02, 24 logical CPU cores). The workload uses three
512×512 views per room, one timestep, consecutive seeds starting at 200,
mixed indoor layouts, furniture density 0.65 and human density 0.25.
Auto quality, shadows, SSAO, all geometry/material detail, and diffuse GI with
256 rays/probe and three bounces remain enabled. Outputs are color, depth,
normals, position, semantics and co-visibility.

The current optimization improves matched capture throughput **1.62×** over the
previous one-room-lookahead/surface-area-BVH implementation. The requested
additional **2× is not achieved**. The [qualification receipt](evidence/throughput_512/pipeline.json)
records commands, source identities, artifact hashes, timing and pixel comparisons.
These are optimized dev builds with the checkout's existing WGPU patches;
registry WGPU performance was not measured.

## Capture and complete exports

Capture-only runs use 128 rooms, excluding eight warmup rooms from rates.
Timing includes construction and completed GPU readback, excluding file encoding,
validation and JSONL writes. Both runs retain exactly the same population sampler;
the presenter-placement correction was applied separately to avoid changing the
workload during the comparison.

| Implementation | Views/s | Median room, s | p95 room, s | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Previous implementation | 6.14 | 0.356 | 1.071 | 3,507 |
| Current pipeline | **9.97** | **0.239** | **0.433** | 3,980 |

CPU time for the whole command falls from 251.5 to 177.9 seconds. The deeper CPU
queue costs about 473 MiB at the observed peak. RSS rises across both finite,
varied-room runs; these measurements do not establish unlimited-process stability.

Complete CLI runs export 64 rooms / 192 views / 16 chunks. All six modes, float32
geometric annotations, lossless raw-sRGB tensors, Zstd, chunk size four and dataset
metrics are enabled. Wall time includes startup, export and shutdown.

| Implementation | Processes | Wall time, s | Views/s | CPU seconds/room |
| --- | ---: | ---: | ---: | ---: |
| Previous implementation | 1 | 39.56 | 4.85 | 2.25 |
| Current pipeline | 1 | **27.36** | **7.02** | 1.73 |
| Current pipeline | 2 | **22.19** | **8.65** | 2.08 |
| Current pipeline | 4 | 22.95 | 8.37 | 2.86 |

The matched single-process export gain is 44.6%. Two processes are fastest among
these tested settings; four add CPU cost without improving this workload.
The portable default remains one. Choose process count and lookahead depth for
the actual hardware, resolution and annotation set. Command exit is observed at
200 ms intervals. NVML activity includes desktop work and is not kernel occupancy.

```sh
cargo run --locked -p bevy_zeroverse_burn --bin zeroverse_gen -- \
  --scene-type procedural-indoor --output out/indoor_512 \
  --samples 64 --seed 200 --workers 2 --chunk-size 4 \
  --indoor-prefetch-depth 3 \
  --width 512 --height 512 --cameras 3 --playback-steps 1 \
  --render-modes color depth normal semantic position co-visibility \
  --color-codec raw --compression zstd --ov-mode disabled --no-ui
```

## Implementation and correctness

- A bounded queue prepares up to three future CPU rooms by default, with a hard
  limit of four. It validates every construction input before promotion and
  drains at the end of a finite run. No future GPU upload, ECS installation or
  motion inference occurs before that scene is requested. All 127 transitions
  hit the queue and no extra room is started after the final request.
- Independent geometry builders run on a bounded native pool while retaining
  deterministic object and surface order. Indexed vertices are transformed once
  for light transport. A 256-entry lookup preserves the original byte-sRGB
  transfer exactly, removing repeated powers during texture mip construction.
- Headless capture retains one allocation set per configured camera, keyed by
  resolution and modality. Only consumed, successful readback storage is reusable.
  New cameras get fresh frame/flow/co-visibility status; active or failed maps
  allocate fresh storage. Changing camera count, resolution or modalities trims
  or replaces the pool. Interactive cameras do not use it.
- Co-visibility reuses one atlas across all 128 rooms, compared with 128 allocations
  in the control. Each dispatch checks the actual GPU texture identities.
  Asset readiness is bound to the current scene root. Original settling delays,
  motion readiness, shader readiness and GPU-copy completion checks remain.
- Every room preserves triangle counts, probe counts, relocated-probe counts,
  texture/transport bytes, ray budgets and bounce counts. The original CPU probe
  classification tree and independent lighting oracle remain unchanged.
- The matched single-process exports are **bit-exact for RGB and all non-color
  tensors** across 50,331,648 pixels: depth, normals, position, semantics,
  co-visibility, manifests, calibration, poses and bounding boxes. Runtime
  diagnostic metadata is excluded. Every capture passes readiness, finite-value,
  dimensions, seed-order and co-visibility membership checks.
- Changing process count can alter rasterization at rare pixels and reorder
  unindexed fixture boxes. Both parallel cases differ at five semantic and five
  co-visibility pixels out of 50.3 million. RGB differs at 44 pixels, with mean
  absolute error 1.82×10⁻⁹ and maximum channel difference 0.0417. Box sets and all
  indexed boxes remain exact. The receipt lists every affected tensor.

`--indoor-prefetch-depth` accepts 1–4. Lower it to reduce CPU residency;
`--indoor-prefetch=false` disables lookahead. Direct random-access, Python indexed
capture and interactive callers remain non-speculative. The file writer stays
bounded. New worker pools are native-only; browser construction retains its
cooperative scheduling and incurs no additional model initialization.

The earlier [lookahead](evidence/throughput_512/measurements.json) and
[transport-index](evidence/throughput_512/acceleration.json) receipts retain their
original experimental identities. These finite throughput measurements do not
establish photographic realism or downstream learning utility. Motion-model
inference, O-voxel, multi-timestep throughput and browser runtime performance are
outside this benchmark.

## Person placement qualification

The [512-room population receipt](evidence/throughput_512/person-placement.json)
identifies the old hotspot as a hard-coded presenter coordinate, not a plotting
error. The corrected generator has **zero people at that anchor**, versus 38 in
the original audit. Total population remains 1,144, including 1,044 people in
primary rooms and 792 seated primary-room occupants. All 512 layouts validate.

The largest primary-room 24×24 heatmap cell now contains **8 people** (previously
38). Standing people occupy 185 cells, with at most 4 people in a cell. Presenters
span 19 of 24 heading bins and 32 spatial cells; none share a normalized position
at 1e-4 resolution. Seated positions are unchanged: natural furniture-related
clustering is retained. The page and paper use the refreshed current distribution,
32 rendered rooms and the matched factor qualification gallery.

## Release checks

182 core tests and all four explicit GPU integration tests pass, including
regeneration/readback coherence, co-visibility, optical flow and the independent
GI reference. The Burn exporter passes 13 unit and 10 dataset/resume tests.
Formatting, strict workspace Clippy and the WebGPU viewer compilation with motion
support pass. The 512-room audit and refreshed page/paper pass the publication
contract (3,751 bound artifacts). The root crate's publication dry run also builds
against registry dependencies, with no WGPU patches in the packaged manifest.
This package-build check does not substitute for registry-renderer benchmarking.
