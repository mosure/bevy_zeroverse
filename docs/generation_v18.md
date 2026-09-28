# Generation performance and dataset audit, v18

The [paper](../tex/bevy_zeroverse.tex) now describes the implemented engine and measured results. It no longer claims a trained reconstruction model or demonstrated real-data gains. [Machine-readable evidence](evidence/generation_v18/report.json), [hardware/compiler](evidence/generation_v18/hardware.json), [population metrics](evidence/generation_v18/metrics.json), [camera CSV](evidence/generation_v18/cameras.csv), [placement heatmaps](evidence/generation_v18/placement_heatmaps.svg), and [generated figures/tables](../tex/generated) accompany it.

## Measured performance

Each benchmark completes 40 rooms (seeds 12000–12039), excludes eight warmup rooms, and captures four 320×240 views per room, one timestep, static human density 0.25, Auto quality and 256 GI rays/probe. Five-channel runs request color/depth/position/normal/semantic. Timing includes construction, GI, rendering and readback; it excludes encoding, disk export, validation and report writing. Hardware: i9-12900K, RTX PRO 6000 Blackwell, Vulkan 610.43.02, Linux. These are optimized development builds without release LTO; the baseline optimizes the root crate at level 1 and candidates at level 3. Dependencies use level 3 throughout.

| Build / workload | Views/s | Room median / p95, seconds |
| --- | ---: | ---: |
| Original preparation, local wgpu patches, independent cameras | 2.84 | 1.348 / 2.364 |
| Shared geometry only, same build settings | 2.68 | 1.382 / 2.540 |
| Referenced finishes + root optimization level 3 | 3.31 | 1.073 / 2.078 |
| Bounded parallel tangent conversion | 4.28 | 0.761 / 1.775 |
| **Registry dependencies, grouped cameras, five channels** | **3.58** | **0.892 / 2.642** |
| **Registry dependencies, grouped cameras, RGB only** | **4.32** | **0.773 / 1.776** |

The matched local before/after mean room time fell **33.6%** (1.407 → 0.934 s), giving **50.6%** more completed views per second. CPU preparation fell from about 871 to 405 ms. Material generation fell from 260 to 109 ms; geometry plus tangent conversion fell from 381 to 170 ms, while GI setup fell from 226 to 124 ms. The initial shared-geometry-only trial did not improve throughput and remains in the evidence. The combined change includes a compiler profile change, not just an algorithmic ablation. There is one run per configuration; seed-paired observations do not quantify run-to-run machine variance.

Implementation:

- Generate architecture, object and human assemblies once; borrow them for transport before transferring their buffers into render meshes.
- Create only referenced finish variants, including expensive display/page textures. A regression compares retained material properties and texture bytes against the full palette.
- Reserve mesh handles in scene order, then convert independent meshes and their MikkTSpace tangents on a bounded four-thread pool. Wasm yields between serial conversions.
- Optimize the procedural crate in `cargo run`, as already done for its dependencies. No mesh/texture detail, GI rays, cameras or annotations are removed.
- Benchmark schema 3 separates the request enqueue from actual asynchronous preparation stages. Old `preparation_seconds` values measured enqueue time, not total preparation.

The registry runs build the normalized 0.22 crate archive against published wgpu-core/hal 29.0.4, without root patches. Local patch results must not be advertised as registry performance. Process memory, residency counters, GPU timestamps and PID-specific NVML samples are retained in compressed raw logs under [benchmarks](evidence/generation_v18/benchmarks). Desktop GPU activity was present and recorded. Asynchronous diagnostic spans are incomplete per-scene samples; NVML active-process samples are not a continuous GPU-occupancy measure. Forty scenes cannot establish a memory plateau or unlimited-process safety. Keep finite worker lifetimes.

## Diversity and correctness

A separate audit uses **1,024 consecutive seeds 24000–25023**, four default grouped cameras and no rejected-seed replacement. All pass layout/camera validation. The first **64 rooms / 256 views** are rendered at 384×288 with all five channels; no aesthetic selection is used.

- 503 numeric distributions (401 with nonzero observed variance), 33 categorical distributions, object counts including zero scenes, joint histograms, camera parameters and placement heatmaps are exported. These are correlated measurements, not 503 independent degrees of freedom.
- Main-interior chairs range 1–26, mean 9.09. Room area spans 38.01–267.28 m². All seven combinations of one to three exterior window walls occur; 337 rooms contain near-full-height openings. Vertical FOV spans 28.07–106.97°. Solar illuminance spans 0.10–99,862 lux.
- The seed/texture-independent, quantized occupancy-signature HLL estimate is 1,018.8 for 1,024 rooms (4,096 registers, approximately 1.63% relative standard error). Its spatial quantization is declared in the metrics file. It is not an exact unique-scene count or evidence of 10M distinguishable images.
- Frozen cached SigLIP2 base-patch16-224 embeds all 256 RGB images. Cross-scene nearest cosine distance is min/median/max **0.0394 / 0.0628 / 0.1832**; no pair falls below the exploratory 0.02 threshold. Equal-weight room centroids have entropy-effective rank **32.29**, with 34 dimensions explaining 90% of variance. All 14 weight shards are verified. Global semantic spacing does not measure physical realism or downstream learning value.
- The 192 rendered reference edges have room-wise worst-overlap mean **47.4%**, minimum **34.4%**. Three pairs fall slightly below the requested 35% proxy bound; these failures remain in the report. A proxy proposal constraint is not a hard rendered-pixel guarantee.
- Across 256 views, the maximum per-view p99 depth/position disagreement is **2.86 µm**, reprojection disagreement **0.00262 pixels**, and unit-normal error **2.38e-7**. At least seven semantic classes occupy 32 or more pixels in every view (mean 11.38).
- All **45** rooms with people in the primary zone contain a person semantic mask of at least 32 pixels in a view. For the broader non-neighbor interior, **14/60** occupied rooms show no person label; people confined to secondary zones are not an obligation of the primary-room camera policy. This is class visibility, not proof that every individual is visible.

The paper uses the first eight consecutive reference images, including dark scenes. Visual inspection still finds simplified people/materials and approximate glass/lighting. The earlier [matched Cycles study](appearance_v14.md) remains a separately scoped physical-rendering diagnostic, not a new validation of every v18 room. Photographic realism and 10M-scale pretraining utility remain unproven.

## Training defaults and reproducibility

`zeroverse_gen --scene-type procedural-indoor` now defaults to four cameras, one worker, 16 rooms/chunk, one timestep, static human density 0.25, furnishing density 0.65 and 256 GI rays/probe. Camera zero anchors a 35% shared-surface policy with 0.25–3 m baselines. Camera paths stay in the primary room. `--indoor-camera '{"multiview":null}'` explicitly restores independent sampling. Viewer camera counts remain explicit to avoid rendering unseen extra views. Other scene types retain their defaults. Choose additional processes from measured device/memory capacity; 16 simultaneous rendering processes are not a useful one-GPU starting point.

Old archived camera settings missing the overlap field retain independent semantics; newly parsed configuration inherits the training policy. Start new capture shards when changing engine identity. `LiveDataset` now rejects a different renderer configuration in the same persistent process instead of letting stale resolution/timestep outputs reach the encoder. Finite CLI process workers remain the supported configuration/lifetime isolation mechanism.

## Release validation

The full workspace test run with `human_motion,embedding_audit` passes, including ten isolated GPU dataset cases, annotation serialization, and SigLIP CPU/GPU checks. Workspace Clippy with all targets and those features passes with warnings denied; formatting and 60 Python reporting/validation tests pass. Explicit GPU/model-dependent tests outside this run retain their documented opt-in status.

An actual `zeroverse_gen` run using the new indoor defaults writes eight rooms with four views each. LZ4 decompression and safetensors decoding verify all five finite image channels, calibration, consecutive seeds, manifests and ceiling-light boxes ([CLI evidence](evidence/generation_v18/cli/validation.json)). This is an export smoke check, not an encoding-throughput benchmark.

The `web,human_motion` viewer builds for Wasm. Headed Chrome/WebGPU checks pass for occupied Auto and Portable scenes in RGB and semantic modes, with complete semantic coverage and no missing solid architecture or WebGPU validation errors ([browser evidence](evidence/generation_v18/browser)). These checks exercise static occupied scenes; they do not repeat the historical ARDY inference qualification. The normalized root crate passes `cargo publish --dry-run --locked` against registry dependencies.

Reproduce in fresh output directories (the occupied scene asset root must contain `assets/burn_human`):

```sh
cargo build --bin indoor_bench --bin indoor_validate --features embedding_audit --bin indoor_embed
python scripts/indoor_telemetry.py --output out/perf -- \
  target/debug/indoor_bench --asset-root . --scenes 40 --warmup-scenes 8 \
  --seed 12000 --cameras 4 --width 320 --height 240 --gpu-timings --output out/perf
# Add --rgb-only for the RGB workload. Benchmark telemetry requires nvidia-ml-py.
target/debug/indoor_validate --seed 24000 --audit-seeds 1024 --renders 64 \
  --cameras 4 --width 384 --height 288 --labels --output out/cohort
python scripts/indoor_report.py out/cohort --figures
python scripts/indoor_multiview_report.py out/cohort
python scripts/indoor_embedding_report.py index --cohort training=out/cohort --output out/index.json
target/debug/indoor_embed --index out/index.json --output out/embeddings --batch-size 16
python scripts/indoor_embedding_report.py report out/embeddings/embeddings.json
```

`paper_indoor_report.py` verifies image identities and completed benchmark work before generating JSON, TeX tables and figures. The checked-in compressed benchmark logs can be expanded into a fresh directory with `gzip -dk *.gz` for the existing `indoor_bench_report.py` tool. Raw RGBA capture files remain local; their SHA-256 list, exact seeds/configuration, manifests' capture hashes, verified embeddings and metrics are archived. Encoding throughput must be measured separately with the actual codec/chunk configuration. The baseline was an instrumented pre-release checkout; it is a retained measurement, not a previously published crate benchmark.
