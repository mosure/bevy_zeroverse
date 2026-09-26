# Reproducible indoor datasets

The supported native generator is `zeroverse_gen`. Indoor captures default to
lossless float32 sRGB color tensors, with lossless geometric and semantic labels.
Legacy scene modes retain their JPEG default. An example bounded dataset is:

```sh
cargo run -p bevy_zeroverse_burn --bin zeroverse_gen -- \
  --scene-type procedural-indoor --output out/office_dataset \
  --samples 100 --workers 2 --chunk-size 16 --seed 777 \
  --indoor-layout mixed --indoor-density 0.65 --indoor-human-density 0.25 \
  --indoor-quality auto --indoor-gi-rays 1024 \
  --width 640 --height 480 --cameras 4 \
  --playback-steps 3 --playback-step 0.5 --rotation-augmentation \
  --render-modes color depth normal semantic position \
  --color-codec raw --compression zstd --ov-mode disabled --no-ui
```

Each process owns one Bevy engine. `--workers` controls separate processes by
default. `--per-process=false --workers=1` runs one engine directly. Concurrent
capture threads sharing one engine are rejected for indoor generation because
they cannot preserve index ordering. The seed for global sample index `i` is
`base_seed.wrapping_add(i)`, independent of process count or chunk size.

Finite indoor jobs using `--per-process=true` replace each child after at most
256 captured scenes by default. Set `--max-scenes-per-process 16` for a shorter
lifetime, or `--max-scenes-per-process 0` to disable replacement; the alias
`--scenes-per-child` is also accepted. At most `--workers` children are alive at
once. The default does not affect other scene types or direct
`--per-process=false` generation. An explicitly positive limit with direct mode
is rejected. Python's persistent in-process engine is not restarted by this CLI
option.

Child replacement limits process lifetime; it does not establish that the engine's
unbounded live heap reaches a plateau. Peak memory still depends on scene and image
sizes, chunk size, and concurrent workers, and every replacement pays engine/GPU
startup costs. Final distribution metrics currently retain numeric samples for
quantiles, so that parent-side phase grows with dataset size; use
`--indoor-metrics=false` for large jobs when those post-generation statistics are
unnecessary. Short child jobs can produce partial chunks even before the final
job. Chunk indices reserve each job's complete span, and global sample indices and
seeds remain unchanged. The limit and worker count are scheduling choices and may
change on a compatible resume. `worker_lifecycle.jsonl` records each parent run,
child PID, reused worker slot, job ID, assigned sample/chunk spans, and terminal
status without retaining an ever-growing in-memory job list.

On Linux with glibc and x86-64, spawned indoor workers default to streaming
memory copies for larger GPU uploads. The child receives
`GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=32768:glibc.cpu.x86_rep_movsb_threshold=1073741824`
only when that environment variable is absent. Any user-supplied value is
preserved; `GLIBC_TUNABLES= zeroverse_gen ...` selects the system defaults.
This is a measured optimization on the qualified NVIDIA/Vulkan workstation,
not a portable speedup guarantee. It does not change tensor values, the capture
contract, the parent process, direct generation, or Python's in-process engine.
Those direct paths can opt in by setting the variable before process startup.

A failed child or process-launch error stops new launches and kills/reaps remaining
children. Completed files remain on disk. If parallel work leaves a gap after a
failure, resume refuses the incomplete index sequence; it does not silently skip
failed scenes or overwrite surviving outputs.

`--playback-step` is **normalized trajectory progress**, not seconds. Three
steps with increment `0.5` capture the start, midpoint, and endpoint. Each sample
contains `[time, camera, height, width, channels]`; the Rust in-memory view list
is time-major. Exported `time` records trajectory progress. Optical flow and
motion-vector capture are rejected for indoor datasets until previous-frame
pose semantics have been qualified. Derive correspondence from position,
depth, and camera calibration where appropriate.

`generation_config.json` records the base seed, generator version, capture-engine
identity, image size,
grammar, density, quality, augmentation, modalities, trajectory schedule,
storage encoding, and geometric conventions. Each sample carries its full
scene manifest, camera calibration, and per-sample `indoor_render_metadata`
with effective lighting/GI settings, transport backend/statistics, annotation
policy, and capture transport. GI settings and the renderer/dependency protocol identity are part of the resume contract.
Native capture uses CPU light clustering for reproducible accumulation; each
sample records the actual `light_clustering` policy. Native capture compiles
pipelines synchronously on the render thread to finish compiler tasks before
worker shutdown. GPU submissions/readback and the CPU writer remain asynchronous;
`pipeline_compilation` records this distinction.
A dataset made by an older capture engine remains readable, but appending with a
new engine (including this Bevy 0.19 migration) requires a new output directory. `--indoor-gi-rays` selects
64–16384 rays per probe: 256 is the efficient default; 1024 reduces measured
probe integration noise at higher GPU cost. Portable mode omits GI.
Successful finite generation also
writes `metrics/metrics.json`, scene/object/camera CSVs, and SVG distributions
and heatmaps for the **complete exported seed population**. Disable these
post-generation statistics with `--indoor-metrics=false` when unnecessary.
Planned object counts are distinct from projected visible counts. Heatmaps
use unaugmented room coordinates, with neighbor-room instances counted
separately; see the policies embedded in the metrics JSON.

To append another 100 samples, repeat the capture settings with `--resume
--samples 100`. The seed can be omitted on resume: the saved base seed is
restored. The sample count means additional samples, not a target total.
Resume verifies the capture contract, counts actual records in every chunk
(including partial chunks), and rejects missing chunk/sample indices or
incomplete records. New generation refuses existing indexed output. Writes
publish completed chunks or sample folders atomically. Failed or missing
modalities, non-finite buffers, unexpected seed/index ordering, and geometric
alignment failures stop generation; they do not silently skip scene seeds.

`--output-mode fs` produces one sample folder with `meta.safetensors`, the
manifest/color metadata, float32 NPZ image planes, and previews. Under the
default indoor `--color-codec raw`, color NPZ is authoritative and JPEG is only
a preview. `--color-codec jpeg` explicitly selects lossy JPEG quality 75 for
RGB storage. Semantic labels are always lossless float32 planes; palette PNGs
are previews. Existing legacy JPEG semantic files remain readable but their
lossy colors must not be treated as exact labels.

The renderer applies its tone map before capture. Both raw and JPEG exports
apply the sRGB transfer exactly once, without per-image contrast fitting.
Depth is linear camera-space Z in metres, normal is view-space `(n + 1) / 2`,
position is world position normalized by the exported AABB, and semantic is
the linear RGB class palette. Matrices are column-major `world_from_view`,
right-handed with camera forward `-Z`; vertical FOV is in radians. Annotation
precision is explicitly recorded per sample: `annotation_precision=1`
(`float32_geometry`) selects the native indoor MRT geometry pass, while `0`
(`float16_hdr`) identifies the legacy/fallback HDR intermediate. This field
records the render path, independently of float32 storage.

Procedural people use an independent seeded stream. `--indoor-human-density`
defaults to `0.25`; zero disables people without changing room furnishings.
The manifest records pose, chair support, stature/build, clothing, skin/hair
palettes, and local joints. `human_instance_ids` align the human axis with
manifest IDs; `human_count` excludes padded rows in a chunk. Pose positions
are world coordinates and remain static across camera trajectory steps.
The 21-joint bone names and parent indices are exported with poses.
`object_obb_instance_ids` associate furniture and person oriented boxes with
manifest IDs; `-1` indicates padding or an unlabelled legacy instance.
People are asset-free articulated geometry with cloth, hair, face and hand
details; their shading and silhouettes remain stylized compared with scanned
humans. No facial-expression or cloth-simulation realism is claimed.

The native CLI uses a rendezvous CPU writer: one batch may be encoding while
the next is captured. There is no waiting batch queue. Maximum batch residency
is therefore two uncompressed batches, plus encoder workspace and GPU/render
assets. Choose `--chunk-size` accordingly: five RGBA32 capture planes at
640×480 with four cameras and three steps require about 281 MiB per scene
before compression (and before metadata). A chunk size of 16 can therefore
consume several GiB per batch; smaller chunks reduce peak memory. The bounded CLI
writer/RSS check is reproducible with `scripts/qualify_indoor_writer.py`.

## Python capture and storage

`BevyZeroverseDataset(scene_type="procedural_indoor", indoor_seed=777, ...)`
uses dataset index `i` to request exactly `777 + i`. Random-access requests,
shuffling, repeated indices, and DataLoader workers therefore refer to the
same scene. Set `indoor_layout`, `indoor_density`, `indoor_human_density`, `indoor_quality`, and `indoor_gi_rays` directly;
the indoor depth default is linear. One engine/configuration is supported per
process. Use DataLoader's `multiprocessing_context="spawn"` when creating
workers after initializing a GPU engine in the parent.

`chunk_and_save(..., color_codec="raw")` preserves float32 sRGB color. It keeps
partial final chunks by default. Existing outputs require explicit
`append=True` and a dataset containing **only additional samples**; this is
append, not automatic dataset-index resume. Chunk iteration assigns every
chunk exactly once across workers, retains unequal final shards, and fails on
corrupt data unless `skip_corrupt_chunks=True` is explicitly requested.
Distributed training must handle unequal shard lengths explicitly rather
than relying on silent truncation. Cache fingerprints detect changed files.

`save_to_folders` and `FolderDataset` preserve all rows/channels, manifest,
calibration, semantic labels, and stored color encoding. NPZ labels are
authoritative. The old FFI `generate` example only implements the legacy
color/depth workflow and rejects indoor requests with a pointer to
`zeroverse_gen`.

## Qualification

CPU regressions are in `crates/burn/tests/indoor_roundtrip.rs` and
`crates/ffi/python/test_indoor.py`. The GPU qualification script below retains
its CLI logs, chunks, metadata, metrics, and final `report.json` in a new output
directory:

```sh
PYTHONPATH=/path/to/built/extension:crates/ffi/python \
  python crates/ffi/python/validate_indoor_dataset.py \
  --binary target/debug/zeroverse_gen --output out/dataset_qualification
```

It checks one versus two CLI workers, partial-chunk resume, rejected overwrite
and incompatible resume, native folder/Python interchange, out-of-order and
repeated indexed Python captures, and two spawned DataLoader workers. It uses
odd image dimensions (161 × 119), two cameras, three trajectory steps, and all
five supported modalities. Deterministic manifests, poses, and semantic
tensors were exact on the qualified native GPU. Pixel equality across other
GPU drivers or rendering backends is not promised.

The current Bevy 0.19 qualification passes native Auto256, including recycled
worker lifetimes and exact raw RGB/labels across scheduling changes. See the
[current dataset report](procedural_indoor/dataset_qualification_bevy019.json),
[process-memory report](procedural_indoor/process_memory_bevy019.json) and
[dependency/memory review](procedural_indoor_review_bevy019.md).

Historical generator-version-3 qualification on Bevy 0.17 passed both Portable and native Auto with GPU GI
(256 rays per probe, three diffuse bounces). The Auto evidence is recorded in
`docs/procedural_indoor/dataset_qualification_v3_auto.json`, including
per-sample lighting provenance and manifest-linked person/OBB instance IDs.
The 128-scene writer residency report is in
`docs/procedural_indoor/writer_report.json`; its RSS still increased during
that bounded run, so it does not establish a long-run memory plateau.

The historical explicit 1024-ray path also passed the full CLI worker/resume and Python
spawn-worker qualification; see
`docs/procedural_indoor/dataset_qualification_v3_auto_1024.json`. This checks
that the requested budget reaches actual render provenance in every process,
and that Python captures retain float32 precision tags and person/OBB IDs.
