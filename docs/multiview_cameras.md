# Multi-view camera sampling

By default, procedural indoor cameras sample a connected set of views for reconstruction.
A reference camera anchors shared visible surfaces; every camera remains inside
the primary furnished room, with swept checks against obstacles. Other scene
modes retain their samplers.

## Camera baseline control

Use **Capture cameras → Camera baseline** in the inspector, or pass JSON through
`--indoor-camera`, Python's `BevyZeroverseConfig.indoor_camera`, or the dataset
configuration:

```json
{"baseline": 0.5}
```

The continuous control runs from **0 (very narrow)** to **1 (very wide)**.
It expands into the following concrete `multiview` constraints. Distances are
metres; camera 0 is the reference.

| Advanced parameter | Program for baseline b | Default b = 0.5 |
| --- | --- | --- |
| Minimum separation of every pair | 0.04 + 0.46 b | 0.27 m |
| Minimum distance from reference | 0.06 + 2.30 b | 1.21 m |
| Maximum distance from reference | 0.35 + 9.65 b | 5.175 m |
| Minimum estimated bidirectional overlap | 0.65 − 0.55 b^0.6 | ≈ 0.287 |
| Minimum horizontal group spread | 0.20 + 0.10 b | 0.25 |
| Independent path variation | b | 0.5 |

These constrain a stochastic sampler, rather than determining exact camera
positions. Room dimensions, furniture, camera intrinsics and visibility rejection
shape the realized spacing and overlap. The [recorded camera-baseline study](camera_baseline_v21.md)
shows their distributions across matched rooms. Wider settings trade shared
pixels for viewpoint variation; they do not guarantee a training-quality optimum.

Expand **Advanced multi-view constraints** to edit individual fields. Editing any
one switches to **Custom** mode; moving the baseline slider replaces all six
advanced values. Settings apply on **Regenerate [R]**, preserving slider behavior.
Serialization records the expanded values, so reload and regeneration have no
hidden preset precedence. JSON must specify `baseline` or `multiview`, not both.
For example:

```json
{
  "primary_room": true,
  "path_length_min": 0.03,
  "path_length_max": 8.0,
  "long_path_fraction": 0.45,
  "multiview": {
    "min_overlap": 0.2871353,
    "min_baseline": 0.27,
    "min_reference_baseline": 1.21,
    "max_baseline": 5.175,
    "min_spread": 0.25,
    "trajectory_variation": 0.5
  }
}
```

`{"multiview":null}` explicitly disables grouping. An omitted `multiview` uses
the baseline-0.5 program. In an explicit `multiview` object, omitted
`min_reference_baseline` is **0** for compatibility with prior custom policies;
other omitted fields receive current defaults. Thus `{"multiview":{}}` is a
custom policy, not the baseline-0.5 shorthand. Archives preserve their recorded
constraints; absent legacy spread and variation fields remain disabled.

`min_overlap`, `min_spread` and `trajectory_variation` must be in [0,1]. Distance
bounds require `0 < min_baseline <= max_baseline <= 100`, and
`0 <= min_reference_baseline <= max_baseline`. Impossible combinations fail
within a bounded search, without a relaxed fallback.

## Trajectories and geometry checks

**Trajectory length and room bounds** controls each camera's temporal travel,
separately from spacing between cameras. Joint collision/overlap rejection can
favor shorter accepted routes at wide spacing; set a larger minimum travel when
needed. Set both path lengths to zero for static
views. A single-camera request remains valid and has no pairwise constraint.
For a fixed translating rig also set `trajectory_variation: 0`; a collinear
array requires `min_spread: 0`.

- With the default policy, every additional camera must overlap camera 0 in both directions, creating a
  connected reference graph with O(N) edges. Nonreference pairs need not overlap.
- Overlap is the fraction of sampled image pixels whose first geometric surface
  lies inside the other frustum and is unoccluded. The smaller of the two
  directional fractions must pass. Shared semantic classes or intersecting
  frusta alone are insufficient.
- The test uses 13×9 rays at normalized times 0, .25, .5, .75 and 1, each camera's
  intrinsics and full capture aspect ratio. Near/far planes are 0.1/50 m.
  Glazing occludes geometric annotation surfaces.
- Euclidean pair separation, reference distance and group geometry are checked
  at 33 synchronized times. Horizontal spread is the minor/major standard-deviation
  ratio of camera centers; stereo is exempt. Relative motion is the normalized
  RMS displacement difference after removing starting offsets; its minimum is
  0.15 times `trajectory_variation`. Entirely static pairs are exempt.
- Reference radius proposals use the reachable room extent and usable height.
  Heading, travel and curvature deformations are bounded by available space.
  All full paths retain swept collision/primary-room checks.
- Search permits up to 64 reference proposals and 384 proposals per additional
  view. High overlap, long paths and broad spacing can conflict in furnished
  rooms. Sampled group/overlap constraints are not continuous-time guarantees.
- No model or GPU initialization is required for sampling. Reference ray samples
  are reused. Furniture, materials, people and cameras use separate seed streams.

Rust callers can use `MultiViewSettings::from_baseline(b)` or
`IndoorManifest::resample_cameras(count, settings, aspect)`; failed resampling
preserves the prior camera set. Manifests retain the concrete policy, intrinsics
and full trajectories. Generator, multi-view policy and capture-engine identities
prevent resumptions from mixing incompatible camera samples. Always retain the
identities exported by the build used for capture.

## Metrics and rendered qualification

Metrics include `camera_overlap.csv`, `camera_groups.csv`,
`camera_paths.csv` (33 times/camera), triangulation angles, reference and pair
separation, spread and motion distributions. Independently recompute geometry:

```sh
python scripts/indoor_camera_group_report.py out/multiview --output out/camera_report --seed 24005
```

Capture actual depth/calibration and evaluate all-camera co-visibility:

```sh
cargo run --bin indoor_validate -- \
  --seed 24000 --audit-seeds 128 --renders 128 --cameras 4 \
  --width 320 --height 240 --human-density .25 --playback-steps 3 --labels \
  --indoor-camera '{"baseline":0.5}' --output out/multiview
python scripts/indoor_covisibility_report.py out/multiview --output out/covisibility
```

Retain raw captures (do not use `--no-raw`). The NumPy diagnostic reprojects each
source depth pixel into all other cameras, using nearest-pixel depth agreement
within `0.01 m + 0.002*z`. It records individual camera membership, cardinality,
bidirectional reference overlap, graph connectivity, target misses, input hashes
and original run IDs. Pixel observations repeat surfaces across views and time;
rooms are the independent sampling units. No low-overlap samples are filtered.

This diagnostic uses a different surface predicate from the production GPU
co-visibility annotation's symmetric tangent-plane test. The gallery exports
exact production masks separately. Both count first geometric surfaces, including
glass; neither measures reflected or refracted RGB correspondence. Proxy geometry,
finite temporal samples, thin objects and moving people can affect overlap.
Photographic realism and downstream learning utility need separate evaluation.

## Calibration and capture time

Every new native/web sample carries `View.calibration`: full row-major **K in
pixels**, `[width, height]`, `lens_model: "pinhole"`, lens model version 1 and
schema version 1. Pixel coordinates increase right/down, with the top-left pixel
center at **(0.5, 0.5)**. `world_from_view` retains its column-major Bevy axes
(right/up, forward −Z) and metres. For axial depth `d = −view.z`, projection is
`u = fx*x/d − skew*y/d + cx`, `v = −fy*y/d + cy`.

Rust FS/chunk exports and the Python dataloader preserve `intrinsics[...,3,3]`,
`image_size[...,2]` and the shared UTF-8 JSON `camera_calibration` tensor. The
leading dimensions are `[time, camera]`, prefixed by `[sample]` in chunks. Legacy
archives remain readable with calibration absent; mixed calibrated/unlabeled
batches, inconsistent dimensions and malformed K are rejected. Cropping or
resizing requires explicitly updating K; the loader does not silently crop a
calibrated image.

The existing `fovy`, `near`, `far` and normalized `time` labels remain. New labels
separate `trajectory_progress` (after playback easing) from `time_seconds`. Set
`indoor_camera.duration_seconds` to assign an explicit metric timeline:
`time_seconds = time * duration_seconds`. Without it, seconds are unspecified
(`None` in Rust/JSON; zero with `time_seconds_valid = 0` in tensors). This retimes
normalized camera/human trajectories; it does not claim the native timing of a
generated human-motion clip. Direction reversal reverses poses, not this timeline.

Rendering currently remains centered pinhole. The versioned K helpers support
pinhole projection/unprojection, including mathematical off-center test cases,
but there is **no principal-point or distortion sampling option**. A distorted
RGB-only warp would invalidate depth, flow and visibility correspondence. A
future lens model must integrate those rendering paths and their oracles first.

## Independent metric handheld motion

```json
{
  "baseline": 0.4,
  "duration_seconds": 1.5,
  "path_length_min": 0.01,
  "path_length_max": 0.5,
  "long_path_fraction": 0,
  "handheld": {
    "translation_m": [[-0.12,0.12],[-0.035,0.035],[-0.35,0.35]],
    "rotation_degrees": [[-8,8],[-5,5],[-3,3]],
    "reverse_probability": 0.5
  }
}
```

Translation ranges use the initial camera's right/up/forward axes. Negative
forward displacement means backward striding. Endpoint yaw/pitch/roll increments
are sampled independently, then interpolated without a shared moving look-target
constraint. Reversal swaps both position and orientation endpoints. Existing
primary-room, swept collision, baseline and overlap checks still apply. To
request independent cameras as well, use `"multiview": null` instead of `baseline`.
To allow rotation without translation, permit zero minimum path length. This is
a bounded short-motion model, not measured human gait or a handheld IMU model.

## High, low and negative reference pairs

```json
{
  "baseline": 0.5,
  "path_length_max": 0.6,
  "long_path_fraction": 0,
  "overlap_mixture": {
    "weights": [0.6,0.3,0.1],
    "high": [0.45,1.0],
    "low": [0.02,0.30]
  }
}
```

For each reference-to-camera pair, the seed selects **high / low / none** once,
before placement retries. Both directed proxy fractions must lie in the selected
band at all five checked trajectory parameters; `none` requests exactly zero
proxy overlap. These bands replace `multiview.min_overlap`, retaining its spacing
and path-diversity constraints. `overlap_mixture` requires multiview enabled.
Impossible requests fail rather than quietly switching strata or skipping seeds.
Weights describe requested pairs, not guaranteed frequencies of rendered overlap.

`indoor_render_metadata.camera_qualification` records:

- `requested_pairs`: sampled strata and proxy bands.
- `placement_estimate`: reference edges measured with 13×9 proxy rays at five
  trajectory parameters. Its denominator includes rays that miss geometry.
- `rendered_overlap`: every directed pair at each captured timestep, from the
  production co-visibility mask. Its denominator is **valid source pixels**.
  Each pair includes actual metric baseline and, when position is captured, the
  mean triangulation angle over up to 4096 shared world-position samples.
- `scene_family`: generator-version/seed ID and deterministic split bucket
  `[0,10000)`. Assign splits by family before expanding frames, camera controls,
  appearances or sensor variants; no automatic train/validation split is imposed.

Request `--render-modes color depth position co-visibility` to obtain measured
pixel overlap and triangulation. Missing measurements are absent/null, never
reported as measured zero. To avoid extra proxy casts for ordinary RGB-only
exports, placement estimates are omitted unless co-visibility or an overlap
mixture is requested. Proxy-zero is **not** a guarantee of zero rendered overlap;
both remain available for training policy. Glass uses the annotation first-surface
convention, not photographic transparency. No downstream-model score filters pairs.

## Appearance, sensor effects and qualification

`--indoor-appearance` accepts independent `material_seed`, `lighting_seed`,
`material_detail` in `[0,1]`, `illumination_scale` in `[0.05,1]`, and
`exposure_ev100_offset` in `[-2,2]`. It changes materials, light transport inputs
and exposure after architecture/furnishing generation. Material detail scales
surface relief and pattern contrast, not mesh complexity. The scene's geometry,
semantic attachments, people and camera program remain unchanged. Positive EV
offsets darken exposure. Omitted factors preserve the scene's existing samples.

The Rust dataset CLI separately accepts optional RGB-only export augmentation:

```sh
zeroverse_gen --scene-type procedural-indoor --output output/multiview \
  --samples 16 --seed 0 --cameras 2 --ov-mode disabled --no-ui \
  --playback-steps 2 --playback-step 0.5 --width 256 --height 160 \
  --indoor-camera '{"duration_seconds":2,"baseline":0.5}' \
  --indoor-appearance '{"material_seed":55,"illumination_scale":0.5}' \
  --rgb-sensor '{"seed":7,"white_balance_gain":[[0.9,1.1],[1,1],[0.9,1.1]],"blur_sigma_pixels":[0,0.6],"noise_std_srgb":[0,0.02],"jpeg_quality":[75,95]}' \
  --color-codec raw --render-modes color depth position co-visibility
```

RGB sensor parameters are sampled per seed; noise is independent per captured
view. The recorded order is display-linear white-balance gains, sRGB transfer,
Gaussian blur, additive Gaussian noise, then optional JPEG encode/decode. This
is a reproducible corruption model, not physical RAW-camera simulation. The
sample records requested and realized settings in `indoor_render_metadata.rgb_sensor`.
Alpha, K, depth, normals, flow, semantics and visibility stay untouched, so these
remain labels of the latent sharp scene. Use raw storage with sensor JPEG to
avoid a second lossy encode. The bounded CPU export worker overlaps this work
with rendering; omitted sensor settings incur no augmentation work. The viewer
and live FFI do not apply this export-only stage.

Run the formal Rust qualification sweep, without publishing or using Python:

```sh
cargo build --locked --bin indoor_validate --no-default-features --features multi_threaded
cargo run --locked -p bevy_zeroverse_publication -- qualify --output out/qualification
cargo run --locked -p bevy_zeroverse_publication -- verify-qualification --output out/qualification
```

The default recipe renders two matched room seeds for eleven separate camera,
aspect, lighting, exposure and material factors: 22 scene variants, two views and
two timestamps each. It writes `recipe.json`, `receipt.json`, logs, per-scene
manifests/calibration, distributions and an HTML annotation gallery. `--recipe`
accepts a versioned JSON recipe to expand the seed count or factor ranges. Every
case, including failures, is retained; output folders are immutable. The receipt
binds artifact hashes, actual executable hash, capture-engine/source identity,
build target, feature flags and locked package versions (including optional
packages). It reports directed pixel-overlap/baseline/angle distributions,
per-view self-reprojection errors, family counts and exact RGB duplicates.

The regular publication `refresh` also builds and verifies this canonical sweep,
caches it only for the same source identity and recipe, and installs its receipt
and gallery with the page and paper in one transaction. `verify` checks that
bundle offline. The paper's factor table is calculated from the same receipt.

This bounded factor sweep isolates renderer factors with people disabled. Sensor
operators have separate export regressions. Neither proves transfer, perceptual
uniqueness, a fitted training mixture or photographic realism. Choose/finalize
mixtures using independent synthetic validation and real downstream evaluations;
matching a reused real-image histogram is insufficient evidence.
