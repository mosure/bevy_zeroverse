# Multi-view camera sampling

By default, procedural indoor cameras sample a connected set of views for reconstruction.
A reference camera anchors shared visible surfaces; every camera remains inside
the primary furnished room, with swept checks against obstacles. Other scene
modes retain their samplers.

## Lens and field-of-view diversity

The **Cameras → Lenses** controls define a continuous proposal distribution.
Bounds default to **28–145° vertically**. **Lens preference** shifts the median
smoothly from narrow (−1) toward wide (+1); **Lens spread** sets concentration
around that median (0 fixes the lens, 1 spans the full interval). The live density
plot shows the proposal PDF and its 10th/50th/90th percentiles. This is independent
of **camera baseline**, which controls separation in metres, not focal length.

```json
{
  "motion_fraction": 1.0,
  "intrinsics": {
    "fov_y_min_degrees": 28,
    "fov_y_max_degrees": 145,
    "bias": -0.2,
    "spread": 1.0
  }
}
```

Pass this through `--indoor-camera`, `BevyZeroverseConfig.indoor_camera`, or the
web viewer's `indoor_camera` query field. UI edits take effect on Apply/Next and
shared URLs retain the settings. Bounds may be equal for a fixed lens; the valid
range is 15–150°. For an ultra-wide-only cohort, use
`{"intrinsics":{"fov_y_min_degrees":125,"fov_y_max_degrees":145}}`.

Sampling interpolates in `log(tan(fov_y / 2))`, equivalently negative log
normalized focal length. For uniform `u`, the interpolation fraction is
`sigmoid(logit(0.5 + 0.49*bias) + spread*logit(u))`.
There is no FOV threshold. At bias=0 and spread=1, log focal length is uniform;
zero spread gives the median exactly, including in rare placement recovery.
Clearance, content visibility and overlap rejection still shape the accepted
PDF; exported FOV/focal-length histograms describe the realized views.
Explicit archived `ultra_wide_probability` values retain their two-band sampling
and RNG sequence. Editing preference or spread in Studio switches such an imported
policy to the continuous sampler; importing it alone does not reinterpret it.

These are rectilinear perspective lenses, with straight-line projection and
`lens_model=pinhole`. At aspect 1.6, a centered 145° vertical lens covers about
158° horizontally. Fisheye distortion is a separate opt-in lens program, described below;
these perspective controls continue to govern pinhole cameras. Full per-view K, shifted principal points, depth, normals,
positions, flow and co-visibility use the same projection. An enlarged swept
clearance sphere encloses the full near-plane rectangle, including optical-center
shift; wide lenses cannot bypass room/furniture clearance. Explicit bounds also
apply to recovery cameras. Archived manifests retain their recorded lenses and
historical bounds when this policy is absent.

## Fixed and moving cameras

**Cameras → Camera motion → Moving cameras (%)** is the single motion control.
Default **100%** requests trajectories for every capture view. **0%** fixes every
capture camera's position and orientation for the entire sequence while human
and furniture animations continue on the shared playback clock. Intrinsics are
constant over time for both fixed and moving cameras.

```json
{"motion_fraction": 0.5, "path_length_min": 0.03, "path_length_max": 8.0}
```

Intermediate values are seeded, independent per-view probabilities, not rounded
counts: a two-view rig at 50% can contain zero, one or two moving cameras. Roles
are fixed before placement retries, stable when adding view IDs, and recorded by
the manifest policy plus each camera's static pose/trajectory. Changing the
fraction never edits the travel/handheld settings; they apply only to selected
moving cameras. Fixed cameras skip navigation-route generation and handheld
increments, but retain near-plane clearance, framing and synchronized multi-view
overlap/separation checks. A moving peer can have a fixed reference camera.

The setting affects procedural indoor capture cameras, not editor navigation or
the viewer's separate whole-scene yaw control. Playback mode still controls the
shared scene clock. Setting both travel bounds to zero remains supported for
existing configurations; `motion_fraction:0` is the recommended motion switch.

## Camera baseline control

Use **Capture cameras → Camera baseline** in the inspector, or pass JSON through
`--indoor-camera`, Python's `BevyZeroverseConfig.indoor_camera`, or the dataset
configuration:

```json
{"baseline": 0.5}
```

The continuous control runs from **0 (very narrow)** to **1 (very wide)**.
For room-only pinhole rigs it expands into the following concrete `multiview`
constraints. Distances are metres; camera 0 is the reference.

| Advanced parameter | Program for baseline b | Default b = 0.5 |
| --- | --- | --- |
| Minimum separation of every pair | 0.04 + 0.46 b | 0.27 m |
| Minimum distance from reference | 0.06 + 2.30 b | 1.21 m |
| Maximum distance from reference | 0.35 + 9.65 b | 5.175 m |
| Minimum estimated bidirectional overlap | 0.65 − 0.50 b^0.7 | ≈ 0.342 |
| Minimum horizontal group spread | 0.20 + 0.10 b | 0.25 |
| Independent path variation | b | 0.5 |
| Visible-region target variation | b | 0.5 |

### Mixed mounting and lens coverage

The baseline preset also writes `multiview.mixed_rig: true`. When a rig contains
ceiling or fisheye cameras, its versioned mixed-rig program accounts for mounting-height
separation and unequal sensor coverage:

| Constraint | Program for baseline b |
| --- | --- |
| Maximum horizontal reference separation across mounting roles | 0.80 + 9.20 b metres |
| Minimum horizontal reference separation across mounting roles | ordinary minimum reference distance |
| Maximum 3D diameter within either mounting group | twice the ordinary maximum reference distance |
| Maximum 3D distance to camera 0 for its own mounting group | ordinary maximum reference distance |
| Nominal minimum shared fraction in each direction | 0.05 − 0.03 b |
| Minimum larger directed overlap, room ↔ ceiling | 0.45 − 0.25 b |
| Minimum larger directed overlap, same mounting role | ordinary minimum overlap |

All-pair physical minimum separation, minimum 3D reference distance, horizontal
spread and relative-motion diversity remain required. A ceiling overview and a
narrow room image can therefore share useful content without claiming that
most pixels of the much wider image lie in the narrow one. No directed overlap
or metric baseline export is rescaled. Use the measured rendered view graph to
select training pairs; these coarse placement bounds are not pixel guarantees.

The nominal directional floor is capped by half the aligned, centered sensor
footprint overlap, with a nonzero lower bound of 0.005. This is exact for
rectilinear footprints and sampled on a 17×17 valid-ray grid for fisheye pairs.
For example, a centered 45° perspective image occupies only about 2.3% of a
140° image at the same pose; imposing a fixed 5% floor would make that useful
nested pair impossible. `MixedRigContract::pair_overlap_floor` exposes the
effective floor. Footprints are cached by immutable lens slot during search;
actual first-surface occlusion is still tested at every temporal checkpoint.

The pinhole reference uses content-conditioned sampling within the configured
FOV bounds, including required human visibility. Peer pinhole draws and every
fisheye assignment remain fixed during search; narrow peers can aim at other
reference-visible landmarks. A new reference calibration invalidates footprint
caches. Accepted intrinsics can therefore vary with baseline and scene content;
the manifest always records the actual calibration. Fixed peer FOVs participate
in proposal screening before clearance and content checks. Recovery re-aims
narrower lenses when needed; broader peers retain the shared landmark. Bounded
retries can place constrained lenses earlier without changing exported camera
IDs, geometry, lens assignments or the overlap and motion acceptance rules.

Narrow rigs propose coordinated paths inside the requested travel intervals.
Moving/static roles, lens assignments and swept collision checks remain intact.
Wider baselines permit larger positional and trajectory differences. Rejection
can still occur where a requested rig, motion range or mounting clearance does
not fit the actual scene; the sampler never returns a partial rig.

`IndoorManifest::mixed_rig_contract()` exposes the effective numeric bounds and
program version. `accepts_camera_geometry()` checks this contract; the existing
`camera_group_geometry()` still reports physical 3D measurements. The CPU camera
audit exports both. Single-camera rigs retain their existing policy.

### Peer overlap for reconstruction

Sharing geometry with camera zero does not ensure that peers share geometry with
each other, especially beneath a broad ceiling/fisheye anchor. The baseline preset
first attempts a **peer-aware rig** using the same lens/mount assignments, spacing,
trajectory and collision checks, with these additional placement targets across
five synchronized trajectory times:

| Preferred peer target | Baseline b | b = 0.5 |
| --- | --- | --- |
| Minimum in each direction | 0.05 − 0.04 b | 0.03 |
| Minimum larger directed overlap | 0.10 − 0.08 b | 0.06 |

The directional target is capped by half the calibrated aligned sensor footprint
(minimum 0.005), allowing useful narrow views within wide sensors. Work is bounded
by two anchor/order attempts, 1,536 peer proposals and an 8,192 reference-proposal
checkpoint (at most one inner proposal can cross that checkpoint). A failed
preferred search falls back to the original sampler with the **unchanged requested
configuration**, never a partial rig or reduced clearance. The failed attempt's
work is recorded separately; feasible rooms do not become new rejection cases.

Fallback rigs get bounded gaze refinement: only room-camera look targets can
change, by at most 60 degrees from their original orientation. Positions, paths,
calibration, mounting roles and camera zero remain fixed during this refinement.
It reduces the worst-time deficit against stronger soft targets of
`0.10 - 0.06*b` in each direction and `0.40 - 0.30*b` in the larger direction.
No previously supported peer pair may lose support or become essentially
disjoint. Original reference visibility, rich content, required human visibility
and swept camera clearance are rechecked. Independent aims and sampled visible
surfaces avoid making every optical axis intersect one room-centre point.

Refinement considers at most `min(12 * camera_count, 144)` framings. Midpoint
bounds prune impossible improvements before full trajectory evaluation. First-hit
rays and lens targets are cached. Single/two-view, all-mounted, explicit handheld,
custom policy, Connected-graph and explicit overlap-mixture requests retain their
existing placement contracts. Sampling diagnostics distinguish preferred-search
success, fallback/discarded work and framing candidates/updates. Weak pair counts
in those diagnostics are populated only when fallback framing is evaluated.

`IndoorManifest::camera_coherence_policy()` exports versioned targets and work
limits. `camera_pair_overlaps()` audits every unordered pair with both directed
fractions; `camera_overlap()` retains its reference-edge/connected-graph API.
The CPU audit exports `all_pair_overlap` and `coherence_policy`. These are
first-surface placement estimates, **not rendered overlap guarantees**. Rare
fallback pairs can remain weak. Use captured co-visibility annotations to assess
valid-pixel overlap, particularly through glass and around thin geometry; no
placement proxy proves downstream learning utility.

Older/explicit policies omitting `mixed_rig` remain exact. Turning **Mount-aware
baseline** off, editing any advanced baseline value, choosing Connected graph,
or enabling explicit overlap strata retains exact Euclidean/bidirectional
constraints. This prevents a stored custom policy from being silently relaxed.

These constrain a stochastic sampler, rather than determining exact camera
positions. Room dimensions, furniture, camera intrinsics and visibility rejection
shape the realized spacing and overlap. The [current camera study](evidence/camera_counts/study.json)
records distributions across matched rooms and camera counts. Wider settings trade shared
pixels for viewpoint variation; they do not guarantee a training-quality optimum.

With positive target variation, peers can aim at different first-surface proxy
hits visible to the reference camera. The probability of proposing a new target
increases with camera count; the aim interpolates by the selected variation.
Every candidate still passes its declared overlap contract, semantic-content, complete
path and group-geometry gates. Feasible candidates are ranked for nearest-peer
spacing, nearest optical-axis angle and additional primary-room surface cells.
Preferred spacing scales with room diagonal divided by the square root of camera
count. This is a preference, not an additional infeasible packing constraint.
Selection uses at most 2/3/4 feasible candidates for 2–4/5–8/9+ views, within a
12-proposal window after the first feasible candidate and the existing total
search budget. Narrow/shared-target policies retain first-fit selection. If normal and room-space
searches fail for 2–11 views, one bounded search for an extra view may supply a
subset; every subset rechecks group geometry while retaining the reference,
per-view swept clearance and all reference overlaps. No thresholds are relaxed.
These coarse surface cells are placement diagnostics, not a measured mesh-area
coverage or a reconstruction-performance guarantee. Temporal routes can remain
related; gaze variation does not imply independent path topology.

Expand **Advanced multi-view constraints** to edit individual fields. Editing any
one switches to **Custom** mode; moving the baseline slider replaces all seven
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
    "min_overlap": 0.342214,
    "min_baseline": 0.27,
    "min_reference_baseline": 1.21,
    "max_baseline": 5.175,
    "min_spread": 0.25,
    "trajectory_variation": 0.5,
    "target_variation": 0.5
  }
}
```

`{"multiview":null}` explicitly disables grouping. An omitted `multiview` uses
the baseline-0.5 program. In an explicit `multiview` object, omitted
`min_reference_baseline` is **0** for compatibility with prior custom policies;
omitted `target_variation` is **0** to preserve shared-target custom rigs.
Other omitted fields receive current defaults. Thus `{"multiview":{}}` is a
custom policy, not the baseline-0.5 shorthand. Archives preserve their recorded
constraints; absent legacy spread and variation fields remain disabled.

`min_overlap`, `min_spread`, `trajectory_variation` and `target_variation` must be in [0,1]. Distance
bounds require `0 < min_baseline <= max_baseline <= 100`, and
`0 <= min_reference_baseline <= max_baseline`. Impossible combinations fail
within a bounded search, without a relaxed fallback.

## Trajectories and geometry checks

Camera eye heights use a room-seeded continuous mixture of low, seated, standing
and elevated viewpoints, relative to the local floor and bounded by roof/fixture
clearance. Gaze is sampled separately: 30–48% of proposals per room use a
near-horizontal optical axis, with a continuously sampled symmetric pitch span
of ±2–7°. Other proposals retain content-directed targets. These are proposal
weights, not promised accepted frequencies; furniture, people and overlap still
constrain the resulting views. Level proposals retain the chosen pitch at both
trajectory endpoints using each endpoint's own height and target distance;
intermediate orientations still follow the actual interpolated path.
Peers use this gaze variation when `target_variation > 0`; explicit shared-target
policies and handheld orientation increments retain their contracts.

Metrics export `camera_eye_height_m` above the local floor and actual runtime
`camera_pitch_degrees`, including explicit handheld orientations. The trajectory
distributions `camera_path_eye_height_m` and `camera_path_pitch_degrees` sample 33
normalized times per view, including endpoints. Each view also records
`camera_near_horizontal_path_fraction`, the fraction of those samples within ±5°
of horizontal. This distinguishes initial framing from sustained level motion.

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

An optional `multiview.graph: "connected"` policy permits a stable spanning
graph of shared surfaces instead of requiring every peer to overlap camera 0.
Each edge must pass both directed overlap thresholds at all five qualification
times; all camera-clearance, spacing and trajectory constraints still apply.
This can extend larger reconstruction rigs across a room without admitting
fully disconnected views. The default `"reference"` policy retains the anchored
group. Connected graphs and explicit reference-pair overlap mixtures are mutually
exclusive. The editor exposes the graph choice under advanced camera controls.
If the bounded connected search exhausts a cramped room, it tries the stricter
reference-star search with the same settings. A star satisfies the connected
contract; this recovery is counted in `connected_reference_recoveries` and never
weakens overlap, spacing or swept clearance.

`indoor_validate --labels --co-visibility` records `reconstruction_graphs` for
every timestep: directed pixel fractions, components, degree and visibility
multiplicity. `primary_geometry` reports these separately for the exact primary
room, together with metric baselines and shared-pixel triangulation angle
quantiles. Exterior/sky overlap therefore remains distinguishable from room
coverage. These are measured diagnostics; the tool retains hard or disconnected
cases rather than filtering them by an image or model score. Its persistent
`qualification_run.json` records requested controls, build provenance, completion
or failure, including runs that never reached rendering.

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

`principal_point_jitter: [0.12, 0.12]` samples independent horizontal/vertical
optical-center offsets up to 12% of image width/height. Both amplitudes default
to zero and accept values in `[0, 0.2]`. The sampled offset is recorded per camera
and stays fixed along its trajectory. This changes the actual pinhole projection,
frustum, placement visibility, schematic and exported K. It does not crop or
shift an already-rendered image. Portrait and landscape capture use the same
pixel-center convention. Lens distortion and rolling shutter remain unsupported.

For example, `{"principal_point_jitter":[0.12,0.12],"multiview":{"graph":"connected","target_variation":0.8}}`
requests both extensions. Old centered-pinhole configurations retain their
projection and random-stream behavior.

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

Pinhole remains the default. Optional [calibrated fisheye cameras](fisheye_cameras.md)
use a versioned angular polynomial, lens-aware RGB and numeric targets, mixed-lens
flow/co-visibility, and per-view tensor model metadata. Enable them with
`"lenses":{"probability":0.5}` in the indoor camera program. Angular depth is
ray range; existing pinhole depth stays axial. Consumers must use the exported
model and K rather than infer a fisheye from the legacy `fovy` field.
Rolling shutter and rectified derived outputs remain future work in the
[fisheye roadmap](fisheye_lens_roadmap.md).

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

The default recipe renders two matched room seeds for fourteen camera,
aspect, lighting, exposure and material settings: 28 scene variants and 120
camera/time images. Thirteen settings use two cameras; the connected-graph
setting uses four. Every setting has two timestamps. It writes `recipe.json`, `receipt.json`, logs, per-scene
manifests/calibration, distributions and an HTML annotation gallery. `--recipe`
accepts a versioned JSON recipe to expand the seed count or factor ranges. Every
case, including failures, is retained; output folders are immutable. The receipt
binds artifact hashes, actual executable hash, capture-engine/source identity,
build target, feature flags and locked package versions (including optional
packages). It reports directed pixel-overlap/baseline/angle distributions,
per-view self-reprojection errors, family counts and exact RGB duplicates.
Primary-room graph components and pair triangulation summaries are reported
separately from whole-image overlap, so exterior geometry cannot conceal a
disconnected reconstruction region.

The regular publication `refresh` also builds and verifies this canonical sweep,
caches it only for the same source identity and recipe, and installs its receipt
and gallery with the page and paper in one transaction. `verify` checks that
bundle offline. The paper's factor table is calculated from the same receipt.

This bounded factor sweep isolates renderer factors with people disabled. Sensor
operators have separate export regressions. Neither proves transfer, perceptual
uniqueness, a fitted training mixture or photographic realism. Choose/finalize
mixtures using independent synthetic validation and real downstream evaluations;
matching a reused real-image histogram is insufficient evidence.
