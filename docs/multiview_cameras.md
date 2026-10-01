# Multi-view camera sampling

Procedural indoor cameras sample a connected set of views for reconstruction.
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

- Every additional camera must overlap camera 0 in both directions, creating a
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
