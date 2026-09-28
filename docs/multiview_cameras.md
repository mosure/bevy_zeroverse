# Multi-view camera sampling

Procedural indoor cameras previously passed visibility tests independently. They
could occupy the same room yet look at different furniture, or see opposite faces
of the same object. Pointing at one target, intersecting frusta, and sharing a
semantic class do not establish visible geometric correspondences.

The `indoor_camera.multiview` policy (default since generator 18) samples a connected camera set for
reconstruction. Existing object, Cornell and legacy room modes retain their
samplers. Object/Cornell views already target a central region, but do not enforce
surface overlap; the new constraint currently applies to `procedural-indoor`.

## Configuration

Pass this JSON through the viewer or generation CLI's `--indoor-camera`, Python's
`BevyZeroverseConfig.indoor_camera`, or the existing dataset configuration:

```json
{
  "primary_room": true,
  "multiview": {
    "min_overlap": 0.35,
    "min_baseline": 0.25,
    "max_baseline": 3.0
  }
}
```

All three nested fields have the defaults above: `"multiview": {}` enables the
policy. Generator 18 enables this policy by default. Set `"multiview": null` explicitly for independent sampling. The dataset CLI defaults to four cameras when `--scene-type procedural-indoor` is selected; explicit camera counts take precedence. Viewer camera counts remain explicit so an editor-only review does not render additional unseen views.
`min_overlap` is in `[0,1]`; baseline bounds are metres and must satisfy
`0 < min_baseline <= max_baseline <= 100`. A zero overlap threshold retains the
nearby-camera proposal and baseline constraints; use `null` to disable grouping.

The inspector exposes **Shared geometry across views**, minimum estimated overlap
and minimum/maximum baseline under **Capture camera paths**. Edits take effect on
explicit regeneration, preserving slider behavior.

- Every additional camera must overlap **camera 0**, in both directions. This
  creates a connected reference graph, with O(N) edges. For N>2, overlap between
  arbitrary nonreference cameras is not required.
- The score is the fraction of all sampled image pixels whose first geometric
  surface is also inside the other frustum and unoccluded there. The lesser of
  the two directional scores must pass. This avoids a narrow view inside a wide
  view passing solely because its own pixels are covered.
- Full capture aspect ratio, per-camera focal length and orientation enter the
  test. It samples 13×9 rays at normalized times `0, .25, .5, .75, 1`. Near/far are
  0.1/50 m, matching the indoor camera. Glazing occludes annotation surfaces.
- Cameras retain independently sampled focal lengths, different targets/rolls and
  nonzero translation. Baseline is Euclidean camera-center separation, not path
  length. Multi-view trajectories use translated versions of the reference path;
  each complete path still passes swept collision/primary-room checks.
- Proposals sample baseline logarithmically and random offset directions. Failed
  sets retry up to eight reference views, each allowing 384 proposals per extra
  camera. An impossible request fails clearly; thresholds are never weakened.
- The CPU sampler loads no models and submits no GPU work. Reference ray samples
  are reused across candidate views. Disabled sampling follows the original
  camera RNG sequence. Furniture, materials and humans use separate seed streams.

A larger overlap threshold concentrates views on the same surfaces, reducing the
proportion of unmatchable training pixels. It can also narrow accepted baselines
and focal-length differences. Use baseline bounds to control the parallax regime,
then inspect overlap **and** triangulation-angle distributions. Very small
baselines can provide many matches while contributing little depth information.
The measured 35% policy below is a practical starting point, not a demonstrated
optimum for a downstream model. Mixed easy/hard training can use separate
configuration cohorts instead of forcing every scene to nearly identical views.

For fixed cameras set `path_length_min` and `path_length_max` to zero. These bounds
control each camera's temporal travel, independently of the multi-view baseline.
A single-camera request remains valid; it has no pairwise constraint.

Rust callers can use `IndoorManifest::resample_cameras(count, settings, aspect)`;
failure preserves the prior camera set. The manifest records settings, aspect
ratio, intrinsics and complete trajectories. `camera_overlap()` returns estimated
reference edges at the five check times. Rotation augmentation preserves scores.

## Measured overlap and dataset qualification

These are **proxy geometry constraints**, not exact pixel guarantees. The sampler
uses scene collision envelopes and structured chair parts, not every rendered
triangle. Thin geometry, holes, moved people and times between check samples can
change actual overlap. Textureless surfaces and transmitted glass RGB also differ
from geometrically matchable content.

CPU dataset metrics now include `camera_overlap.csv`, bidirectional overlap,
baseline and triangulation-angle distributions in `metrics.json`, and the exact
camera policy. The audit, metrics and live scene use the same policy/aspect.

For actual visible overlap, capture depth and run the independent report:

```sh
cargo run --bin indoor_validate -- \
  --seed 0 --audit-seeds 128 --renders 8 --cameras 2 \
  --width 320 --height 240 --human-density .25 --playback-steps 3 --labels \
  --indoor-camera '{"multiview":{"min_overlap":0.35,"min_baseline":0.25,"max_baseline":3.0}}' \
  --output out/multiview
python scripts/indoor_multiview_report.py out/multiview --previews \
  --require-min-overlap 0.3
```

The Python report needs NumPy (and Pillow for previews). Retain raw captures; do
not pass `--no-raw`. It reprojects each source depth pixel using exported camera
calibration, tests the target's rendered depth, and evaluates both directions at
each captured time. It writes `rendered_camera_overlap.csv` and
`multiview_report.json`, including baseline/parallax, per-room worst overlap,
misses against the requested proxy threshold and input hashes. Preview pixels
outside the shared region are darkened. Nearest target pixels use a documented
`0.01 m + 0.002*z` depth tolerance. The optional measured threshold exits nonzero
on any failing pair **after retaining the reports**; it does not silently filter
rooms or regenerate them. Use this to qualify generator settings and use the CSV
scores for pair selection. Report limitations include the sampling/tolerance and
annotation-opaque glass; these measurements do not prove learning utility.

## Local evaluation, 2026-09-27

The [retained report](evidence/multiview/summary.json) compares identical seeds,
furniture, people and lighting under three camera policies. Each CPU cohort has
128 consecutive seeds, two cameras, density 0.65 and human density 0.25. All
384 scene-policy combinations passed. Actual captures cover seeds 0–7 for each
policy at 320×240 and times 0/.5/1: **144 rendered views, 72 synchronized pairs**.
Baked GI was disabled for this geometry experiment; lights/shadows stayed enabled.

| Policy | Mean rendered overlap | Worst rendered overlap | Rooms with a pair below 10% | Mean baseline | Mean triangulation angle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 14.0% | 0.9% | 5/8 | 3.61 m | 55.1° |
| Minimum estimate 35% | 59.5% | 39.0% | 0/8 | 0.84 m | 9.0° |
| Minimum estimate 60% | 69.4% | 56.4% | 0/8 | 0.71 m | 7.2° |

The 60% policy missed its requested threshold in **3/24 measured pairs, all in
one room**. These remain in the report. Larger independent-camera angles above
come from their small shared regions, not more useful correspondence coverage.

CPU generation **plus layout/camera validation** took 0.18/0.98/1.62 seconds per
128-scene cohort respectively, or about 1.4/7.7/12.7 ms per scene on this machine.
These are bounded local audit timings, not isolated sampler or 10M-scene throughput
benchmarks. No rendered scenes were discarded. The small rendered sample does not
establish a population success rate or downstream training improvement.

![Same scenes with independent, 35% and 60% overlap policies; both cameras at time zero](evidence/multiview/comparison.jpg)

All ten camera tests passed, covering occlusion, FOV/aspect, baseline bounds,
trajectories, four-camera connectivity, reproducibility, manifest round trips,
unchanged independent sampling and atomic failure. The full library passed
125 tests (three explicit long-running tests ignored). Strict workspace/all-target
Clippy with motion/embedding features and the Wasm viewer/motion check passed.
The three Python analytic tests check known stereo-plane overlap, occlusion,
invalid calibration, opposite views and asymmetric focal lengths. No browser
runtime or downstream training experiment was performed in this review.

A separate real `zeroverse_gen` smoke run exported two samples, two cameras and
three times, with RGB/depth/position in safetensors. Calibration, finite tensors,
policy metadata and baseline bounds survived the export. Its empty-human fixture
at 160×120 had a worst measured overlap of 34.1% for a requested 35% estimate;
this independently demonstrates why the rendered threshold should be checked.
The [CLI validation](evidence/multiview/cli_validation.json) retains those six
pairs separately from the matched comparison. All 52 indoor Python report tests
passed. The measured gate accepted the 35% cohort at 30% and rejected the 60%
cohort at 60%, preserving its failing rows.

Recreate the paired review by running the capture command above for three output
directories (omit `--indoor-camera`, use 0.35, then 0.6), adding `--no-gi` to each.
Run the report for every directory, then:

```sh
python scripts/indoor_multiview_compare.py \
  out/multiview_review/independent out/multiview_review/shared35 \
  out/multiview_review/shared60 --output out/multiview_comparison
```

This verifies identical scene manifests apart from cameras, capture settings,
resolution and time schedules before producing the comparison.
