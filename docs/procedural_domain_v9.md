# Generator 9: continuous furnishing and captured distribution evidence

Generator 9 adds continuously sampled workstation arrangements, recessed wall
niches, plant morphology and camera positions. The local qualification covers
**12,048 CPU-audited rooms and 144 rendered rooms / 576 views**. These establish
bounded constraint and annotation checks, with measured diversity and quality
gaps; they do not establish 10M-sample learning utility or photographic realism.

Open the [capture dashboard](evidence/domain9/consecutive/capture_dashboard.svg),
[placement and trajectory heatmaps](evidence/domain9/consecutive/capture_placement.svg),
or [all capture contact sheets](evidence/domain9/README.md). Machine-readable
[capture distributions](evidence/domain9/consecutive/capture_distribution.json)
and [scene](evidence/domain9/consecutive/captured_scenes.csv) /
[view](evidence/domain9/consecutive/captured_views.csv) CSVs describe the rendered
cohort, separately from the larger CPU audit.

## Changes to the generator

Each functional zone has a serialized furnishing field: offset, row stagger,
curvature, angular fan, local jitter, occupancy gradient and opposing-workstation
probability. Workstations sample that field rather than a fixed regular grid.
Chair offsets rotate with their desk, retaining their interaction geometry.
Conference/lounge groups use the field's offset; the workstation-specific
coefficients apply to desk zones. Collision, portal and support constraints still
condition the accepted layouts. Metrics for sampled coefficients should not be
interpreted as a count of independently expressed visual dimensions.

Classic-style recessed wall openings vary in width, position, height, depth and
shelf pitch. Wall geometry and wall-mounted object exclusion use the same aperture
definition, so a whiteboard cannot be supported by the empty recess. Other styles
retain their existing architectural programs. The shell and partitions remain
orthogonal; this change does not add arbitrary building footprints.

Plants sample pot proportions/taper, leaf density and phyllotactic angle independently
of species. Individual blades vary their silhouette exponent, bend, twist and
asymmetry. Species retain growth constraints and pots. Frond counts are capped to
retain the existing mesh budget; all six species pass topology/bounds checks over
twelve seeds each, including a check that their mesh topology actually varies.

Camera proposals mix free interior positions (60%) with perimeter proposals (40%)
whose wall inset varies from 0.65 to 1.8 m. Full trajectory clearance and near-view
checks remain active. Previously, almost every accepted start was on a fixed wall
rail. On the same 10,000 room manifests and 20,000 cameras:

| CPU camera diagnostic | Previous camera policy | New camera policy |
| --- | ---: | ---: |
| Median distance to nearest outer wall | 0.85 m | 1.43 m |
| Starts more than 1.8 m from every outer wall | 0.005% | 30.245% |
| Occupied cells in normalized 20×20 start-position grid | 254 / 400 | 396 / 400 |

This is a [matched CPU camera comparison](evidence/domain9/camera_policy_comparison.json),
not a paired RGB ablation. Cell occupancy measures coverage at one resolution,
not uniformity or useful visual content.

Generator/capture identities are now `9` / `capture-v13`, with metric schema `6`.
Keep those identities with dataset shards; the same seed produces different pixels
from previous generators. Existing scene types remain available. No additional
downloaded assets or motion-model initialization are introduced by these changes.

## Cohorts and measured results

The consecutive cohort used seeds **100000–100127**, furniture density **0.65**,
human density **0.25**, two cameras and two trajectory endpoints per scene,
at **480×360**. Its CPU audit covered seeds **100000–109999**. The stress cohort
selected **16 strata** from seeds **200000–202047**, using densities **0.95 / 0.55**
and **640×480**. It deliberately balances categories and is not a population
estimate. No selected scenes or views were discarded or replaced after rendering.

Both cohorts used native Auto quality, shadows, SSAO, specular transmission and
diffuse GI with **64 rays per probe** (below the normal 256-ray default), on an
NVIDIA RTX PRO 6000 Blackwell / Vulkan. People are static AnnyBody-based meshes;
this cohort does not evaluate ARDY motion or optical flow. Full CLI commands and
artifact/source hashes are retained in [provenance](evidence/domain9/provenance.json).

| Quantity | Consecutive 128 rooms | Denser 16-room stress cohort |
| --- | ---: | ---: |
| Completed views | 512 / 512 | 64 / 64 |
| Main-room chairs, min / median / max | 0 / 2 / 9 | 0 / 3 / 9 |
| Main-room plants, min / median / max | 0 / 2 / 10 | 0 / 3 / 8 |
| Main-room people, min / median / max | 0 / 1 / 5 | 1 / 3 / 7 |
| Functional zones, min / median / max | 1 / 2 / 10 | 1 / 5 / 10 |
| Rooms containing people | 79 | 16 |
| Those rooms with no person pixels in any view | 23 / 79 | 1 / 16 |
| Rooms with a view containing at most two semantic classes | 6 / 128 | 3 / 16 |
| Rooms with a view dominated by one class (>90% pixels) | 4 / 128 | 3 / 16 |
| Rooms with every view flagged | 0 | 0 |

The consecutive captured rooms span **38.36–262.66 m²**. Captured vertical FOV
spans **28.12–106.65°**, and camera height across both endpoints spans
**0.723–3.313 m**. The endpoint range includes vertical trajectory motion beyond
the initial-position sampling range. Projection remains a centered pinhole;
intrinsic variation here is focal length/FOV, not distortion or principal-point
variation. CSVs include focal lengths, principal point, position, forward direction
and trajectory endpoint displacement. The population export additionally includes
curve length, roll, yaw/pitch and sampled controls.

The 10,000-room CPU audit has **zero invalid manifests, placements or swept camera
paths**. It covers room areas **38.00–270.00 m²**, heights **2.65–4.80 m**, **1–11**
zones and **0–23** main-room chairs. The main-room chair median is **2**, with
**10.54%** of scenes containing no chairs. Counts include zero-instance scenes.
The [full metrics](evidence/domain9/consecutive/metrics.json) include object counts,
material scale/roughness, glazing recipes, human pose parameters, lighting,
architecture, plant morphology and joint histograms. Sampling ranges already
introduced in generator 8 remain active; this turn does not expand all of them.

## Quality, annotations and uncertainty

Review flags are exported without filtering: >95% pixels below linear luminance
0.002, >10% above 0.99, >90% of one semantic class, or at most two semantic classes.
Neither cohort triggered the first two thresholds, but **this does not mean all
images are well exposed**: the darkest consecutive view has **90.5%** dark pixels.
Its mean linear luminance is **0.00136**; the cohort's median is **0.06446**.
The [darkest views](evidence/domain9/consecutive/darkest_views.jpg) preserve that tail.

The at-most-two-class scene rate is **4.69%**, with a scene-level Wilson 95% interval
of **2.17–9.85%**. For zero capture/annotation failures in 128 scenes, the one-sided
95% upper failure-rate bound is still **2.31%**, assuming consecutive seeds behave
as independent generator draws. Views within a scene are correlated and are not
treated as independent trials. Stress-cohort statistics have no population
confidence intervals. These sample sizes cannot certify rare failures at 10M scale.

Two diagnostic attempts initially stopped on an old validator requiring at least
three semantic classes. Inspection found legitimate close views containing only
floor/wall classes. The validator now accepts any nonempty known palette and still
checks every pixel; low semantic richness is reported as a quality flag. Both final
cohorts were then run completely from their declared selections. Incomplete attempts
remain local diagnostics and are excluded from the qualification report.

All 576 captures passed the RGB signal, semantic palette, camera pose and geometric
annotation gates. Worst per-view p99 depth/position disagreement was
**0.00000477 m**; worst p99 reprojection disagreement was **0.00287 pixels**.
Recorded camera poses matched exactly. Native geometry annotations are direct
float32; these metrics do not validate physical lighting or material realism.
Glass remains an opaque first surface for semantic/geometric annotations, so
visible RGB content behind glass can differ from semantic visibility.

Visual repetition diagnostics compare start frames from different scenes, excluding
temporal pairs and within-scene views. Among **256** consecutive-cohort start views,
there were **no exact decoded-RGB duplicates**. The nearest cross-scene 64-bit
difference-hash distance had median **19 bits**, minimum **9**; inspect the
[closest pairs](evidence/domain9/consecutive/closest_pairs.jpg). This inexpensive
image hash is sensitive to composition and is not evidence of semantic novelty.

Contact sheets, exposure tails, closest pairs and sampled full-resolution views
were inspected. Rooms and camera coverage vary, but sparse furnishings, repeated
orthogonal shells, exaggerated procedural grain, blurred glass/indirect lighting,
and visibly synthetic hair/clothing remain evident. People being present but
unseen is a substantial visibility gap for human-centric training. Room-purpose
density, visibility-conditioned sampling and real-image/downstream evaluation
remain necessary before accepting this distribution for 10M-sample pretraining.
No claim of photographic realism, matched real-world statistics, or unlimited
process memory stability is made here.

## Local checks and reproduction

**90 library tests passed, 3 explicitly ignored**; **29 Python report tests passed**.
Native build, strict Clippy, root-package formatting and the Wasm viewer compile
check passed. This is compile coverage for the new shared geometry on Wasm, not
a new browser-render qualification. Whole dependency-tree formatting also sees
unrelated in-progress changes in the sibling `burn_human` checkout; those were
left untouched.

Consecutive capture wall time was **0.50–2.66 s per four-view scene**, median
**0.86 s**, including generation, warmup and IO. Stress-cohort median was **1.68 s**.
CPU compilation and other GPU work overlapped parts of this run, so these are not
isolated throughput or utilization benchmarks.

```sh
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo test --lib -j12
cargo clippy --lib --bins --tests -j12 -- -D warnings
cargo build --bin indoor_validate -j12
target/debug/indoor_validate --seed 100000 --audit-seeds 10000 --renders 128 \
  --cameras 2 --width 480 --height 360 --playback-steps 2 \
  --density .65 --human-density .25 --labels --no-raw --gi-rays 64 \
  --output out/domain9_qualified
target/debug/indoor_validate --seed 200000 --audit-seeds 2048 --renders 16 --stratified \
  --cameras 2 --width 640 --height 480 --playback-steps 2 \
  --density .95 --human-density .55 --labels --no-raw --gi-rays 64 \
  --output out/domain9_stress
python scripts/indoor_capture_distribution.py out/domain9_qualified
python scripts/indoor_capture_distribution.py out/domain9_stress
cargo check --target wasm32-unknown-unknown --no-default-features --features web --bin viewer
```

The report script requires NumPy, Pillow and Matplotlib. It verifies completion,
run IDs, selected seeds, dimensions, generator/configuration coherence and semantic
pixel totals before producing CSVs, distributions, uncertainty, heatmaps and contact
sheets. Full images, manifests, per-instance CSVs and diagnostic logs remain in
the ignored local `out/domain9_*` directories. Nothing was committed or published.
