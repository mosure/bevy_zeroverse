# Local generator-v6 program review

This local revision uses generator 6 and capture-v8. It has not been committed,
published or qualified by CI. Category counts are not evidence of suitability for
10 million training samples or of photographic realism.

## What is sampled

`program.rs` recursively subdivides the usable rectangular envelope. Split locations,
leaf proportions, portal position/width/height, wall thickness, glazed area, furniture
orientation, aisle width, workstation dimensions, seat pitch and occupancy are sampled.
Portals constrain subsequent splits, furnishings, people and camera clearance. Headers
participate in camera collision separately from floor-level navigation. Each leaf has
an activity and independently sized furniture groups, with repair of underfilled rooms.
The outer envelope is still rectangular; there are no stairs, multi-level buildings,
curved structural walls or general arbitrary polygon floor plans in this revision.

Chairs compose back construction and armrests independently of the frame category,
with continuous shell curvature, taper, thickness, rake, spindle pitch and arm height.
Laptop programs sample hinge angle, aspect, chassis/bezel dimensions, keyboard coverage
and trackpad fraction. Screens point toward their intended chair, and chairs record the
work surface they serve. Both relationships are validated and exported in manifests.

`materials/program.rs` samples role-specific PBR recipes. Correlated periodic layers
supply wood grain, woven fibres, mineral variation, panel seams, wear and roughness.
Relief is expressed in metres and differentiated using metres per texel. Colour mipmaps
are filtered in linear space; normal mipmaps are renormalized. Objects have independent
metric UV phases. Maps remain bounded at 256 squared pixels and are shared across the
scene. These are periodic material models, not scanned materials; close-up texture
fidelity and uniformity across unseen parameter combinations remain limitations.

People retain the full AnnyBody phenotype surface and all skinning influences. Activity
labels constrain continuously sampled arm/foot targets, fixed-length limb IK, stance,
lean and torso rotation. Appearance samples pigmentation, cloth colours/roughness,
garment ease/folds and combed hair detail. These are static retargeted bodies and offset
garments, not cloth dynamics, hand-object manipulation or a spectral skin/hair model.
The retargeter now anchors the torso at the shoulder girdle, preserves the rest skull's
orientation in the sampled torso frame, and attaches facial details in that frame.
Gravity-biased elbow swivel avoids the earlier flared-arm poses. Face-corner UV indices
are retained across Anny's texture seams; mesh vertex indices are not texture indices.

## Performance and ownership

The native compute job prepares the layout, materials, images, Anny surfaces, mesh
normals/tangents and GI acceleration data. Assets reserve handles from the live Bevy
allocator and are inserted when the completed scene replaces the previous scene.
Configuration changes cancel stale preparation. Quality, rotation and all GI settings
participate in the preparation key. Capture waits for preparation and render readiness.
Interactive native preparation uses the existing CPU transport integrator on a
separate bounded worker, while headless generation retains the full GPU transport dispatch.
They use the same transport model and sample budget but independent random sequences.
Interactive geometry is published before that lighting finishes; a complete irradiance
volume is attached when ready. Automatic regeneration and dataset capture wait for it.
Manual scene changes can cancel a pending volume, and an old job cannot attach to a new room.
This is a responsiveness/latency tradeoff, not a faster GPU integrator. A small-dispatch
GPU experiment was rejected after it increased frame latency and reduced throughput.
The browser yields to the event loop between material and object jobs. Wasm does not
run these jobs on additional threads, and a single object job is not preemptible.

There is no sample-count-sized scene cache. The existing human geometry cache is
bounded at 24 people. This ownership design and bounded checks do not establish
unlimited-process memory stability.

## Distribution evidence

`indoor_validate` exports existing object counts, camera calibration, path and placement
heatmaps, plus continuous zone/portal/material/furniture/pose/appearance distributions.
Joint histograms expose zone-count/chair-count and wood-roughness/repeat correlations.
A constant-memory HyperLogLog sketch estimates distinct coarse spatial signatures,
ignoring seeds and colours (4096 registers; about 1.63 percent relative standard error).
That estimate is a spatial diagnostic, not a measure of semantic training diversity.
Captured manifests contain the complete scene program and resolved joints for replay.

Constraint rejection changes the accepted distribution. Marginals, empty-chair counts,
rejection counts and joint histograms must be reviewed for the intended dataset. The
current bounded audit does not extrapolate an effective 10M-sample training corpus.

## Physical reference

The comparison target remains Cycles rendering the same exported geometry, textures,
lights and cameras. Raw independent repeats estimate reference sampling noise. Display
exposure is fixed by the recorded camera; no fitted alignment or exposure is permitted.
The native probe-volume GI, analytic local light proxies and environment reflections
remain approximations to path-traced transport. Passing mesh/annotation checks alone
does not qualify those approximations as physically accurate.

The design uses compositional parameters and placement constraints, consistent with
the general procedural-generation direction described by
[Infinigen Indoors](https://openaccess.thecvf.com/content/CVPR2024/papers/Raistrick_Infinigen_Indoors_Photorealistic_Indoor_Scenes_using_Procedural_Generation_CVPR_2024_paper.pdf).
It does not claim that system's asset coverage, quality or evaluation results.

## Local validation, 2026-09-26

The native cohort uses seeds 0..2047, default density 0.65 and human density 0.25.
Twelve strata-selected scenes were captured at 800x600, two cameras and three
trajectory times each. Selection preceded rendering; poor-looking images were
not removed. See the [72-view contact sheet](procedural_indoor/contact_generator6.jpg),
[render summary](procedural_indoor/render_summary_program6.json),
[distribution audit](procedural_indoor/distribution_program6.json) and
[complete metrics](procedural_indoor/metrics_program6.json).

| Check | Result |
| --- | --- |
| Layout audit | 2,048 seeds; zero invalid; 1,806 occupied strata |
| Sampled spatial program | 1..7 zones; 14..82 main-room object instances |
| Main-room chairs | 1..15 per scene, mean 4.91 |
| Main-room plants | 0..9 per scene; 185 scenes without a main-room plant |
| Coarse spatial signatures | HLL estimate 2,017 of 2,048, approximately 1.63% relative standard error |
| Native RGB and labels | 72 views; maximum per-view p99 reprojection error 0.00267 pixels |
| Geometry annotations | Direct float32; maximum per-view p99 depth/position disagreement 2.87 micrometres |
| Rust library tests | 78 passed; 3 explicitly ignored diagnostic/export tests |
| Native GI integration | Passed GPU/CPU oracle check, background-lighting capture readiness and GI on/off ablation |
| Python reference tests | 11 passed, including partial-edge radiance conservation and source identity rejection |
| Browser | Auto and Portable; inspector enabled; seed 115 then 116; both passed with no GPU errors |
| Compilation | Native viewer/tools, Wasm viewer and strict Clippy passed |

The browser harness waits for the requested seed's generation event before measuring
post-regeneration frames. Frame submissions alone are insufficient when the previous
scene remains visible during preparation. The [browser report](procedural_indoor/web_program6.json)
records the actual Wasm hash, adapter, errors and fallback warnings.

On the RTX PRO 6000 Blackwell / Vulkan 610.43.02, a 30-second interactive stress
run used seed 31, human density 1.0 and a 4-second automatic regeneration interval.
The interval waits for pending lighting; it does not cancel unfinished bakes.

| CPU timing | Before separating interactive lighting | Final |
| --- | --- | --- |
| First render submission | 1,350 ms | 1,280 ms |
| First furnished-scene event | 1,978 ms | 2,089 ms |
| Worst frame gap after 5 seconds | 656 ms | 53 ms |
| Frame-gap p95 | 17.4 ms | 16.8 ms |
| Frame-gap p99 | 22.8 ms | 21.1 ms |

These are CPU intervals from [the viewer profile](procedural_indoor/viewer_profile_program6.json),
not GPU timestamp measurements. First submission does not prove loaded pixels, and
the furnished-scene event now precedes final indirect lighting. Dense-room lighting
can take several seconds; headless capture retains the faster full GPU dispatch.
The final stress run still had one 252 ms startup frame gap and a 45 ms maximum main
schedule interval. This demonstrates a substantial reduction in the measured recurring
hitch, not a guarantee of hitch-free behavior on all devices, seeds or browser builds.

## Matched physical result

Prespecified seeds 0 (evening), 6 (daylight) and 115 (overcast) each use two cameras at
480x360. Each reference was rendered twice with independent seeds, 2,048 samples,
12 bounces, OptiX and no denoising. The pinned passing Blender 5.2.2 LTS radiometry
control was reused. No exposure was fitted. All six geometry/camera alignment checks
passed, and all six independent repeats passed the existing coarse convergence gate.

| Seed/view | Native vs Cycles RGB relative MAE, 32-pixel blocks | Independent reference disagreement, same blocks |
| --- | --- | --- |
| 0 / 0 | 23.64% | 0.28% |
| 0 / 1 | 18.36% | 0.18% |
| 6 / 0 | 8.08% | 0.33% |
| 6 / 1 | 17.26% | 3.76% |
| 115 / 0 | 20.33% | 4.60% |
| 115 / 1 | 12.57% | 3.46% |

Reports: [evening](procedural_indoor/cycles_program6_seed0.json),
[daylight](procedural_indoor/cycles_program6_seed6.json),
[overcast](procedural_indoor/cycles_program6_seed115.json).
The comparator now includes partial edge blocks, weighted by their actual pixel area;
it previously omitted the 32-pixel measurement for these image dimensions. Thresholds
were unchanged. Whole-image mean agreement and the convergence gate use all pixels.
Raw pixel MAE is 21.0..38.9%, with appreciable reference pixel noise; coarse convergence
does not make those individual pixels converged ground truth.

These results do **not** establish physical accuracy or photographic realism. Visual
inspection still shows limited skin/hair/garment detail, uniform-looking surfaces at
some scales, simplified clutter, and limited architectural scope. Window reflection,
ceiling and indirect-light differences remain visible in the matched renders. The
procedural space is broader and has measurable continuous parameters, but suitability
for 10M-sample pretraining requires a broader rendered audit and downstream evaluation.

## Reproduction

```sh
cargo test --lib -j2
cargo clippy --lib --bins --tests -j2 -- -D warnings
cargo build --target wasm32-unknown-unknown --no-default-features --features web --bin viewer -j2
cargo test --test procedural_indoor_gi -j2 -- --ignored --nocapture
python -m unittest discover -s scripts -p test_indoor_cycles.py

target/debug/indoor_validate --audit-seeds 2048 --renders 12 --stratified \
  --cameras 2 --width 800 --height 600 --labels --no-raw --playback-steps 3 \
  --output out/program_review_new_run
python scripts/indoor_report.py out/program_review_new_run --figures

ZEROVERSE_PROFILE_SECONDS=30 ZEROVERSE_PROFILE_OUTPUT=out/viewer_program_profile.json \
  target/debug/viewer_profile --scene-type procedural_indoor --indoor-seed 31 \
  --indoor-human-density 1 --regenerate-ms 4000
```

The physical runs are retained in `out/program6_physical`; use the commands in
[the physical-reference guide](procedural_indoor_physical.md) with seeds 0, 6 and 115,
`--samples 2048` and a second run with `--seed-offset 104729`. The final browser run is
`out/program6_web_final`. No commits, CI runs or publication were performed.
