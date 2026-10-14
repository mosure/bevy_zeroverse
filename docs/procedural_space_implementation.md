# Procedural space implementation and qualification

This is the local execution record for [the roadmap](procedural_space_roadmap.md).
Matched captures and diagnostic comparisons stay under `out/roadmap_qualification/`.
The subsequent ceiling, exterior-context and floor-detail qualification is under
`out/room_context_expansion/final/`; its evidence and limits are recorded below.
The starting checkout is `91c5c96361970cc38867e2e78f27813fb92d833c`.

## Delivery gates

| Phase | Implementation | Qualification |
| --- | --- | --- |
| Functional diagnostics and light occlusion | Accepted assembly receipts; all Auto ceiling emitters shadowed; local ceiling attachments | Native unit/mesh tests; isolated thirteenth ceiling emitter has zero measured light beneath an opaque deck |
| Transactional furniture assemblies | Bounded whole meeting-group placement, explicit seats-at relationships and coordinated finishes | Functional, seating/contact and density-extreme tests passed |
| Reflection locality, glazing and material identity | Up to three box-projected zone probes; installation-coherent glass; metric filtered frosting; opaque frit | CPU/GPU map and annotation parity; 672 controlled window views; two independently repeated Cycles references, with remaining transport error reported below |
| Architectural topology and window construction | Additional polygonal returns, closed soffits, exact clipped regions/portal graph, unequal panes and articulated sashes | 1,024 actual mesh audits passed; roof/ray, fit and historical polygon regressions passed |
| Plants, garments, hair and interaction states | Age-correlated growth; rounded body-derived garment fit/ease; curved parts and crown guides; fitted tabletop gestures | Anatomy, contact and envelope tests; controlled body/hair/plant galleries; 7/8 generated motions admitted, with one rejected clip retained as static fallback |
| Camera routes and calibration | Reflected route endpoints; sampling/rejection diagnostics; bounded free-space visibility recovery | Native calibration/replay; 2,879/2,880 camera requests admitted across counts 1–12 and three aspects; the rejected request remains reported |
| Distribution, rendering and consumer performance | Larger Rust embedding analysis; multi-time study viewer; separate, verified cohort reporting | 64-room matched study, 512 rendered rooms, 10,000 manifests, completed WGPU/CUDA consumers and a 96-step optimizer probe |

These gates refer to implemented and measured behavior, not the number of
available parameters. Optional research directions in the roadmap retain their
stated evidence requirements. Existing scene modes, physical units, semantic
ownership, geometry annotations, deterministic replay and full-quality defaults
remain compatibility requirements.

## Measurement conditions

The device is an NVIDIA RTX PRO 6000 Blackwell (Vulkan, driver 610.43.02).
An independent workload initially occupied about 81 GiB and 96% device activity;
it finished without intervention before the final performance runs. The final
40/160-room timing runs therefore do not represent concurrent large-model
training. Their board activity averaged approximately 10–14%, with 1.8–4.5 GiB
resident. A separately specified optimizer probe measures concurrent generation
and training. Raw timestamps, process/board telemetry, quality settings, package
versions and source identities accompany the runs. CPU-only preparation,
completed capture, tensor transport and optimizer throughput are distinct metrics.

## Implemented contracts

The current local generator identity is **40**, capture engine **v58**, metrics
schema **18** and semantic taxonomy **4**. Crate versions remain unchanged; this
work has not been committed, pushed or published. The broader roadmap campaign
below was frozen at generator 38 / capture-v56. Those measurements do not
automatically qualify the subsequent geometry additions. The current object
construction campaign is recorded in [object geometry qualification](object_geometry_qualification.md).

- `layout/assemblies.rs` records requested and accepted capacity and bounded
  rejection causes. A meeting proposal commits its surface and usable seats
  together. It does not pass by deleting unrelated furniture or relaxing mesh,
  doorway or operational clearance. `support` and `interaction_target` remain
  the authoritative attachment relationships. Shared finish identity is separate
  from geometry identity.
- `materials/reflections.rs` bounds reflection residency by the largest three
  admissible zones, with a global fallback. These are static, box-projected
  radiance approximations, not dynamic planar reflections. WASM retains its
  global environment because Bevy's localized probes require texture arrays.
  Local probe transforms inherit room rotation once; a regression exercises
  Bevy's actual extraction matrix on a rotated room. Camera environment maps
  retain their independent world rotation.
- `materials/glazing.rs` samples installation thickness and aligned privacy
  heights, with clear, uniform, band, gradient, dot, stripe and opaque-frit
  treatments. Roughness fields are baked once in metric coordinates, including
  their full roughness-moment mip chain. Frosting is not a white alpha overlay.
  Pane, sash, handle, beads and frit move together for bounded tilt openings;
  shades restrict opening when their sweep would interfere. There is no hidden
  closed pane behind the open sash.
- Through-glass annotations skip both native transmission and the explicitly
  thick blended portable glass. Frames and opaque frit remain occluders.
  This is straight geometric visibility: diffuse frosting and refractive RGB
  do not acquire an exact pixel correspondence merely because a depth is exported.
- `envelope/ceilings.rs` and `construction/ceiling_mesh.rs` share analytic
  headroom with closed ceiling/soffit meshes. Extra re-entrant returns preserve
  the supported neighbor connection. `room_topology()` exports nonoverlapping
  clipped region triangles, primary membership, and realized portal connections;
  the primary reconstruction AABB does not expand to include the exterior.
- Garment ease and waist taper are independent of body phenotype. Curved torso
  sections, clothing/button clearance and feasible tabletop gestures preserve
  the Anny body and uncovered anatomy. Growth maturity correlates branching and
  leaf occupancy; rosettes use fleshy leaf geometry and cacti retain their stem
  construction. Hair guide parts/whorls vary continuously while retaining rooted
  connected tresses.
- Camera recovery preserves the existing baseline, overlap, collision, path
  length and anatomical-visibility predicates. Seed `109429` exposed a narrow
  valid viewpoint missed by person-centered rings; a bounded free-space fallback
  now recovers it, including a three-view request. Ordinary accepted samples do
  not enter this recovery. Search diagnostics distinguish the main reference
  and peer proposals, route adjustment and group restarts; ring/grid fallback
  proposal counts are not included in the main reference counter.

## Qualification protocol and tools

All comparison artifacts are local in `out/roadmap_qualification/`. The Rust
publication tool prohibits matched-study output in the deployable publication
cache. Its study viewer retains camera, annotation and timestep selection, and
embedding neighbor links resolve to the correct captured timestep.

The qualification protocol uses three 512×512 views at two times, with full
Auto quality, RGB, depth, normals, position, semantic labels and co-visibility:

1. A 64-room seed-matched local comparison, with architecture changes explicitly
   declared. Its 3,072 RGB/annotation PNG planes replayed byte-for-byte in the
   frozen final qualification build.
2. 256 consecutive rooms, 128 prespecified factor-stratified rooms from an
   independent 2,048-room population, and 128 held-out consecutive rooms.
3. 10,000 separate manifests with the default three-camera policy; 1,024
   consecutive rooms independently checked using actual constructed meshes.
4. Baselines 0/.25/.5/.75/1 × camera counts 1–12 × aspects .625/1/1.6,
   with all requests and failures retained.
5. Metric material panels, plant silhouettes, clothed body extremes, hair
   close-ups, occupied-table contacts, generated motion, and independent
   Cycles runs with fixed radiometry and a higher-sample repeat.

`bevy_zeroverse_publication audit-cohorts --input <directories...> --output
out/<report>` verifies completed captures, source identities, unique seeds,
geometry/annotation metadata, every delivered image and the numeric membership
masks. It reports each sampling population separately. It does not turn selected
strata into an IID population estimate. Ordinary publication and paired-study
loaders still require consecutive render selections.

The embedding analysis supports up to 1,024 room pairs. A prespecified,
stratified-jitter subset of at most 256 pairs fits the common basis; every room
contributes to projected covariance and room-bootstrap estimates. Fit indices
and bounded effective-rank scope are exported. Nearest neighbors remain exact
within the existing row limits. This is not a ten-million-row approximate index.

## Evidence and limits

The final core suite passed **500 tests**, with 26 explicitly ignored hardware
or diagnostic tests. Separately executed native tests passed WGPU and CUDA
capture parity, retained tensor lifetime, indexed worker replay, aligned shared
allocations, through-glass exterior coverage, exposure isolation, CPU/GPU scene
annotation parity, and 55 GPU material recipes with all maps and mip levels.
Strict native Clippy passed. Hardware Chrome (NVIDIA Blackwell) rendered all
seven viewer modes without shader or device errors.
The publication crate also passed 23 unit tests and 21 contract tests, including
declared architecture changes and exact camera/timestep neighbor links. Its
study verifier checked 6,405 local artifacts. The Cycles export contract passed
14 tests.

The final scheduling refactor additionally replays all **192** RGB/annotation
PNG planes for four rooms, three cameras and two times exactly against the
frozen native qualification build. Cooperative and synchronous reflection maps
and every mip level also agree byte-for-byte. Strict WASM Clippy passes.
After the final room-rotation correction, the same 192 planes replay exactly
again. A separately rotated room passes six-view annotation checks, with worst
per-view P99 reprojection error 0.000368 pixels and depth/position disagreement
3.82e-6 m. Browser runtime receipts identify the tested WASM binaries; the later
local-probe rotation correction affects the native-only probe path, and the
final source also passes static and motion-enabled WASM Clippy.

Remaining dependency notices are recorded rather than suppressed: Burn 0.22
reports a future Rust incompatibility, and the Bevy human adapter clamps eight
source bone influences to four supported rendering influences. Neither is a
new local compiler warning, and neither is evidence of exact eight-weight skinning.

The bounded implementation does not add arbitrary angled partition networks,
curved/multilayer glass, unrestricted roof intersections, a full cloth/hair
simulation, off-center/distorted lenses, rolling shutter or depth of field.
Those roadmap extensions retain their geometry/optical qualification conditions.
People and some material responses remain visibly stylized. No photographic
realism, 10× perceptual-volume, downstream learning improvement, or unlimited
process-lifetime claim follows from these tests. Performance and optical results
below must be read with their finite sample sizes and shared-device contention.

## Rendered distribution and camera coverage

The Rust cohort audit verified **512 rooms / 3,072 views** and the hashes of all
delivered RGB/annotation planes and co-visibility membership masks. The cohorts
remain separate; the factor-stratified sample is not an unbiased population.

| Cohort | Rendered rooms | Minimum semantic classes/view | Valid pixels shared with any peer | Primary-human rooms showing a person |
| --- | ---: | ---: | ---: | ---: |
| Consecutive seeds 0–255 | 256 | 5 | 75.09% | 172/172 |
| Prespecified strata from 2,048 independent rooms | 128 | 5 | 75.22% | 67/67 |
| Held-out consecutive seeds 40000–40127 | 128 | 6 | 74.86% | 90/90 |

No view had fewer than four semantic classes. These person counts establish
class visibility somewhere in the six views of a room, not visibility of every
actor. People in neighboring zones are not required to appear. The worst
per-view P99 self-reprojection error was **0.000683 pixels**, depth/position
disagreement **3.82e-6 m**, and maximum normal-length error **2.39e-7**.
All 512 first views were inspected as contact sheets; not every individual view
was manually inspected at full resolution.

The separate 10,000-manifest population (seeds 100000–109999) had no invalid
default three-camera rooms. It exports requested/accepted assemblies, object
counts, placement heatmaps, topology, glazing and camera distributions. The
1,024-room audit constructed and checked actual meshes; it is separate from
the larger manifest-only check. Historical failing seeds remain regression tests.

The camera audit requests 16 rooms × five baselines × counts 1–12 × three
aspects (.625, 1, 1.6). **2,879 of 2,880 requests succeeded**, with no exact
trajectory duplicates. Seed 4, Studio, aspect .625, baseline 1, twelve cameras
exhausted the bounded search after finding at most eight feasible cameras. This
is a reported solver rejection, not proof that no feasible twelve-camera set
exists. The audit keeps its denominator and failure. The local camera diagrams
show nearest-camera spacing, optical-axis separation and normalized path
occupancy; the latter combines different room shapes and is not one floor plan.

## Windows, people and physical references

The window matrix contains seven treatments × two view angles × two sides ×
open/closed × day/night × two glass policies × three times = **672 views**.
Two times repeat the exact pose; the third translates the camera by 4 mm.
RGB is byte-identical between glass annotation policies and between repeated
poses. Transparent pixels disappear from window annotations in Through mode;
frames and opaque frit remain. Maximum RGB MAE after the small translation was
2.10/255. This temporal fixture detects obvious instability but does not prove
absence of flicker under every trajectory or lighting condition. Corrected
exposure runs are `windows_exposed/` and `windows_exterior/`; the initial
overexposed backdrop experiment remains recorded as failed visual QA.

Controlled galleries cover 32 hair, 32 garment and 32 body cases, 28 plant
views, 12 occupied-table contacts and 20 materials at two scales. A separate
224-plant triangle audit covers all 14 growth habits. Leaf coverage remains
habit-dependent: vines and bamboo are intentionally much sparser than ferns and
rosettes. Their existence or selector counts are not a photographic-realism test.

Native motion qualification used four rooms, two actors per room, 60 generated
frames, cached ARDY/Llama models and ten diffusion steps. Seven clips passed
admission; one walk crossed the floor by 26 mm at frame 48 and correctly fell
back to a static actor. Accepted locomotion includes root paths of 1.70 m and
2.40 m, with maximum waypoint errors of 25 mm and 41 mm. Accepted stationary
gestures remain stationary by design. Rendered flow checks include visible
moving-person pixels, no missing expected static-valid pixels, and maximum
static-surface P99 reprojection error of 0.00388 pixels; text-prompt diversity
alone is not the motion result.

A hardware Chrome probe also completed actual ARDY/Llama inference: one
60-frame actor at ten diffusion steps was admitted with no static fallback or
WebGPU errors. Cold download, shader compilation and inference took 211 seconds,
including 455 successful pinned-model HTTP responses. This is one functional
browser smoke test, not a motion-distribution or native/browser parity result.

Two complete rooms (seeds 3 and 6) were exported with real CPU material maps
and rendered in Blender 5.2.2 LTS / OptiX Cycles at 256 and independently seeded
512 samples. Their radiometric comparison uses the established linear/HDR
contract. Thin eyeglass coverage alpha is exported as coverage; it is not
misrepresented as a refractive solid. Coarse 32×32 reference-block disagreement
was 0.11% and 0.14%, while pixel-level sampling noise remained approximately
3.7–3.8%. Native/Cycles mean luminance ratios were **1.503** and **1.068**, with
raw relative RGB MAE **0.630** and **0.206**. Geometry alignment passed, but
these results show material/indirect-light differences. The renderer has not
passed a claim of path-traced physical accuracy.

## Perceptual diversity

The local paired study embeds 64 room pairs, three views and two times per
room, using one recorded SigLIP2 model and preprocessing contract. Exact
cross-room nearest neighbors, a common PCA basis and room-bootstrap intervals
are computed by the Rust publication crate. All 3,072 new-study PNG planes
matched the first 64 rooms of the frozen qualification capture byte-for-byte.

Mean nearest-other-room cosine distance changed from 0.06841 to 0.07059; the
paired difference is 0.00218 with 95% interval **[-0.00005, 0.00435]**. The
rank-16 common-PCA volume ratio is **1.42**, with interval **[0.53, 4.55]**.
There are no exact cross-room embedding repeats or cross-room distances below
0.01 in this sample. These small-sample diagnostics do not establish a 10×
increase, statistical improvement in every factor, or downstream training utility.
The matched study stays local and is not included in the public page or paper.

## Completed throughput and memory

All native capture comparisons retain full Auto quality, shadows, GPU material
synthesis, three 512×512 views, one time and Color/Depth/Normal/Position/Semantic.
Warmup is excluded from rates. These are debug-profile binaries with the
repository's optimized development settings, not production release estimates.

| Workload | Measured rooms | Completed rooms/s | P50 / P95 room latency |
| --- | ---: | ---: | ---: |
| Starting checkout, 8 warmup rooms | 32 | 1.348 | 0.752 / 0.971 s |
| New construction/lighting, 8 warmup rooms | 32 | 1.206 | 0.798 / 1.200 s |
| Sustained new construction/lighting, 16 warmup rooms | 144 | 1.230 | 0.786 / 1.207 s |

The added reflection and occlusion work costs about **10.5% throughput** in the
short matched run. It is a quality/correctness tradeoff, not a speedup. The
sustained run completes 3.69 views/s. Its peak RSS is 5.80 GB; RSS slope falls
from 11.15 MiB/room over the measured run to approximately 0.78 MiB/room over
the final 40 rooms. GPU assets and caches have bounded residency, but 160 rooms
do not establish unlimited-process memory stability. CPU-only full material
preparation measured 3.27 rooms/s with four concurrent preparations; that number
excludes rendering, uploads and the production GPU material path.

The sustained preparation timers average 2 ms for layout, 38 ms for geometry,
65 ms for cameras, 220 ms for material preparation, 221 ms for transport,
126 ms for reflection environments, and 368 ms for the inclusive GI setup.
Several timers overlap or contain other stages and must not be summed. They
point toward transport and material work before another layout-solver rewrite.

A separate 24-room, four-warmup consumer test measures 2.007 rooms/s through
CPU readback/upload, 1.327 through shared WGPU, and 1.302 through Vulkan/CUDA.
The complete GPU-consumer paths are still slower in this workload despite
eliminating image readback. Their scheduling/capture paths require further
profiling; transport savings must not be reported as generation savings.

The isolated same-captured-room transport test, 64 measured iterations, takes
27.88 ms through CPU versus 3.23 ms through shared WGPU per three-view transfer
plus identical completed GPU reductions. The shared path transfers zero image
bytes through the host; the roundtrip transfers 25.17 MB/iteration. This is an
8.62× isolated transport improvement, not an end-to-end generator improvement.
The 24-room concurrent WGPU optimizer probe completed **96 finite Adam steps**
at 0.979 rooms/s and 3.91 steps/s, with weight-update completion fenced. It is a
small specified training workload, not a Gekko learning-utility experiment.

Raw receipts, failed probes and successful repeats are retained locally.

## Browser preparation

Hardware Chrome profiling found substantial synchronous reflection tracing,
filtering, material and mesh preparation. The browser path now cooperatively
slices tracing/filtering and material insertion while preserving arithmetic,
sample order and texture contents. Native work keeps its bounded preparation
pool. Browser yielding improves responsiveness; it does not create parallel
WASM workers.

Where available, the implementation uses prioritized `scheduler.yield()`
continuations, with a real timer-task fallback. A timer-only experiment reduced
main-thread stalls but increased regeneration time; its failed latency result
is retained. The final check explicitly disables the scheduling API for one
render mode to exercise fallback behavior. Static WebGPU builds require no
motion-model feature for this scheduling bridge.

The browser test measures interactive display readiness after the material bake,
30 subsequent GPU submissions and completed queue work. It is deliberately
separate from headless capture throughput. Exact binary hashes, seed/URL replay,
adapter identity, task durations and failures are recorded under
`web/scheduler/`. One sequence is a diagnostic, not a statistically qualified
browser performance distribution.

| Eight-room browser sequence | Median regeneration | Longest regeneration main-thread task |
| --- | ---: | ---: |
| Synchronous preparation | 8.801 s | 2.627 s |
| Cooperative timer-only experiment | 14.391 s | 0.566 s |
| Scheduler continuations, concurrent native compilation | 5.643 s | 0.725 s |
| Scheduler continuations, no concurrent qualification/compilation | 6.040 s | 0.642 s |

Each sequence loads seed 400 and regenerates seeds 401–407 with three 512²
cameras. The final uncontended sequence improves the observed regeneration
median by 31%, with unchanged material-map arithmetic and seven-mode rendering
checks. Initial load, excluded from that median, was 10.20 s versus 8.15 s in
the synchronous sequence. Substantial stalls still remain during indivisible
asset/renderer work; this does not meet a frame-budget guarantee. The timer
fallback favors responsiveness but has a measured latency cost on this browser.

## Ceiling, exterior context and floor-detail qualification

The current construction adds continuously dimensioned acoustic clouds,
baffles, timber slats, round/rectangular ducts, sprinkler pipework and suspended
truss modules. Installation sampling reserves lights, walls, columns, partitions
and mezzanines; suspension reaches the actual local ceiling and headroom checks
include raised floor patches. These are bounded decorative/service assemblies,
not a structural or mechanical engineering simulation. Exposed slabs use the
existing concrete material program. Components retain the existing mesh batching,
PBR material synthesis, shadow and annotation paths.

Exterior parcels prioritize actual window normals, including oblique facade
segments. Paving, curbs, railings, facade relief, downpipes, supported pergolas,
roof equipment and denser planting add nearby context. Exterior geometry remains
outside the primary-room reconstruction/Ovoxel region. Visual review corrected
missing pergola cross-supports, rotated planting offsets and building/equipment
contact heights; the finite exterior envelope remains in place.

Rugs sample shape, dimensions, orientation, textile finish and a joined bound
edge. Placement requires a flat free floor patch clear of people, doorways and
existing objects. Floor outlets sample circular/rectangular housings and covers,
with a shared physical connection point used by collision-checked equipment
cables. Unused covers close only when doing so preserves nearby cable clearance.
The appended `floor_outlet` semantic class is RGB `(98, 158, 171)`; existing class
IDs and colors are unchanged.

The local [gallery](../out/room_context_expansion/final/index.html) contains 32
consecutive rooms with three calibrated 512×512 views and all six RGB/annotation
modes, plus inspection views. Its [summary](../out/room_context_expansion/final/summary.json)
links measured distributions to the exact source identity. Qualification:

- 507 core tests passed, 26 explicitly ignored. After the final exterior repairs,
  all three focused exterior tests passed, including 48 transformed pergola
  support/planting cases; 512 full mesh audits and 96 views were rerun.
- All 512 layouts and constructed meshes passed. Strict native and motion-enabled
  WASM Clippy passed; this addition has not had a new browser runtime benchmark.
- All 96 aligned views passed RGB/depth/normal/position/semantic/co-visibility
  validation. Maximum per-view P99 reprojection error was 0.000312 pixels;
  maximum depth/position disagreement was 0.00000763 m.
- The population contains 3,116 ceiling modules spanning all seven systems,
  249 rugs in 167 rooms, and 897 floor outlets in 387 rooms. There are 369
  floor-power cable connections. Minimum installation headroom is 2.351 m.
- Among the 499 rooms with exterior windows, mean window-facing parcel coverage
  is 99.86%. This is a layout proxy (a parcel centroid within 15 m and 31.8° of
  a window's outward normal), not measured pixel coverage or photographic realism.

Two full-quality, six-mode, three-view timing runs measured **1.014** and
**0.950 rooms/s**, versus **1.396** and **0.930 rooms/s** for the saved starting
checkout. Each excludes four warmups and measures 12 rooms. A Gekko training job
and two capture workers shared the GPU throughout this campaign; they were not
stopped. These short, variable results neither establish an isolated speedup nor
prove performance parity. Device/process telemetry and all individual receipts
are retained. The gallery and comparisons remain local and unpublished.
