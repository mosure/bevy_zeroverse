# Realism, procedural coverage and configuration-conditioned capacity

Implementation snapshot: 2026-10-15, local generator 50. The typed capacity
contract, fixed-bin observers, bounded audit/capture tooling, functional graph,
shared primary-region queries, expanded public-suite adjacencies and context-aware
static poses are implemented. Qualification status and source-bound local results
are tracked below. Scientific promotion gates are separate from code completion;
this document does not certify photographic realism, arbitrary CAD topology,
unlimited memory stability or a ten-million-sample learning benefit. Existing
training runs and capture defaults are unchanged.

The next gain should come from **more plausible relationships and more visibly
distinct accepted constructions**, together with correct light transport. More
seeds, object names, texture noise or parameter ranges do not establish more
useful reconstruction data. Capacity must be reported relative to configuration,
sampling policy, representation, resolution and sample unit.

## 1. Current foundation and actual gaps

The current source has 100 object kinds and 36 base material roles. These counts
describe an inventory, not independent choices or effective dataset size.
Existing public-space and kitchen programs, static poses, doors, temporal objects,
screen/art/rug content, ceiling services and exterior context must be extended,
not reimplemented as a competing scene hierarchy.

| Area | Foundation in the working source | Next boundary to address |
| --- | --- | --- |
| Spatial program | `program.rs`, `program/topology.rs`, polygonal envelopes, derived region/portal graph, kitchen and public-space assemblies | Zones are still rectangular and partitions axis aligned; primary selection and vertical support need a shared explicit domain |
| Functional placement | `layout/assemblies.rs`, `layout/programs/`, support and interaction links, continuous furnishing fields | Required functions, accepted relationships and operational space need stronger coverage accounting; random clutter is not evidence of a usable room |
| Objects | Multipart constructions, articulation and category studio audits | Continuous silhouettes, construction joints, support-specific states and within-category accepted variance |
| Appearance | `materials/program.rs`, role/finish variants, correlated PBR maps, metric UVs, filtered GPU bakes with CPU oracles | Material continuity across parts, material-scale plausibility and visible variance under multiple views/lights |
| Light transport | Bounded GI and reflection probes, shared transport geometry, Cycles export fixtures | Tinted/frosted glass transmission and spatial/secondary lighting remain approximations; exposure cannot correct the wrong energy path |
| Humans | AnnyBody, fitted clothing, surface hair, contact poses, optional ARDY and temporal collision checks | Garment folds and silhouette, facial/hair construction, context-conditioned gestures and credible interaction transitions |
| Cameras | Count-aware baseline, connected overlap, full K, handheld paths, fixed/moving mixture and continuous lens controls | Accepted coverage and failure bias across counts/room shapes; task-relevant shared surface coverage rather than raw pose entropy |
| Audit | Structural signatures, HyperLogLog occupancy, object/material grids, exact duplicates, SigLIP2 nearest pairs and covariance diagnostics | No calibrated population-capacity estimator; empirical sample spread must not be relabeled global capacity |
| Execution | Bounded preparation/prefetch, GPU maps, same-device WGPU/CUDA tensor routes, stage timings | Completed capture plus training is the objective; browser cooperative yielding is not CPU parallelism, and native host-memory plateau is not established |

The existing [implementation evidence](procedural_space_implementation.md)
documents transport, memory and performance limitations under its recorded
source identities. They remain open qualification gates; the timing percentages
and image statistics there are not silently reused as current measurements.

## 2. Work sequence and promotion gates

The first coverage correction is implemented: `Mixed` and its `All`/`all` aliases
sample all 20 concrete programs uniformly. The ten office profiles are no longer
the entire root population. The narrower office list remains only for compatible
secondary activities inside an office; public/kitchen roots execute their full
architecture, service reservations and functional placement builders. Tests bind
the Mixed catalog to the concrete enum variants so a new profile cannot silently
remain unsampled. Uniform proposal weights are not a claim of equal accepted
pixel coverage: retain failure yield, actual counts and visible coverage by program.

Default capture accounting is equally important. The conditional indoor CLI
defaults are four 256×256 RGB views, one time at progress zero, density 0.65,
human density 0.25, baseline 0.5, Auto lighting and GPU materials. ARDY and object
motion are disabled. Camera motion fraction is 1, but only its initial pose is
observed; path duration/endpoints do not multiply the dataset size. The separate
`GenConfig::indoor` convenience preset uses three 512×512 views and six channels.
The published gallery's four 640×400 views/two times is a third population.
These three configurations must never share an unlabeled default-size estimate.

### 2.1 Current local qualification

The source-bound evidence is under `out/capacity_implementation/qualified/`.
The default audit uses 1,024 pseudorandom root draws (audit seed 17), four planned
256-square RGB views at t=0, layout/camera checks for every room and actual mesh
checks for the first 64 draws. All 1,024 succeed. All 20 concrete programs occur,
with 41–62 rooms per program; all 99 observed object kinds and all 36 constructed
material roles are represented. This is construction evidence, not a rendered
population estimate. Failed draws would remain in the ordered records.

| Diagnostic | Current observation | Interpretation |
| --- | --- | --- |
| Joint accepted-program bins | 1,024 / 1,024 distinct at the frozen 10 cm/0.1 rad definition | Neither a perceptual equivalence result nor a ten-million-draw forecast |
| Architecture, furniture, materials, lighting and planned cameras | Each block has 1,024 observed bins | These correlated block counts are not multiplied |
| People | 294 rooms have no primary-region person; 731 observed people-block bins | The shared empty-person state counts once; static pose/body/wardrobe remain represented |
| Collision uncertainty | Zero matches among 512 disjoint pairs; 95% two-sided lower bound on D2 is 139.30; upper bound unresolved | A small zero-collision audit provides a weak lower bound, not a large finite support estimate |
| Ten-million successful draws | No point estimate; conservative bound approximately 139.29–10,000,000 at this representation | The interval explicitly exposes lack of extrapolation evidence; its lower endpoint is not the expected dataset size |
| Captured time | One state in the default; human/object motion disabled | Camera endpoint variation is admission-only and adds no captured temporal multiplicity |

A separate prespecified 128-room native cohort uses three 512-square views at t=0,
six simultaneous RGB/geometry/visibility planes, and 16 actual mesh checks. Its
rendered-grid descriptor, source identity and configuration stay separate from
the default construction audit. A six-draw fixed-seed capture checks exact replay.
An independent short-clip check covers fixed, mixed and moving cameras, furniture
motion, two nonzero-start times and both glass annotation policies. Qualification
receipts, aligned previews and execution timings remain local; no old gallery is
relabeled as these captures.

The optional SigLIP2 study freezes the cached model's loaded-weight SHA256,
preprocessing, complete image index and RBF bandwidth before inference. Its
room-weighted sample spectrum and nearest-pair gallery are a distinct whole-image
diagnostic, including exterior context. They do not estimate global support or
replace primary-region geometry evidence. The receipt-bound report is the source
of exact results; no manually selected image subset enters the estimate. All 384
images from the 128 rooms enter this study. With bandwidth 0.35, sample Vendi is
22.2819 and participation is 4.8424; the top normalized eigenvalue is 0.4474.
These scores expose substantial image-space similarity despite the unique
construction bins. They neither count 22 distinct rooms nor estimate the unseen
population. The kernel's maximum off-diagonal eigensolver residual is
2.54e-14; all 14 cached weight shards were checksum verified.

The completed native cohort has zero failed rooms. Maximum depth/position
disagreement is 5.72e-6 m, maximum self-reprojection error is 0.00150 pixels, and
maximum normal-length error is 2.38e-7. Mean per-view shared visibility is 73.6%
(range 36.5–97.8%). The six fixed-seed captures have one complete capture hash,
including annotations, calibration and time. All twelve temporal clips deliver
their requested progress values, 0.15 and 0.75, with 12/12, 5/12 and 0/12 fixed
cameras for the 0%, 50% and 100% moving-camera conditions. Temporal collision
admission accepts 7 of 18 selected mechanisms in each condition; the other
11 remain recorded rejections, not silently relaxed clearance. Human motion is
disabled in this clip cohort. A separate eight-room camera audit accepts all
288 groups spanning counts 1–12 and baselines 0, 0.5 and 1. Its geometric overlap
proxies remain distinct from the measured rendered shared visibility above.

All these receipts bind source
`4c2e6bf429985ce6acfc321a03fbf1c9989901132bdd5773d41dd39bf8485d36`.
Use `default-report/`, `render-report-reviewed/` and `kernel-report-reviewed/` under the
qualified directory for the complete tables, definitions and nearest-pair
gallery. Earlier intermediate directories are not interchangeable evidence.

#### Execution cost

The matched 16-room, two-worker preparation probe (seeds starting at 4700,
two warm-up rooms) completes in 8.674 s, or 1.845 prepared rooms/s. Its initial
probe measured 9.520 s, or 1.681 rooms/s. Both ran on a shared machine with an
existing training job; this small comparison shows no evident preparation
regression, not a statistically established speedup. Geometry detail, materials,
lighting and shadows were not reduced. The 1,024-room construction/descriptor
audit takes 96.80 s with two workers; that lighter audit is not production scene
preparation or rendering.

Completed three-view native capture takes 318.81 s for 128 rooms (0.401 rooms/s),
with per-room p50/p95/p99 latencies of 2.27/4.33/5.58 s. Startup takes 1.66 s and
the observer takes another 20.53 s; total runner time is 347.79 s. This run shares
the GPU with existing training, and a CPU audit overlaps part of it. It is an
annotation-heavy qualification run, not an isolated JIT tensor throughput
measurement. No causal training-step or GPU-utilization improvement is claimed.
Normal generation does not run the capacity observer, functional-graph audit,
SigLIP2 model or report pipeline. Qualification uses bounded persistent workers
and one persistent render app/device; it never changes the training job.

### 2.2 Work and promotion gates

| Phase | Concrete output | Acceptance and dependency |
| --- | --- | --- |
| A: measurement contract | Versioned capacity query/receipt, factor identities, accepted-state diagnostics and analytical estimator fixtures | No population number without its unit, metric, uncertainty and provenance; establish before expansion claims |
| B: transport and construction correctness | Window/light fixtures, contact/seam audit, clear region ownership | Correct annotation alignment and bounded physical-reference errors before calling broader RGB variation realism |
| C: functional and architectural space | Generalized regions/supports and coherent assembly graphs | More accepted topology/relationship diversity at unchanged collision, support and navigation requirements |
| D: visible object/material/human variety | Category-level construction and material interventions, plausible interactions | Distinguishable geometry/appearance at capture resolution, coupled RGB/annotation correctness and visual review |
| E: reconstruction-conditioned sampling | Calibrated camera/trajectory distributions and family-aware mixtures | Shared-surface quality, negative-pair retention, camera-count feasibility and calibration/flow oracles |
| F: bounded execution | Lower critical-path costs, stable resident resources and explicit platform capabilities | Completed native/WASM capture and concurrent-consumer measurements, with unchanged quality settings |
| G: capacity and utility qualification | Nested independent cohorts, held-out novelty forecasts, frozen-budget model study | Calibrated intervals and forecast horizon; real-data reconstruction gains assessed separately |

### 2.3 Implementation status and unclosed gates

This pass implements and exercises a vertical slice across A–G. It does **not**
complete every expansion in section 3 or convert scientific promotion gates into
passed checks. The following distinction is part of the deliverable:

| Phase | Implemented and exercised | Still unimplemented or unqualified |
| --- | --- | --- |
| A | Portable typed contracts; pure configuration query; joint accepted/rendered observers; explicit factor activity; exact fixed-partition statistics, confidence bounds and analytical fixtures | Perceptually calibrated partitions, unseen-tail estimator and learned utility model |
| B | Shared primary polygon/floor/ceiling membership; mesh/layout validation; full-calibration annotation checks; corrected raised-arm/floor-mat false collision; exact replay and nonzero-start capture fixes | Physically complete tinted/frosted transport, new matched radiometric qualification, arbitrary layered supports |
| C | Auditable accepted functional relationships and capacity accounting; all 20 root programs; 28 compatible directed public-suite program pairings with continuously sampled dimensions | General oblique partitions, holes, stacked support domains and transactional redesign of every functional assembly |
| D | Context-conditioned continuous working/listening/talking engagement and gaze with clearance checks; construction material/category coverage and whole-image nearest-pair inspection | Photographic garment/hair/skin, new interventions for every object/material category, causal visible-factor attribution |
| E | Fixed/mixed/moving camera captures, explicit initial progress, source-bound counts/baselines audit and measured rendered co-visibility | Real-model reconstruction benefit, fisheye/rolling-shutter contracts, broad motion-model fidelity study |
| F | Bounded ordered CPU workers and capture budgets, persistent renderer, opt-in measurement/model cost, stage timings and native/WASM compile checks | Unlimited-process memory plateau, uncontended JIT/concurrent-consumer benchmark, current Chrome/Metal image qualification |
| G | 1,024 construction draws, 128 rendered rooms, exact replay, twelve clips, sample-kernel audit, provenance-checked HTML/TeX/PNG and optional canonical publication integration | Calibrated extrapolation beyond observed samples and a frozen-budget downstream learning study |

Native tests cover the new contracts and regressions; strict native and WASM
Clippy checks pass. The broad native suite passes 628 tests with 29 explicitly
ignored, followed by 26 targeted sampler tests after the final initial-progress
change. Publication tests pass 66 unit tests (one explicitly ignored) and 21
integration contracts; strict publication Clippy and formatting checks also pass.
These are distinct from hardware browser rendering: Chrome is unavailable
in this execution environment, so compilation is not presented as a Metal test.
The local paper draft includes the new capacity evidence; previously published
galleries retain their own source identities and were not republished.

Deliver vertical slices throughout these phases. First slice: a meeting/work area
with a transmitting window, table/chairs, task lighting, a person interacting
with the surface, cables and a useful camera rig. Second: a kitchenette/cafe
service and seating group. Third: gallery/lecture/transit connectivity. Each
slice exercises support, topology, materials, lighting and observation together.
No wholesale rewrite is needed before a slice becomes usable.

## 3. Realism and diversity implementation backlog

### 3.1 Architecture and exterior context

Extend the existing region/portal graph to own polygonal region boundaries,
wall-local frames, opening ownership and explicit primary reconstruction regions.
The same accepted representation must drive triangles, navigation, contact,
light transport, schematic, AABB membership and O-voxel selection. Start with
simple connected polygons and oblique partitions; add holes and multi-level
connectivity only after robust triangulation and membership support exist.

Sample topology before dimensions: offset wings, vestibules, alcoves, connected
bays, splayed walls and circulation loops. Condition region area, ceiling height,
door width, furnishing capacity and services on intended use. Compact kitchenettes
and service landings should not inherit a large-office minimum everywhere.
Split ceilings into compatible patches with explicit joints, soffits, clerestory
bands, acoustic fields and service zones. Layer ducts, suspended trays, pipes,
sprinklers, hangers and trusses by clearance and plausible connections rather
than distributing isolated decorations. Use separate support intervals for floors,
stairs, landings and mezzanines; a single height at X/Z cannot represent all of them.

Window programs should jointly sample aperture group, sill/head heights, corner
condition, reveal, sash/track/hinge mechanism, frame/gasket/pane stack, curtains
or blinds and operating state. Couple tint, thickness, roughness, partial privacy
bands, frit, attenuation and exterior lighting. Preserve distinct surface and
through-glass annotation policies. A physical frosting implementation must affect
transport as well as apparent blur; straight-ray labels remain explicitly labeled.

Generate a coherent exterior parcel: facade spacing, setbacks, ground/terrace,
vegetation layers, neighboring windows, roof equipment and distant context. Keep
the finite sky proxy as an identified fallback. Do not increase apparent indoor
diversity by changing only the distant backdrop. Report exact primary-region and
whole-image metrics separately, including sky/neighbor pixel fractions.

Tests: nearly collinear/thin polygons, narrow valid portals, shared-wall ownership,
closed surfaces, stepped supports, sloped junctions, opening sweeps and all prior
reported failing seeds. Require topology membership and actual triangles to agree.

### 3.2 Furnishing, clutter and placement

Use a functional graph over existing assemblies and objects: `supports`,
`seats_at`, `faces`, `powers`, `serves`, `stores`, `within_reach` and
`shares_circulation`. Record desired and accepted roles/capacity separately.
Transactional local placement should adjust group translation, rotation,
proportions and compatible constructions before abandoning a required function.
Keep bounded deterministic search; never hide failure by dropping a required
member or loosening collision clearance.

Sample coherent tidy, occupied, recently used and maintenance states. Let nearby
objects share purpose and orientation without forcing a straight wall-aligned row.
Variable-density fields, incomplete groups, nested supports and reachable empty
surface area should coexist. Wires need sockets at both ends, route slack,
gravity-compatible hanging segments, connectors and collision-aware paths. Route
power/AV/plumbing before dressing surfaces. Floor outlets, wall outlets and
switches need plausible access and attachment heights.

Keep category-level construction parameters meaningful:

- Tables/desks: swept outlines, unequal radii, trapezoids, modular joins, edge
  treatments, overhang, pedestal/trestle/frame/leg arrangements and cable management.
- Seating: continuous shell/back curvature and height, arm shape, upholstery
  thickness, headrest, stool/no-back state, caster/frame supports and seat insertion.
- Storage/appliances: bay count, wall/floor mounting, door/drawer construction,
  handle families with continuous proportions, visible interiors and supported contents.
- Small props: tissue opening and folded-sheet structure, bin rim/lid/liner,
  bottle/cup cross-section, caps, phone construction, outlets and switch plates.
- Plants: branching hierarchy, leaf attachment and orientation, juvenile/mature
  blade size, clustering, density, stem visibility, pot/soil/root clearance and
  species-conditioned architecture; qualify leaf back/front reflectance together.
- Screens, boards and art: compositional grammars at multiple spatial scales,
  meaningful hierarchy, margins and content density, with independent object and
  content identities. Pixel-level noise and invisible writing do not count as
  additional structural diversity.

Measure each category at fixed envelope/light/view and again in accepted rooms.
Use depth/normal silhouette descriptors, part relations, visible projected area,
support state and active parameter influence. A factor that changes a manifest
but never the visible geometry should have zero visible influence at that scale.

### 3.3 Materials, lighting and physical appearance

Sample building/room/assembly finish context first, then local deviations. Preserve
wood grain through joined parts and face/end-grain transitions; separate plank or
tile installation from substrate microstructure. Give fabric coherent seams,
weave direction, yarn scale and contact compression. Vary leather grain, stretch,
creases and coating coherently. Stone/paint/plaster/ceramics need correlated
mesoscale variation, edge construction and restrained wear rather than a stronger
full-surface noise amplitude. Rugs need borders, pile direction, weave/pattern
hierarchy and contact with the floor. Object/role assignment must exercise the
space, not leave new recipes unused.

For every substrate: measure normal/roughness mip behavior, map frequency,
albedo/roughness/metalness ranges, world-scale pattern size and angular response.
Use multiple grazing distances and lights, especially native Vulkan and Chrome
on macOS/Metal for GPU-baked cloth. Include CPU/GPU maps and complete mip chains;
matching only level zero misses the reported chrome-like fabric failure mode.

Prioritize the known glass-energy path: tint/attenuation, opaque frames, frost/frit
coverage, occlusion and secondary lighting. Validate radiometry fixtures first,
then complete rooms against matched Cycles optics with independent reference
seeds and a noise floor. Keep linear HDR comparisons distinct from tone-mapped
RGB. Do not fit one global exposure multiplier to hide room-dependent transport
error. Local probes improve reflections but are not a claim of full path tracing.

Lighting distributions should condition sun/sky, artificial circuits, fixture
beam shapes and active fraction on spatial use, while retaining genuine low-light
and strong-daylight tails. Record EV100, lux/radiance and exposure independently.
Monitor clipping and dark-pixel fractions by intended lighting stratum. Keep
sensor noise, blur, white balance and compression in a separately recorded RGB
transform; they cannot manufacture geometry diversity or alter semantic labels.

### 3.4 Humans and temporal interactions

Retain AnnyBody and upstream model loaders. Couple morphology, clothing fit,
garment length/ease, seam placement, neckline, sleeves and layering. Favor fitted
surface construction and pose-aware deformations over added block geometry.
Hair programs need connected roots, mass/silhouette and continuous length,
parting, curl, bundle and tie parameters. Qualify backlighting and temporal
aliasing, not only portrait screenshots. Facial features must follow anatomy;
beard/lip/eyebrow masks must not leak onto noses or unrelated skin regions.

Condition static gestures on context: speaking, listening, pointing, looking over
a shoulder, reaching, resting on a table, crossed/relaxed legs, holding a phone,
reading and collaborative postures. Vary spine/head/limb direction within joint
and support constraints. Report accepted pose distribution and visible joints,
not just the number of nominal pose families. Avoid forcing every person to face
the cameras. Export world and per-view joints, visibility and instance identity.

For moving people/furniture, plan contacts and reserved swept volumes jointly.
Door/chair/cabinet and human paths must not collide after independently passing
static checks. Use bounded ARDY retries and record selected, generated, admitted
and static-fallback actors. Rejected complex actions remain visible in metrics.
No motion model or temporal clearance cost is added when motion is disabled.

### 3.5 Cameras and reconstruction utility

Retain baseline as a spatial control, lens distribution as an intrinsic control,
and motion fraction as a temporal control. Audit their *joint accepted* output
across 1--12 cameras, baseline extremes, room topology, density, aspect and human
motion. An increasing request must not silently become repeated paths or an
unreported fallback to a narrow common optical-axis intersection.

Measure full-K and pose distributions, height above the actual support, view
pitch/roll, path length/speed/curvature, nearest trajectory distance, gaze spread,
directed co-visibility, triangulation angle, unique primary-surface area and graph
connectivity. Include 0%, mixed and 100% moving-camera conditions while objects
move. Fixed camera K/pose must remain invariant; flow must use declared times,
not render FPS. Wider pinhole FOV is not a fisheye lens model. Distortion and
rolling shutter need separate projection/flow/visibility oracles before enabling.

Store requested overlap strata and realized overlap separately. Retain hard and
negative pairs according to the declared mixture; no model-dependent filtering.
Compare new scenes versus extra views versus extra times at fixed rendering and
training budgets. More views can add correspondence constraints without adding
new independent rooms. Protect family splits across geometry/appearance/camera
counterfactuals, worker scheduling, shards and resume.

### 3.6 Performance without loss of support or quality

Profile the existing `PreparationTimings` wall partition and GPU completion,
upload, queue/readback and consumer timelines. Report startup and warm streams,
p50/p95/p99 latency, accepted scenes/s, training steps/s, peak memory and rejection
work. Separate GPU-only tensor transport from completed generation. Compare CPU
and GPU baking with the *same* recipes and annotations under a concurrent consumer;
GPU execution is not automatically faster when training owns the device.

Optimize immutable assembly reuse, BVH staging/traversal, batch dispatches and
bounded upload/lookahead before enlarging worker pools. Cache by complete recipe,
geometry and resolution identity. A cache keyed only by category would reduce
procedural support. Keep deterministic stream independence and output ordering.
Browser cooperative slices need bounded work; introduce Web Workers only with
explicit transfer/isolation support and demonstrated net benefit. GPU features
must remain portable and fully readiness-gated. Native shared-device lifetime
must preserve tensors held by the learner; no implicit device reset is acceptable.

## 4. What “effective dataset size” means

Let `C` be the resolved generator/capture configuration, `E` the exact engine,
assets and platform contract, and `s ~ pi` the declared seed sampling law.
Generation includes bounded retries and recovery. Separate success probability
`a(C)` from the distribution of successful outputs; failure is not a useful
diversity class. Let `X = Phi(render(G_E(C,s)))` and let a frozen partition
`Q_epsilon(X)` define classes with masses `p_j` conditional on success.

The unit can be a structural room family, fully dressed scene, synchronized
multi-view capture, single observation, or temporal clip. They answer different
questions. Multiple views/frames of one room never become independent rooms in
a confidence interval. Appearance-only variants retain their geometry-family ID.

| Quantity | Meaning | Can configuration alone answer it? |
| --- | --- | --- |
| Seed/program support ceiling | Number of possible deterministic inputs or quantized feasible programs | Sometimes a loose upper bound; not useful image diversity |
| `D0 = count(p_j > 0)` | Number of reachable classes under a frozen equivalence rule | Usually unknown; sampled distinct count is only observed support |
| `D1 = exp(-sum p_j log p_j)` | Entropy-equivalent balanced class count | Requires the joint accepted distribution; marginal products are invalid |
| `D2 = 1 / sum p_j^2` | Collision-equivalent class count | Estimable at a specified resolution, with uncertainty and possible censoring |
| `U(n) = sum [1 - (1-p_j)^n]` | Expected distinct classes in `n` independent successful draws | Exact if masses are known; otherwise horizon-limited inference |
| `V_sigma(sample)` | Similarity-kernel effective diversity of an observed sample | A sample diagnostic, not global capacity or statistical independence |
| Learning-effective sample size | Data needed for a specified model/loss/domain/compute target | Requires controlled downstream learning experiments |

For fixed configuration and one deterministic 64-bit root seed, support is at
most `2^64`; a fixed seed has support one. Independent domain-separated streams
do not create extra seed entropy. Additional independently sampled material,
camera or model seeds count only when the seed policy declares them. Continuous
parameter intervals describe an idealized support, not infinitely many
distinguishable digital captures. Finite precision, dormant parameters, occlusion,
constraints and aliasing can collapse many inputs to the same observed class.

The constructor graph should expose active factors, parent conditions, proposal
laws and accepted outcomes. For a correctly factorized discrete program,
`H(Z) = sum H(Z_k | parents_k)`, with renderer/quantizer entropy no greater than
input entropy. A product of discretized marginal ranges is merely a loose
combinatorial ceiling. Report inactive branches, correlated factors and measured
acceptance; independent random draws do not imply independent visible outcomes.

### 4.1 Resolution and representation

The full sample descriptor is a **joint** record over the following blocks. It
must not default to the existing architecture-only signature. A block can have
zero observed influence while its parameters have nonzero program entropy.

| Block | Accepted state to include | Visibility/capture reduction |
| --- | --- | --- |
| Architecture | Functional program, primary polygon, connected regions/openings, floor/ceiling support surfaces, doors/windows and exterior context | Primary region and whole image reported separately; hidden neighboring topology cannot substitute for reconstruction coverage |
| Furniture and props | Actual category/construction parameters, sizes, transforms, support/interaction graph, object states and actual surfaces | Rejected proposals excluded; projected area, occlusion and silhouette at requested resolution determine visible contribution |
| People | Accepted count, body parameters, joint poses, contact state, wardrobe, hair, face and placement | Absent/invisible people contribute no image variation; body identity versus pose/motion reported separately |
| Materials and content | Assigned material roles/finish variants, PBR maps at metric scale, screen/board/art/rug compositions | Unused recipes, subpixel relief, filtered mip detail and unviewed backsides do not increase visible capacity |
| Lighting | Sources, geometry of emitters, direction/intensity/spectrum, environment, exposure and enabled transport | Separate scene lighting from RGB-only sensor transforms; measure clipping and shadow/texture readability |
| Observation and time | Actual captured K/pose, shared surfaces, baseline/triangulation, captured times and actor states | Use captured transforms, not proposed targets or unobserved path endpoints; one timestep has temporal multiplicity one |

For each block retain proposal, accepted-program, constructed-surface and rendered
diagnostics as different levels of evidence. Factor ablations vary one stream on
the same geometry family; population estimates use the original joint draws with
their real correlations. A useful intervention must improve its targeted visible
block and full-sample novelty without merely increasing background/noise variance.
Rank expansion work by conditional redundancy, projected feature coverage and
measured capture cost, not by new enum variants alone.

The local Rust research probe `cargo run --example audit_capacity -- --output DIR`
collects all six blocks in source-bound accepted manifests, the four t=0 camera
poses, primary construction material areas and geometry checks. It retains every
requested seed/failure. It does not initialize a renderer, bake textures or load an
embedding model, and is **not** the proposed capacity API or a visible-capacity
estimator. Its 512-room default uses an independently seeded ChaCha8 population
draw; actual generated seeds and capture settings are recorded in the receipt.

Use a vector of qualified diagnostics rather than one opaque aggregate score:

1. Structural family: canonical primary-region topology and object/support graph,
   with declared metric bins; no seed, UUID or floating-point noise in signatures.
2. Surface geometry: sampled actual primary-room surfaces/depth descriptors,
   metric distances, silhouette/normal detail and category-conditioned coverage.
3. Appearance: matched surface materials plus visible RGB, with fixed renderer,
   view/exposure policy, image resolution and optional object crops.
4. Observation/task: K, pose, overlap graph, triangulation and surface coverage,
   including a temporal descriptor when clips are requested.
5. Frozen learned representation: verified SigLIP2 checkpoint, preprocessing and
   kernel/threshold, supplemented by geometry and local-feature descriptors.

Quantizers/codebooks are frozen on a disjoint pilot and hashed. A continuous
distance threshold is generally **not transitive**; do not turn pairwise-near
relations into “species” using connected components that chain dissimilar rooms.
Use a declared partition for occupancy formulas. Report direct near-pair rates
separately. Codebook overflow/new-cell behavior and coordinate canonicalization
must be specified. A material-sensitive metric and a geometry-only metric should
disagree predictably under an appearance-only intervention.

Existing occupancy HyperLogLog estimates distinct *observed* signatures with sketch
error, not unseen support. Existing centered SigLIP covariance effective rank and
rank-4/8/16 ellipsoid ratios are also sample diagnostics. They do not become a
capacity estimate by exponentiation or multiplication. For a similarity-aware
companion, use a declared PSD kernel, e.g. an RBF on normalized embeddings, and
the Vendi spectrum of `K/n`. Distinguish this uncentered sample kernel from the
existing centered covariance calculation. Kernel bandwidth, view policy and
sample size must be fixed for comparisons. A linear embedding kernel is rank
limited by feature dimension; an RBF avoids that algebraic cap but is still
sample- and bandwidth-dependent. Neither proves nonmemorization or realism.

### 4.2 Occupancy and scaling relations

For `n >= 1` independent successful draws from fixed masses:

```
collision_probability c2 = sum p_j^2
expected equal-class pairs in n draws = n(n-1)c2/2
expected distinct classes U(n) = sum [1-(1-p_j)^n]
next-draw novelty U(n+1)-U(n) = sum p_j (1-p_j)^n
n / [1+(n-1)c2] <= U(n) <= min(n, D0)
```

The lower bound follows by applying Cauchy-Schwarz to sample class counts and
Jensen to their squared-count sum. It is a lower bound, not a fitted novelty curve.
`D2` alone does not identify all masses or `U(n)`. Equal collision diversity can
coexist with different rare tails and different long-run novelty.

For **uniform** support `S`, the exact occupancy is
`S[1-(1-1/S)^n]`, approximately `S(1-exp(-n/S))`. Ten million draws from one,
ten and one hundred million equiprobable classes give approximately 10.0%,
63.2% and 95.2% distinct draws, respectively. These are analytical examples,
not estimates of Zeroverse. A first repeat is expected much earlier than substantial
saturation: expected colliding pairs reach one near `sqrt(2 D2)`. Different
questions must not share an unexplained “dataset size” number.

For sampling seeds without replacement from a finite pool of size `M`, replace
the absence probability by `choose(M-m_j,n)/choose(M,n)` when class multiplicities
`m_j` are known. Consecutive seed sweeps are deterministic audits, not automatically
IID confidence samples. A separately seeded randomized audit over the intended
population supports statistical inference; family-grouped/stratified sampling
needs its declared weights and variance method.

At several fixed geometric resolutions, test a local relation
`log D2(epsilon) approximately a - d log epsilon`. Its slope is a resolution
scaling exponent over the measured interval, not a universal dimension or a
forecast over sample count. Discrete branches, occlusion and finite precision
create plateaus/crossovers. A proposed learning curve
`L(n) = L_infinity + A U(n)^(-beta)` is only a downstream hypothesis; fit and
compare it to raw-count/family-count alternatives on held-out learning runs.
Do not infer `beta` from SigLIP rank or use it to promise 10M-scale transfer.

### 4.3 Estimation, uncertainty and refusal cases

For `n >= 2` IID room-level observations with cell counts `n_j`, use
`c2_hat = sum n_j(n_j-1) / [n(n-1)]`. It is unbiased for collision probability;
its inverse is not unbiased for `D2`. Estimate uncertainty at the room-family
level using a justified U-statistic method. Pair counts share rooms and are not
independent Bernoulli trials. Ordinary resampling of all-unique observed labels
cannot reveal an unseen rare tail.

Include a conservative fallback based on **prespecified disjoint pairs** of
independent rooms. For zero matches in `m` such pairs, the exact one-sided
`1-alpha` upper bound on `c2` is `1-alpha^(1/m)`, yielding a lower bound on `D2`
and no finite upper bound. With 512 rooms, `m=256`: that 95% lower bound is only
about 86, despite 512 observed distinct classes. Using all pairs can be more
informative with a valid dependency-aware method; treating 130,816 pairs as
independent binomial trials is invalid. A uniform-support birthday inversion is
stronger only under its explicit model assumption.

Publish sample coverage, singleton/doubleton counts, observed rarefaction and
held-out novelty. Good-Turing/unseen-species methods can inform limited forecasts;
they do not permit an arbitrary jump from hundreds of scenes to ten million.
No distribution-free finite support upper bound follows from a small sample
without a declared finite domain. Confidence levels, optional stopping, multiple
resolutions/configurations and fitted quantizers need explicit treatment.

Use a frozen discovery/validation split; choose thresholds before final audit.
Start with no extrapolation beyond observed sample size. Enable an extended
horizon only after nested held-out cohorts demonstrate predictive calibration,
and record that horizon in the receipt. “Insufficient collisions”, “unseen tail
unresolved”, “different configuration” and “unqualified representation” are useful
API results, not exceptions to be replaced by infinity or a guessed integer.

## 5. Implemented public API and further qualification

Portable schema types live in `bevy_zeroverse_capture::capacity`; the
configuration adapter/factor inventory is in
`bevy_zeroverse::scene::procedural_indoor::capacity`. Offline statistics and report generation are in `bevy_zeroverse_publication::capacity`.
No model load, download, render or GPU synchronization occurs during an estimate
query. Qualification is a separate explicit operation using the existing capture
stream; neither API should require downstream consumers to orchestrate Python.

The callable interface is documented in [capacity.md](capacity.md). Qualification
is explicit; the fixed-partition implementation has no unseen-tail model. An optional
SigLIP2/RBF sample diagnostic is kept separate from occupancy estimates. A typical integration is:

```rust,ignore
let request = CapacityRequest::default();
let plan = config.capacity_plan(&request, capture.clone())?;
// CPU: capacity::generate + observe. Native: completed Sample + observe_rendered.
// Offline: bevy_zeroverse_publication::capacity::qualify.
let estimate = config.estimate_capacity(&request, capture, &receipt)?;
```

| Type/field | Required semantics |
| --- | --- |
| `ResolvedPopulation` | Generator/source/capture-engine identity, resolved config hash, platform capabilities, image/time/mode plan, seed law and failure/recovery policy |
| `DiversityDefinition` | Unit, canonicalization, descriptor/partition/kernel version+hash, resolutions/bandwidths, scope and frame/view weighting |
| `FamilyIdentity` | Stable structural ancestry across material/camera/time interventions, separate from observation seed and exact archive hash |
| `FactorInventory` | Stable factor ID, parent conditions, active distribution, quantization units, proposal/admission counters; no independent-dimension claim |
| `CapacityReceipt` | Complete/failed status, requested/delivered counts, failures, seed/family membership, raw descriptor/count hashes, model/partition identities, estimator version and validation horizon |
| `EstimateStatus` | `ObservedOnly`, `Bounded`, `ModelConditional`, `NotQualified { reasons }`; stale/malformed evidence is an error |
| `ScalarEstimate` | Optional point, lower/upper bounds, bound kind, confidence, population or sample scope, method and assumptions; no NaN/Infinity JSON |
| `CapacityEstimate` | Per-definition D0/D1/D2 where supported, novelty/occupancy curve, failure yield, conditional coverage and sample-kernel diagnostics |
| `ForecastPoint` | Requested draw count, unique/novelty interval, model identity, extrapolation status and maximum qualified horizon |

`config_hash` excludes the particular observation seed but includes its **policy**
and every resolved distribution-affecting setting: camera count/lens/baseline,
static fractions, motion model settings, appearance, quality, resolution, glass
policy and sampler/recovery versions. Hash canonical typed finite values, not
user JSON key order or unexpanded defaults. Timing/hardware scheduling is excluded
only after reproducibility qualification; renderer capabilities still belong in
the observation identity. Exact seed lists and model stochastic keys are stored
in the receipt. Cross-configuration reuse is prohibited unless the metric-specific
invariance is explicit and tested (e.g. a truly camera-independent structural
descriptor). No automatic interpolation across configs in the first version.

Planning without a receipt returns the resolved factor inventory; an estimate
query requires a matching receipt and never starts qualification. Zero completed
observations produce `NotQualified` estimates. Empty/one-sample cohorts,
failed captures, unknown fields, mismatched model hashes, duplicate family IDs,
unsupported temporal units and incompatible glass contracts are handled explicitly.
Quality gates are reported separately, never folded into an opaque diversity score.

Resolve a `CapturedFactorMask` from the actual frame/view schedule and enabled
actors. Its per-factor `Observed`, `AdmissionOnly`, `Inactive` or `Unqualified`
state accompanies every result. With one t=0 capture, camera path endpoints are
`AdmissionOnly`: path clearance can change the accepted starting-pose distribution,
so these controls remain in the configuration hash, but they do not create a
temporal multiplier. Disabled human/object motion is `Inactive`; static body,
clothing and pose variation remains `Observed` if visible. Repeating a static frame
at several times adds zero distinct captured states. Extra modalities describe the
same state and do not become independent samples. Motion readiness and actually
delivered times must agree with this mask before a receipt can be complete.

CLI: `indoor_capacity plan`, `audit`, `capture` and `estimate`, plus publication
`capacity-qualify` and `capacity-report`, with typed JSON and immutable receipts.
The viewer has a read-only “Coverage evidence” panel
showing unit, resolution, sample count, status and forecast horizon, with an
explicit receipt import; expensive audits remain deliberate CLI operations. Avoid a live “millions of rooms” counter on every slider.
Changing a relevant setting marks evidence mismatched; it must not silently
trigger a new render or interrupt slider behavior.

## 6. Experimental grounding and bounded budgets

1. **Analytical oracles first.** Uniform and skewed finite populations, collapsed
   support, correlated factors, inactive parameters, invisible texture changes,
   duplicate frames, finite seed pools and a common-mode/rare-tail mixture. Verify
   occupancy/collision identities, monotonicity, bounds, numerical stability and
   interval coverage over repeated independently seeded runs. Include zero
   collisions, missing classes, rejection-biased sampling and degenerate kernels.
2. **Calibrate equivalence on disjoint pilots.** Render controlled geometry,
   material, lighting and camera changes. Review nearest pairs and local crops.
   Fix metric thresholds in metric/pixel units and freeze codebooks/kernel scales.
   Use holdout scenes to check which meaningful changes each descriptor detects.
3. **Structural audit ladder.** Start with 1,024 independently sampled seeds and
   escalate to 4,096/16,384/65,536 only when precision or uncovered strata justify
   it. Use production manifests and actual-mesh checks on a prespecified subset;
   a CPU descriptor cannot certify rendered appearance. Include consecutive-seed
   regressions separately, with no IID confidence interpretation.
4. **Rendered ladder.** Prespecify 128/512/2,048 rooms, three 512-square views,
   then extend only if forecast validation calls for it. Keep a discovery and
   independent holdout cohort. Static default, mixed camera motion, human/object
   motion, lighting tails and public/kitchen programs are separate weighted strata.
   Bounds are ceilings, not instructions to spend the whole budget automatically.
5. **Nested forecast checks.** Fit on prefixes and predict later independent
   blocks; measure novelty calibration, interval coverage and residuals by room
   function, lighting, occupancy and camera overlap. Compare uniform, finite
   mixture and nonparametric occupancy models with held-out scoring. Reject a
   model when it misses rare tails; do not choose it for the largest capacity.
6. **Reconstruction experiment.** Freeze learner, optimizer, compute, annotation
   contract and real evaluation data. Sweep scene-family counts at matched steps;
   separately sweep views/times per family and duplication. Measure calibration,
   depth, normals, reconstruction and correspondence by overlap/lighting strata.
   Fit learning laws only after replicated runs support identifiable coefficients.
   No current downstream training is restarted or modified by this plan.

At 10M scale, stream exact fixed-partition counts or external-sort hashes with
bounded RAM. Preserve hash collision checks when practical. Approximate sketches
must report their own error separately from sampling uncertainty. For embeddings,
use a fixed reservoir and batched cached inference; exact all-pairs checks on
bounded cohorts qualify any approximate-neighbor/index method. Random-pair
estimation measures a different quantity from nearest-neighbor search. Avoid a
10M-square similarity matrix, and do not claim Nyström rank/volume is unbiased
without an explicit residual test. All views from a room stay together in splits
and uncertainty resampling.

## 7. Delivery modules and acceptance checklist

| Deliverable | Code home | Required qualification |
| --- | --- | --- |
| Portable request/receipt contracts | `crates/capture/src/capacity/` (new) | Serde roundtrips, unknown-field rejection, finite-number checks, identity mismatch and version conversion |
| Factor registry and config adapter | `src/scene/procedural_indoor/capacity/` (new), existing `metrics` | Real builder factors, active denominators, reproducible accepted-state records; never hard-coded cardinality products |
| Statistical core | `crates/publication/src/capacity/` (new) | Exact finite oracles, Monte Carlo coverage, grouped observations, rare-tail refusal, finite-pool sampling |
| Capture/embedding integration | Existing `indoor_validate`, `indoor_embed`, publication cohorts | Readiness, all-frame identity, checked weights, immutable partitions, failures retained and no render filtering |
| Paper and reporting | Publication Rust templates/figures | One source of truth; exact formulas and qualified observed curves, no manually copied estimates |
| Runtime expansion | Existing architecture/layout/objects/materials/humans/cameras modules | Seed regressions, actual meshes, six aligned annotations, flow and temporal contacts, native plus Chrome WebGPU |

The first implementation milestone should deliver portable contracts, conservative
collision/rarefaction estimates and refusal cases, with **no long-horizon scalar
capacity claim**. The second adds qualified structural descriptors and live
production counters. The third adds image/geometry kernels and nested forecast
validation. A downstream learning law remains a separate research milestone. Exact bounded
qualification supports 65,536 observed rooms, not a 10M in-memory audit. The
optional RBF sample-kernel implementation supports up to 512 rooms with a checked
eigensolver and exact nearest neighbors; external sorting and unseen-tail
forecast models remain unimplemented.

Publication shows the instantaneous qualified state: configuration, sample unit,
resolution, sample count, confidence, success yield and forecast horizon next to
every capacity number. No before/after gallery, stale measurement relabeling or
private matched-study upload is required. The paper may present the formal model
and analytical examples now, but must label empirical population capacity and
learning exponents as unmeasured until their receipts exist.

## References and theory ownership

The occupancy identities above follow directly from indicator expectations;
the lower bound and finite-seed adaptation are stated with their assumptions.
Established diversity and unseen-support literature informs the statistical
design, not a claimed new universal scaling law:

- M. O. Hill, [Diversity and Evenness: A Unifying Notation and Its Consequences](https://doi.org/10.2307/1934352), 1973: effective-number family and differing sensitivity to rare classes.
- A. Orlitsky, A. T. Suresh and Y. Wu, [Estimating the number of unseen species](https://arxiv.org/abs/1511.07428), 2016: extrapolation has distribution-free limits; small audits do not justify arbitrary future horizons.
- D. Friedman and A. B. Dieng, [The Vendi Score](https://arxiv.org/abs/2210.02410), 2023: similarity-kernel sample diversity; it does not establish generator support or learning utility.
- A. P. Pasarkar and A. B. Dieng, [Cousins of the Vendi Score](https://proceedings.mlr.press/v238/pasarkar24a.html), 2024: order-dependent sensitivity to common and rare modes.
