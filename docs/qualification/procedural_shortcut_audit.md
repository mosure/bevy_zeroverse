# Procedural regularity and shortcut audit

The objective is to remove unnecessary predictors of metric geometry while retaining
physical relationships. Objects must stay supported; furniture can be manufactured
in coordinated sets; lecture seats should face a lecturer. Independently perturbing
every part, every view, or every frame would harm reconstruction supervision.

A finite procedural engine cannot promise that a model cannot learn its rules.
The useful test is whether cross-view evidence improves geometry under held-out
construction, placement, appearance, camera, and real-scene conditions.
[Shortcut learning](https://arxiv.org/abs/2004.07780) motivates those interventions;
[domain randomization](https://arxiv.org/abs/1703.06907) motivates varied training
conditions, but does not establish transfer for this generator.

## Findings and implemented sampling

| Area | Shortcut risk | Current implementation |
|---|---|---|
| Public furniture dimensions | Repeated exact chair envelopes and desk/table heights give an unusually precise metric-scale prior. | Seeded collection dimensions plus smaller continuous instance differences. Dimensions are sampled before placement, with the same support and clearance gates. |
| Classroom arrangement | Row/column parity determined collaborative yaw and offsets. | Continuous cluster orientation, row shifts and per-station perturbation. Tiered lecture rows retain their actual support levels, aisle and sightline constraints. |
| Cafes/cafeterias | Fixed 2.7 m pitch, mirrored table rails, repeated depths/heights and chair setbacks. | Sampled pitch, phase/stagger, row drift, per-table rotation and dimensions. Chair setback depends on the actual table and chair depths plus sampled clearance; later tucking remains clearance-aware. |
| Exhibitions | Panel type alternated by row parity; exhibits were centered on their stands. | Sampled program-level class probabilities, stagger and placement disorder. Supported exhibits use available area after accounting for rotated dimensions. |
| Showroom contents | A shared finish seed also selected exhibit class, so coordinated stands repeated the same class. | Exhibit class uses the stand's independent instance seed. Changing its finish identity does not change the exhibit class. |
| Office/services/recovery | Exact cabinet/bookcase depths and fallback desk dimensions recur despite random positions. | Continuous, bounded dimensions for those active paths, including mezzanine workstations and initial seat clearance. Round coat-rack bases retain round proportions. |
| Desktop input devices | The monitor-relative branch discarded earlier position variation and imposed fixed 0.27 m reach and 0.26 m mouse offset. | A shared, continuously sampled reach arrangement plus independent small perturbations; sampled keyboard/mouse dimensions. Devices remain associated with their monitor. |
| Workstation contents | Optional clutter thinning independently removed members of already-selected monitor/laptop arrangements, disproportionately removing dual-screen setups. | Workspace affinity chooses the arrangement; each screen then passes the unchanged physical support gate. Loose clutter density still controls accessories. |
| RNG domains | Zone service transforms reused streams 5810/5811, also used by wall composition. | A dedicated namespace separates zone transforms from wall composition. Seed replay remains deterministic for the current generator identity. |
| Activity recovery | Sparse-room repairs introduced fixed coffee/side-table, storage and bin dimensions, centered-first proposals and fixed gaps. A library could meet its generic count without shelving. | Repair products sample ergonomic metric dimensions before admission. Seeded lateral order, radial phase, offsets and gaps retain support/clearance checks and bounded attempts. Missing library storage is repaired independently of the generic count. |
| Projector admission | An exact-center first proposal won before random proposals were evaluated. | All proposals use continuous support-local positions; one product size is retained through a support's attempts. Optical-axis, support and beam-clearance checks remain active. |
| Screen convergence | Monitor/laptop normals pointed exactly at the associated chair, including after chair tucking. | A seed-stable ergonomic yaw offset survives seating adjustments and physical admission. Monitor offsets are bounded by 0.19 radians, laptop offsets by 0.18 radians. |
| Public anchors and recovery | Movable teaching/presentation furniture always faced +Z; recovery started at the same corner of an exact lattice. | Category-appropriate orientation and offsets, shuffled stratified recovery proposals, and rotationally sampled local alternatives. Service counters and elevator approaches retain their operational orientation. |
| Gallery/showroom fields | Station centers repeated two narrow rails, with a small amount of jitter. | Continuous placement across the actual aisle-bounded available fields, with depth stratification to limit clumping and unchanged circulation reservations. This broadens fields, not aisle topology. |
| Floor lamps | Fixed dimensions and room-axis yaw survived richer lamp construction. | Metric footprint/height distributions in both initial furnishing and activity repair, with independent yaw. Emitter placement still derives from the same physical lamp program. |
| Service construction | Counter and hood geometry varied little at a fixed admitted envelope. | Open preparation frames, storage carcasses, slatted and tapered counter bodies; box, tapered, elliptical and glass hood canopies with variable flues and filter/light layouts. Counter tops preserve their full rectangular support contract. |

The implementation does not resample materials or geometry independently per view.
The updated capture identity records the changed furnishing, tabletop and public
program contracts. Existing unpublished work is retained.

The category-specific construction, scale, service-core and supported-content
qualification is documented in [Public furnishings and small-object regularity](public_object_regularity.md).
It measures actual fixed-envelope meshes and accepted room dimensions separately.

## Diagnostics available in dataset metrics

`metrics::export_metrics*` now includes `regularity_policy` and
`regularity_{ObjectKind}_*` distributions. They operate on accepted main-scene
objects (`neighbor == false`), including secondary interior spaces; they do not
claim to be visibility-weighted image statistics.

- Local width, height and depth in metres, with raw observation counts.
- Within-room same-kind equal-dimension pairs at **rounded 1 cm XYZ resolution**.
  Self pairs are excluded. Counts and fractions are both exported; undefined
  pair fractions for singleton categories are omitted.
- Fraction of yaw angles within **2 degrees** of a room X/Z axis.
- Monitor/laptop facing error relative to the actual associated chair, with explicit linked-target and within-2-degree counts. Unlinked screens do not enter this denominator.
- Actual support-local X/Z coordinates divided by support width/depth, and
  relative yaw. Centered placement means both absolute coordinates are at most
  **0.02**; supported-object counts supply its denominator.

Regularity observations are emitted for present categories. Use the existing
`object_counts_per_scene` distributions for zero-count/absent categories.

Equal envelope dimensions do not imply equal meshes or images. An axis-aligned
cabinet, centered trophy or matched lecture chair is not automatically a defect.
Inspect conditional distributions by room program, density and support type;
do not optimize one aggregate randomness score.

The collector is only used by explicit metrics export. It adds no collection,
sorting, GPU work or retained history to normal generation.

## Qualification

The bounded CPU audit is
`metrics::regularity::tests::accepted_regularity_audit_covers_programs_and_preserves_geometry`.
It uses all 20 concrete programs, seeds 0–15, densities cycling through 0.35,
0.65 and 0.95, three cameras and human density 0.25. It checks accepted layouts,
seed replay, representative changed meshes and accepted size/orientation/content
distributions. Its local artifacts are `out/regularity_review/accepted.json` and
`schematic-scenes.json`; artifacts are not published.

This sampling protocol is a generator diagnostic, not evidence that a model uses
multi-view fusion or transfers to photographs. Manifest timing excludes mesh
construction, rendering and training.

Measured accepted distributions for the current generator (320 rooms):

| Program/category | Objects | Distinct XYZ size bins at 1 cm | Within 2° of a room axis |
|---|---:|---:|---:|
| Classroom Desk | 31 | 29 | 3 / 31 |
| Classroom Chair | 52 | 48 | 6 / 52 |
| Cafe Table | 37 | 36 | 5 / 37 |
| Cafeteria Table | 54 | 54 | 7 / 54 |

| Category | Accepted objects | Within 2° of a room axis |
|---|---:|---:|
| Lectern | 37 | 4 / 37 |
| TeachingBoard | 37 | 6 / 37 |
| InformationStand | 67 | 2 / 67 |
| FloorLamp | 9 | 65.3 / 9 |

Of 33 supported projectors, 0 were centered within 2% of both support dimensions. Of 641 screens linked to a chair, 135 were aligned within 2 degrees of it; all retained the existing forward-facing validation gate. Accepted floor-lamp heights span 1.219–1.837 m across 8 centimetre-size cells.

15 of 16 showroom samples contained multiple plinth exhibits; 12 contained more than one exhibit class. All 320 layout checks and 40 full-room mesh checks passed.

The fixed-envelope service audit uses 64 seeds per category. At a 0.015 normalized six-view depth difference, ordered greedy coverage produced:

| Category | Representatives / 64 | Median construction (µs) | Triangle range |
|---|---:|---:|---:|
| ServiceCounter | 42 | 31.2 | 300–3588 |
| ExtractionHood | 64 | 42.1 | 476–1124 |
| FloorLamp | 64 | 65.3 | 1884–4120 |

Manifest preparation measured 15.66 ms median / 79.11 ms p95 on the shared i9-12900K host. This excludes mesh construction, render, readback and training. Concurrent workloads were active; this is a bounded timing diagnostic, not an isolated throughput comparison.

These counts describe admitted dimensions, orientations and actual geometry, not unique rendered images or an effective dataset size. The CPU preview in `out/regularity_review/service-geometry.png` holds each category’s envelope fixed. Geometry measurements exclude texture and PBR appearance.

Qualification logs and JSON remain local under `out/regularity_review/`. Older render/plan grids must not be interpreted as current-source evidence; source-linked render receipts identify the scope of each illustrative capture.

Validation: **85 targeted native regressions plus the floor-lamp emitter test passed**;
1,344 metric-extreme/replay cases, the 192-sample fixed-envelope service audit and
a separate 240-mesh public-category scan passed. Strict workspace Clippy covers
all targets with CUDA IPC/tensor-fusion features; the WASM `web,human_motion`
build check passed. Browser runtime was not exercised in this pass.

Twelve illustrative counter/hood samples rendered through the production PBR
path using software Vulkan, with shadows and automatic 2× SSAA. The harness
checked color/normal sizes and finiteness, semantic presence and the expected
object-box count. Color captures and representative matched normal maps were
visually reviewed. The gallery is local at `out/regularity_review/index.html`;
its source-linked receipt is `qualification.json`. These small studio captures
are neither a photographic-realism benchmark nor a GPU throughput measurement.

## Remaining risks and next tests

1. **Constraint-induced structure.** Reserved routes, wall setbacks, recovery
   admission and anchored service units still shape accepted distributions.
   Measure them conditionally. Do not widen priors by accepting collisions or
   replacing every failed station with a smaller stereotyped object. Public
   furnishing still uses two aisle-bounded fields. Broader field coverage does not
   replace higher-level topology diversity: alternate aisle counts, pods, perimeter
   seating and asymmetric service arrangements remain useful future extensions.
   Some classroom samples remain sparse after admission; monitor realized capacity
   alongside geometric variance.
2. **Sensor/target coupling.** Screens retain a bounded ergonomic relation to a
   chair; cameras intentionally favor usable overlap and clear floor regions.
   Alternate camera rigs and ergonomic holdouts deserve controlled experiments.
   Completely disjoint camera pairs are not a useful default for fusion.
3. **Finite appearance vocabulary.** Material variants use a bounded per-scene
   atlas (12 slots, two or three structural programs for several substrates).
   Scene seeds vary these programs, but within-room repetition remains possible.
   Measure rendered patch recurrence and held-out construction/content families
   before increasing memory or texture budgets. Do not count tint changes as
   independent geometric examples.
4. **Program-conditioned priors.** Public room sizes, tiered floors, service zones,
   human activity and object catalogs remain correlated with room function.
   These are partly real relationships. Training/validation should include
   compositional holdouts, not only unseen integer seeds.
5. **Thin product programs.** Simple manufactured products and supplies still
   have limited silhouettes in fixed-envelope scans (for example cans, boards,
   bread and towels). Shape coverage alone cannot distinguish appropriate
   product standardization from a harmful shortcut. Prioritize actual pose,
   support, thickness, fold and content programs before adding arbitrary noise.
6. **Metric dimensions versus world coordinates.** Real standard products and
   human ergonomics are useful priors, but exact dimensions must not dominate
   every program. Retain metric consistency across objects, cameras and labels;
   arbitrary per-object or per-view scaling would invalidate supervision.

For downstream model qualification, keep scene-family membership consistent across
appearance/motion variants and evaluate:

- Same geometry with independently resampled material and lighting seeds through
  `AppearanceSettings`, without moving annotations.
- Held-out material/geometry/placement combinations and camera baselines/intrinsics.
- View-order permutation with the cameras/targets permuted consistently.
- Single-view, correctly paired multi-view, repeated-view, and mismatched-view
  ablations, evaluated separately by overlap. A fusion model should benefit from
  the correct additional view under an actual held-out reconstruction benchmark;
  randomization metrics alone cannot establish this.
- RGB-only task inputs. Semantic colors, object IDs, seed, program/profile IDs and
  exact scene metadata are supervision/analysis artifacts, not accidental inputs.

No downstream training or real-world transfer benchmark is claimed by these diagnostics.
Illustrative PBR object captures, when generated, qualify their recorded samples only.
