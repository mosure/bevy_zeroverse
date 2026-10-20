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
| Activity recovery | A library could meet its generic furniture-count minimum without shelving. | Missing library storage is repaired independently of the generic count, with unchanged placement clearance. |

The implementation does not resample materials or geometry independently per view.
The updated capture identity records the changed furnishing, tabletop and public
program contracts. Existing unpublished work is retained.

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

Measured accepted distributions:

| Program/category | Objects | Distinct XYZ size bins at 1 cm | Within 2° of a room axis |
|---|---:|---:|---:|
| Classroom desks | 50 | 49 | 10 / 50 |
| Classroom chairs | 50 | 45 | 13 / 50 |
| Cafe tables | 42 | 42 | 8 / 42 |
| Cafeteria tables | 62 | 62 | 8 / 62 |

All 16 showroom samples contained multiple plinth exhibits; 15 contained more
than one exhibit class. Their exhibits comprised 17 sculptures, 14 vases,
15 laptops and 9 desk lamps. Across the full sample, 2 of 233 notebooks and
9 of 84 sculptures were centered within 2% of both support dimensions.
These statistics count geometry envelopes/placements, not unique rendered images.

Validation on the final source:

- 320 audit configurations and a separate 1,280-configuration sweep of all 20
  programs at density 0/1, seeds 0–31, eight cameras.
- 341 representative changed-object mesh checks, plus the existing 12-room full
  geometry test and 128-seed-per-category loose-item geometry test.
- 33 native tests passed across the focused groups: accepted distributions,
  functional programs, known recovery seeds, workspaces, sizing, geometry,
  independent appearance/camera factors, metrics/calibration, wall art and motion
  admission. One older fixture now checks that its activity surface serves real
  seating instead of requiring the earlier grammar's exact first-table category.
- `cargo clippy --locked --all-targets -- -D warnings`, formatting and diff checks
  passed. `cargo check --locked --lib --target wasm32-unknown-unknown
  --no-default-features --features web,human_motion` passed.

The 320-room CPU manifest sample measured median **15.36 ms**, p95 **85.12 ms**,
mean **26.96 ms**, maximum **481.67 ms** on the shared training host. This is a
single-thread diagnostic including camera/person planning, with other bounded
checks running concurrently. It is not a matched performance comparison or an
end-to-end rendering throughput claim. The sampling adds no GPU passes or texture
budget; admission remains bounded, and diagnostics only run on explicit export.

Local, ignored artifacts include `accepted.json`, `focused-validation.json`,
`extremes.log`, and the inspected `arrangements.svg`/`arrangements.png` plan grid
under `out/regularity_review/`. The grid shows actual admitted envelopes, not RGB
renders; its consecutive seeds are not selected by appearance.

## Remaining risks and next tests

1. **Constraint-induced structure.** Reserved routes, wall setbacks, fixed recovery
   search directions and anchored service units still shape accepted distributions.
   Measure them conditionally. Do not widen priors by accepting collisions or
   replacing every failed station with a smaller stereotyped object. Public
   furnishing still largely starts from two aisle-bounded bands. Jitter does not
   replace higher-level topology diversity: alternate aisle counts, pods, perimeter
   seating and asymmetric service arrangements remain useful future extensions.
   Some classroom samples remain sparse after admission; monitor realized capacity
   alongside geometric variance.
2. **Sensor/target coupling.** Screens currently face the associated chair exactly;
   cameras intentionally favor usable overlap and clear floor regions. Targeted
   ergonomic offsets and alternate camera rigs deserve controlled experiments.
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
5. **Metric dimensions versus world coordinates.** Real standard products and
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

No downstream training, RGB render study or real-world transfer benchmark is
claimed by this CPU audit.
