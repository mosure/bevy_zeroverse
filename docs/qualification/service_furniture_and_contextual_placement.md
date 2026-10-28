# Service furniture and contextual placement

Local qualification of generator **108**, capture **v133**, source SHA-256 `b45c116a8d518e35ac861ce9599aca1f3585ae049306314a02f2405aa1303aae`. Package versions are unchanged; this iteration is not committed or published.

## Review and implementation

The 107-category object audit was reviewed alongside actual layout and finish samplers. The strongest selected bottleneck spans six categories: drawer units, carts, benches, dividers, ottomans and side tables were proposed only along the left/right edges of accepted furniture, with a shared narrow heading range. A fixed category order also made a limited furnishing budget favor earlier choices. Drawer units always had one bank, and four service-furniture builders ignored independent finish seeds when choosing their substrates.

**Furniture relationships.** The placement module now samples continuous edge weights, setback, along-edge displacement and heading spread. Priors retain functional relationships: side tables usually flank sofas, ottomans favor conversational space, and drawers/dividers seldom occupy the working front. Proposals include end and rear positions, alternate orientations and both tidy and looser arrangements. Projected extents use the proposed rotation before placement. Category priority is shuffled while the existing context probabilities and room budget remain intact.

Each requested product is sampled once and reused across at most 12 pose attempts. Rejection changes the pose, not the product's seed or size. The existing exact OBB, room envelope, floor support, entrance, circulation and ceiling checks remain the admission gate. Neither furniture nor obstacles are removed to make a proposal fit.

**Drawer construction.** A continuous metric sampler spans 0.34–0.96 m width, 0.48–0.85 m height and 0.36–0.59 m depth. Wider products may have two unequal banks with independently sampled drawer counts and heights. Fronts, interior boxes and central partitions share the same measured boundaries. Drawer count is limited by useful vertical clearance; hardware moves with its drawer, and the full top remains a level support for props. The dimensioned bank program and front iterator are shared by construction, motion and numeric metrics.

**Appearance.** Drawers, carts, benches and room dividers now honor independent finish seeds. Frame and working/front surfaces can vary within appropriate wood, metal, polymer, upholstery and glazing choices. A slatted bench retains rigid slats. Finish interventions preserve actual triangle positions/normals and constructor metric UVs; the common material assignment stage intentionally retains substrate-specific UV phase variation. Existing PBR recipes and texture pools are reused; no new renderer or baking passes are added.

## Realized sampling

A controlled study generated **455 accepted amenities across 256 isolated desk/sofa contexts**. All six categories occur as the first accepted product somewhere in this cohort. Positions, orientations and sizes below describe admitted products, not proposals. The side classification uses the dominant position coordinate normalized by anchor width/depth; it is an analysis label rather than a stored attachment relationship.

| Category | Accepted | Left / right / rear / front | Heading bins (0.1 rad) |
|---|---:|---|---:|
| Bench | 60 | 26 / 11 / 14 / 9 | 24 |
| DrawerUnit | 76 | 24 / 31 / 19 / 2 | 23 |
| MobileCart | 82 | 17 / 17 / 27 / 21 | 27 |
| Ottoman | 73 | 11 / 12 / 3 / 47 | 29 |
| RoomDivider | 61 | 13 / 24 / 22 / 2 | 27 |
| SideTable | 103 | 40 / 35 / 15 / 13 | 29 |

The controlled drawer test covers 256 metric products: **179 single-bank and 77 two-bank units**, **2–10 drawers**, **60 centimetre-width bins**, and **67 two-bank units whose first rows differ by more than 1 cm**. Narrow units legitimately retain one bank. A separate fixed-size regression verifies that compact five-drawer pedestals remain available. Independent material tests exercise 64 shape/finish interventions for each of four categories.

Across 240 complete generated rooms, category counts are `{"Bench": 63, "DrawerUnit": 238, "MobileCart": 195, "Ottoman": 91, "RoomDivider": 97, "SideTable": 202}`. Drawer bank counts are `{"1": 161, "2": 77}`. These complete-room figures include existing public-program service furniture, whereas the controlled study isolates the changed amenity policy.

| Category | Fixed-envelope coverage at 0.005 / 0.015 / 0.03 | Mean triangles | Median nearest-shape distance |
|---|---:|---:|---:|
| MobileCart | 64 / 64 / 64 | 4,173 | 0.1708 |
| DrawerUnit | 62 / 46 / 13 | 3,778 | 0.0153 |
| Bench | 64 / 64 / 62 | 2,645 | 0.1297 |
| RoomDivider | 64 / 63 / 62 | 2,338 | 0.1228 |

Coverage uses the existing six-view normalized depth descriptor; materials are excluded. The narrow fixed drawer fixture cannot contain two banks, so the metric-size study is essential. An unchanged or lower fixed-envelope coverage value does not measure the new placement/material combinations. These diagnostics do not estimate independent training samples or demonstrate improved transfer.

## Correctness and bounded cost

- **44 focused Rust tests pass**: physical support and mesh bounds, drawer partitions/front clearances, opt-in motion ownership and unchanged rest pose, independent appearance interventions, accepted placement/replay, complete-room clearance, existing botany/refreshment/storage integration, coplanar checks, metrics and catalog contracts. Two pre-existing ignored population tests were not run; explicit population checks cover this iteration.
- **6,848 meshes across all 107 categories** pass the construction audit. A further 256 metric service-furniture inputs supply the render study.
- **240 rooms** pass CPU layout and real-mesh checks: 128 mixed, 16 dense eight-camera rooms and 32 each of open office, lounge and breakroom.
- **72 production-renderer images** cover RGB, geometric normals and semantics for 24 selected metric fixtures. Software Vulkan is restricted to two threads. Galleries were inspected and image/palette/annotation contracts checked. These are object fixtures, not full-room photographic qualification.
- Strict workspace/all-target Clippy with CUDA IPC and WGPU fusion features, formatting, and WASM viewer compilation with human motion pass. The existing `burn-cubecl-fusion 0.22.1` future-compatibility notice remains upstream.

The fixed fixture corpus contains **19,554,167 triangles**, compared with **19,538,091** in the saved local baseline (**+0.08%**). This is a corpus aggregate; individual constructions have different costs.

Three short matched-seed CPU audit pairs had median subprocess times of **4.374 s** for the saved baseline and **4.222 s** for this candidate. The median paired ratio was **0.965**, with individual ratios **0.962–0.978**. Additional pairs alternate execution order. These audits use four internal preparation workers and include validation and JSON reporting, which now includes extra amenity diagnostics. Generated contents differ. This is a bounded CPU check, not a completed-render throughput or production speedup measurement.

## Inspection and evidence

- [Accepted furniture positions and headings](../../out/diversity_review_108/accepted-amenity-placement.png)
- [RGB gallery](../../out/diversity_review_108/renders-color.jpg)
- [Geometric-normal gallery](../../out/diversity_review_108/renders-normal.jpg)
- [Semantic gallery](../../out/diversity_review_108/renders-semantic.jpg)
- [Qualification receipt](../../out/diversity_review_108/qualification.json)

Commands, sources, programs, checks, population receipts and timing observations are in `out/diversity_review_108/`. Render selections cover constructions with varied metric sizes and independent finishes; they are illustrative, not frequency estimates. The prior [service-equipment study](service_equipment_and_placement.md) remains separate retained work.

No hardware GPU/CUDA replay, concurrent-training throughput, browser execution or long-run memory was qualified here. This iteration does not prove photographic realism, prevent every procedural shortcut, or establish an effective independent dataset size. Remaining narrow regimes should be prioritized from realized joint geometry/appearance/placement statistics and downstream evidence, not arbitrary shape noise.
