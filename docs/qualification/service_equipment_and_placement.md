# Service equipment and support-local placement

Local qualification of generator **107**, capture **v132**, source SHA-256 `0c9ccf421f1a243187cf1855819d2862a2840cb339aa7a9d0a6666c5970d250c`. Package versions are unchanged. These changes have not been published.

## Bottlenecks and implementation

The review sampled the existing 107-category geometry audit and traced placement and material dependencies. It identified an especially strong spatial regularity: every accepted tabletop power strip in the earlier 128-room cohort occupied the same normalized rear-right location, with 30 mm height and only ±0.07 radians of rotation. Beverage dispensers also retained a fixed three-tap fountain arrangement and seven tray bars, while coffee makers reused panel, tray and carafe proportions within their otherwise distinct brewing constructions.

**Distribution hardware.** Power strips now sample compact and elongated metric products, one or two rows, socket count and orientation, casing height, end caps, and independently seeded chassis/socket materials. Socket centers and the actual deck height supply the mains attachment point. The USB and inlet connections retain their physical directions. Geometry, routing and exported metrics consume the same program. Metric ranges are 0.17–0.46 m long, 0.025–0.052 m high and 0.048–0.13 m deep, conditioned on compact/elongated construction; dimensions are not sampled independently over that bounding range.

Placement chooses one product before up to eight attempts. Edge-related proposals span all four sides; other proposals use the existing surface composition sampler. Heading, edge setback and position vary continuously; a seeded half-turn also varies the inlet end of aligned products. Existing clutter permissions, supported-object clearance, room-envelope checks and subsequent cable routing remain in force. Failed attempts do not resample smaller products or fall back to an axis-aligned fixed position. A separate module owns this placement policy.

**Brewers.** Existing espresso, capsule, automatic and filter constructions now have independently sampled base, cup-bay, front projection and control-panel proportions. Pressure machines use three control constructions with one to four controls. Trays vary in width, depth, height, grille direction and three to ten bars. Filter carafes vary in radius, height and handle side; basket height derives from the actual carafe clearance. These are functional proportions, not arbitrary surface distortion. The rear service connector remains compatible with wiring.

**Beverage dispensers.** Enclosed fountains sample one to four taps, limited by available width, alongside varied chassis depth, head projection, working-bay height and product-label layout. Tank units vary their body/top proportions. Trays independently sample width, depth, grille direction, four to eleven bars and trim material. Actuators overlap their nozzles rather than merely touching at rounded edges. The sampled tank count now records the count that fits the admitted width. Gravity-fed units still have no fictional power connection.

Additional metrics report the built control/tap counts, working dimensions, casing/trim choices and support-relative strip positions/headings. Fields belonging to unused construction branches are omitted from aggregate content metrics. The object audit retains the full sampled program for reproducibility. Material assignments reuse the existing procedural PBR pool; this pass does not change material synthesis or add rendering/baking passes.

## Realized diversity

The current 128-room cohort contains **198 tabletop strips** on tables, desks and coffee tables. Their support-quadrant counts are **48, 44, 51 and 55**. There are **194 occupied position bins** at one-percent support resolution, **49 heading bins** at 0.1-radian resolution after reducing headings modulo a full turn and **27 height bins** at millimetre resolution. These describe accepted placements, not only proposals. The increase from 106 accepted tabletop strips reflects broader, bounded admission attempts; it is not a change to the initial equipment-request probability.

The full 240-room cohort contains **346 power strips, 82 coffee makers and 79 beverage dispensers**. The fixed-envelope 64-sample construction audit covers both socket-row counts, all four socket headings, both cap constructions, all four brewer constructions and all four dispenser constructions. Separate 256-sample tests with metric size variation cover all 16 row/cap/heading combinations, all 12 pressure-brewer control-layout/count combinations, both grille directions across every allowed bar count, and all four fountain tap counts. Narrow fixtures legitimately limit tap count; the fixed dispenser fixture reaches three taps while the metric-size test reaches four.

| Category | Shape coverage at 0.005 / 0.015 / 0.03 | Mean triangles | Median nearest-shape distance |
|---|---:|---:|---:|
| Coffee maker | 64 / 62 / 49 | 1,915 | 0.0345 |
| Beverage dispenser | 64 / 64 / 59 | 2,744 | 0.0401 |
| Power strip | 61 / 40 / 11 | 2,371 | 0.0125 |

Shape coverage uses the existing normalized six-view depth descriptor and greedy distance thresholds. The power-strip score remains low at the coarsest threshold: row/port details and placement diversity can change without substantially changing its outer silhouette. This score is neither a perceptual sample count nor a measure of learning utility. Standardized products and flat artwork should not be distorted just to increase it.

## Correctness and performance

- **46 focused Rust tests pass**, covering replay, metric size extremes, socket-cell separation, physical cable endpoints, support-local placement coverage, appliance bounds and semantics, existing wiring/refreshment/floor behavior, coplanar checks, content metrics and the object catalog. Two pre-existing ignored population tests were not run; the explicit studies below cover this change.
- **6,848 meshes** across all 107 categories pass the object audit.
- **240 rooms** pass CPU layout/geometry checks: 128 mixed rooms, 16 dense rooms with eight cameras, and 32 each of cafe, kitchen and cafeteria programs.
- **54 production-renderer captures** cover RGB, geometric normals and semantics for 18 selected fixtures, using software Vulkan with two threads. All three galleries were visually inspected. Semantic PNG pixels exactly match the expected palette classes; capture sizes, finite values and object annotation counts also pass. This is selected-object rendering, not full-room RGB qualification.
- Strict workspace/all-target Clippy with CUDA IPC and WGPU tensor-fusion features, formatting, and WASM viewer compilation with human motion pass. Cargo still reports the pre-existing future-compatibility notice from `burn-cubecl-fusion 0.22.1`; workspace Clippy has no errors.

The fixed 6,848-fixture corpus contains **19,538,091 triangles**, an increase of **0.26%** over the saved generator-106 audit. The 128 mixed-room candidate cohort contains 10,197 checked object envelopes and 73,780,065 triangles, compared with 10,150 and 73,835,030 in the matched-seed baseline. Broader equipment admission has not reduced overall object count in this cohort.

Five sequential matched-seed CPU audit pairs had median subprocess durations of **4.323 s** for the saved baseline and **4.173 s** for this candidate. The median paired candidate/baseline ratio was **0.950**; individual ratios ranged from **0.912 to 1.157**. The first pair was slower for the candidate (3.986 s versus 4.612 s), so four additional pairs alternated execution order rather than drawing a conclusion from that run. All runs passed. Each audit used four internal preparation workers and included validation and reporting. This bounded check found no consistent slowdown; scene contents change with the generator, so it does not establish a controlled production speedup. Hardware rendering, concurrent training, browser execution and long-run memory were not measured.

## Local evidence

Commands, sampled programs, accepted placements, test logs, CPU timings and source provenance are under `out/diversity_review_107/`. Selected render fixtures keep their envelope, finish seed and studio lighting fixed to expose construction differences. They are illustrative selections, not room-frequency or appearance-diversity estimates.

- [RGB construction gallery](../../out/diversity_review_107/renders-color.jpg)
- [Accepted placement distributions](../../out/diversity_review_107/accepted-strip-placement.png)
- [Geometric-normal gallery](../../out/diversity_review_107/renders-normal.jpg)
- [Semantic gallery](../../out/diversity_review_107/renders-semantic.jpg)
- [Qualification receipt](../../out/diversity_review_107/qualification.json)

The previous [handset and wall-light qualification](handsets_wall_lights_and_floor_services.md) and [surface-response qualification](surface_response.md) cover retained changes. Remaining narrow regimes should be assessed jointly across geometry, appearance, composition and actual training outcomes. This work does not establish an effective independent dataset size, eliminate procedural memorization, or prove photographic realism or improved downstream transfer.
