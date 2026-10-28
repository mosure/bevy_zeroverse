# Object construction and surface composition

Local qualification of generator **105**, capture **v130**, with source SHA-256 `47b46b11ae002952c2e9610b0c5d8d22ce6e2d0ae98259b934b8554c9a198d2f`. Package versions are unchanged. This work has not been published.

## Review and implementation

The review sampled all **107 object categories** and traced shape, material, placement and transform controls. It identified three actionable restrictions: identical item-category priors on every surface, duplicated display-stand constructions, and towel holders whose radius and construction barely changed within each orientation. Flat panels and standardized appliances also have low normalized shape variation, but that alone is not a reason to deform their functional geometry. The audit does not measure appearance or learning utility.

Surface composition now samples a shared room preference and a local surface preference over **27 distinct loose-item categories** in six existing task groups. A positive probability floor preserves every member. Room coherence varies continuously from 0.25 to 0.82; independent random streams preserve deterministic replay. Task membership, functional substitutions, size sampling, support placement and collision admission remain in force. There is no displacement after admission. This introduces variation in room composition and co-occurrence rather than claiming a larger marginal categorical entropy.

The sampled probabilities, room coherence and per-task normalized entropies are exported as distribution metrics. They describe the **proposal prior before functional substitutions and clearance rejection**. Accepted item counts remain separate. These probabilities must not be interpreted as accepted frequencies or an effective dataset-size estimate.

Display construction is separated from laptop construction while retaining the public display APIs. Monitors have six connected base constructions and three riser constructions, sampled base height, support proportions, finishes and mounting position. Rear electronics housings vary in width and height; curved panels use their actual rear curvature for support attachment. Screen geometry, calibration, socket location, object envelope and semantic class remain shared with their existing consumers.

Paper-towel assemblies independently sample roll fullness, core diameter, metric dimensions, base construction, arm construction, holder finish and optional hanging paper. The cardboard core is hollow, the spindle fits inside it, and the free sheet stays attached to the roll and above the base. Horizontal and vertical orientations retain their support contract. Metal, chrome, wood and plastic holders reuse the existing material pool.

## Realized sampling

Across **1,122 furnished-surface proposals**, every member of every task group appears as the highest-probability item somewhere in the cohort. Median normalized task entropy ranges from **0.940 to 0.953**; the largest observed individual probability is **0.512**. These diagnostics show varied preferences without a single-category collapse. They are not independent sample counts.

The 128 mixed-room cohort contains **624 surfaces and 2,760 accepted loose items**. Median occupancy is three items per surface; **16.35%** of surfaces remain empty. Clean surfaces remain part of the distribution.

The geometry audit uses 64 samples per category and the existing normalized shape descriptor. Greedy coverage depends on its distance threshold; it is not a count of perceptually distinct training samples.

| Category | Coverage at 0.005 / 0.015 / 0.03 | Mean triangles | Median nearest-shape distance |
|---|---:|---:|---:|
| Display | 64 / 64 / 62 | 336 | 0.080 |
| Monitor | 64 / 64 / 64 | 1,181 | 0.219 |
| Paper towel | 64 / 64 / 64 | 1,093 | 0.153 |

The monitor descriptor was already saturated at 64 representatives before this change. Its unchanged coverage score does not quantify the additional stand constructions. Construction tests separately observe all 18 base/riser combinations and more than 95 rear-housing proportion pairs. Towel tests observe all eight orientation/base/arm combinations and more than 30 roll-radius bins.

## Correctness and performance evidence

- **54 focused Rust tests pass**: replay, distribution support, mesh validity, support contact, construction bounds, cable routing, display geometry, coplanar checks, placement, kitchen construction and metrics. One explicitly ignored slow kitchen sweep was not run.
- **6,848 object meshes** across all 107 categories pass the mesh audit.
- **240 furnished rooms** pass geometry checks: 128 mixed rooms, 16 dense rooms with eight cameras, and 32 each of kitchen, kitchenette and open-office scenes. These are CPU construction/placement audits, not rendered-room qualification.
- **90 production-renderer captures** cover RGB, geometric normals and semantics from 18 front views and 12 rear views. Front and rear galleries were visually inspected. Every semantic capture contains the expected object and floor classes; there are no invented categorical boundary colors. Object annotations and image lengths are checked by the capture utility.
- Formatting, strict workspace/all-target Clippy with CUDA IPC and WGPU tensor-fusion features, and the WASM viewer check with human motion pass. WASM execution of this change was not tested.

The saved generator-103 geometry baseline and current generator-105 audit contain **19,255,655** and **19,324,475** triangles respectively across the same 6,848 fixtures: **+0.36%**. The intervening generator-104 change affected surface shading. Per-category geometry costs increase where detail was added; no new texture atlas, material-bake pass or renderer pass was added by this change.

Sequential, matched-seed 128-room CPU audit subprocesses took **4.607 s** for the saved baseline and **4.604 s** for the candidate. The executable's internal audit durations were 4.430 s and 4.359 s. The audit uses four preparation workers; Cargo/Rayon and software Vulkan were separately bounded to two. These timings include CPU validation and reporting, omit GPU rendering, and comprise one short pair. They support a bounded regression check, not a speedup or production throughput claim. Fresh hardware CUDA replay, concurrent-training throughput and long-run memory behavior were not measured.

## Inspection artifacts

The local evidence directory is `out/diversity_review_105/`. It contains commands, source provenance, test logs, complete category/room reports, sampled priors, RGB/normal/semantic images and a qualification receipt. Gallery fixtures deliberately share size and finish seed to expose construction differences; they are illustrative selections, not a room-frequency estimate.

- [Front RGB gallery](../../out/diversity_review_105/renders-color.jpg)
- [Rear RGB gallery](../../out/diversity_review_105/rear-renders-color.jpg)
- [Geometric-normal gallery](../../out/diversity_review_105/renders-normal.jpg)
- [Semantic gallery](../../out/diversity_review_105/renders-semantic.jpg)
- [Local qualification receipt](../../out/diversity_review_105/qualification.json)

The prior [surface-response qualification](surface_response.md) covers normal/roughness and coating changes retained in this checkout. Neither audit establishes photographic realism, prevents all procedural shortcuts, or measures downstream generalization. Remaining low-coverage categories should be assessed using joint geometry, appearance and composition evidence, rather than expanding arbitrary shape noise to improve one score.
