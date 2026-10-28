# Handsets, wall lights and floor services

Local qualification of generator **106**, capture **v131**, source SHA-256 `c5c920b0cbf1bbe432e14afa0ca2fc06ef5f74c9e0ce7f582d26123e30f11904`. Package versions are unchanged; this work is not published.

## Review and changes

The review reuses the 107-category construction audit to identify narrow samplers, then traces their material, placement and transform dependencies. Low normalized shape variation is a diagnostic, not a reason to distort standardized objects. Flat artwork, counters and appliances still need joint appearance/layout assessment; this pass addresses three concrete restrictions rather than claiming to eliminate every procedural shortcut.

**Phones** now independently sample bare, bumper and padded cases; four rear-camera arrangements; one to four lenses; asymmetric side controls; rounded-body proportions; screen borders and position; and shell/back materials. Rear-camera islands fit inside the rounded back before lenses are packed. Camera bumps and rims reserve their height before meshing. Hollow bumper construction avoids overlapping solid caps. Face-up/face-down placement, support contact, full-screen UV mapping and phone semantic labels remain intact. Finish seeds change substrates without changing geometry or orientation.

**Wall lights** use hollow cone, drum, bell, pleated or cage shades with separate bulbs, diffusers, connected mounting arms and rectangular/oval wall plates. Dimensions, taper, offsets, arm construction and material assignment vary within physical mounting envelopes. Fixture width spans 0.12–0.40 m, with correlated height and projection limits. The geometry program supplies the emitter used by lighting and the corresponding metrics. Elliptical shade normals use inverse-scale correction. Mesh construction reuses existing luminaire primitives and material pools.

**Floor outlets** no longer always search behind the host desk first. A seeded side order considers all four sides within the existing bounded search. Installation orientation mixes furniture-related quarter turns with local variation and freely oriented installations. Metal, chrome and plastic finishes are sampled independently of outlet shape and operating state. Candidate transforms are chosen before admission and cable routing; nothing is displaced after clearance approval. Closed-lid restrictions and physical cable port positions remain shared with the geometry.

The phone and wall-light programs live in separate modules; material, metrics, photometry and geometry consumers use the sampled construction rather than duplicating dimensions. Existing room-level and surface-level item preferences from the [catalog qualification](catalog_and_display_sampling.md) remain enabled.

## Sampled coverage

The fixed-envelope audit samples 64 constructions per category. Its greedy coverage at three shape-distance thresholds is a representation-dependent diagnostic, not a count of distinct training examples.

| Category | Coverage at 0.005 / 0.015 / 0.03 | Mean triangles | Median nearest-shape distance |
|---|---:|---:|---:|
| Phone | 63 / 50 / 34 | 1,209 | 0.0200 |
| Wall light | 64 / 64 / 64 | 2,412 | 0.3692 |
| Floor outlet | 48 / 48 / 25 | 654 | 0.0158 |

The 64 phone samples include all three cases, four camera layouts and lens counts from one through four. Nineteen expose their backs; 45 expose screens. Sconces cover all five shades, all three arms and both plate shapes. Separate 256-sample tests with metric dimension sampling cover all **24 case/camera-layout/face combinations**, all **30 shade/arm/plate combinations**, more than 230 handset dimension bins and more than 100 shade-height bins. These combinations do not multiply into an effective dataset-size estimate.

Across the 240-room audit, accepted instances include **426 phones, 168 wall lights and 138 floor outlets**. In the 128 mixed-room subset, 77 rooms contain phones, 71 contain wall lights and 55 contain floor outlets. The gallery subset has no floor outlets; this is retained as a contextual outcome rather than forcing every category into every room. Full per-stratum counts are in `out/diversity_review_106/sampling-summary.json`.

## Correctness and cost

- **50 focused Rust tests pass**, including deterministic replay, actual mesh bounds, screen/semantic contracts, mounting/emitter agreement, floor services and cable routing, coplanar surfaces, existing clutter/display behavior, and metrics.
- **6,848 meshes** across all 107 categories pass the object audit.
- **240 furnished rooms** pass CPU geometry validation: 128 mixed rooms, 16 dense rooms with eight cameras, and 32 each of cafe, classroom and gallery programs.
- **54 production-renderer captures** cover RGB, geometric normals and semantics for 18 selected fixtures. All three galleries were visually inspected. Semantic pixels match only the expected object, floor and wall/background classes, with at most one byte of palette rounding error. Capture sizes, finite values and object annotation counts also pass.
- Strict workspace/all-target Clippy with CUDA IPC and WGPU tensor-fusion features, formatting, and the WASM viewer compilation with human motion pass.

The fixed 6,848-fixture corpus contains **19,486,991 triangles**, 0.84% more than the saved generator-105 corpus. Additional geometry is concentrated in camera clusters and hollow lamp assemblies; no new rendering pass, texture atlas or baking pass is introduced. Floor-outlet geometry cost is unchanged.

Sequential, matched-seed 128-room CPU audit subprocesses took **4.792 s** for the saved baseline and **4.759 s** for this candidate. Internal audit durations were 4.499 s and 4.520 s respectively. These include validation/serialization, use four internal preparation workers, and form one short pair. They show no material regression in this check, not a measured production speedup. CPU construction timing does not establish GPU rendering or concurrent-training throughput.

## Render inspection and boundaries

Render evidence and the final source-bound qualification receipt are stored in `out/diversity_review_106/`. Selected fixtures share dimensions and finish seeds to expose construction differences. The floor-outlet gallery depicts open units; independent finishes and closed states are covered by separate mesh/program tests. The gallery is illustrative, not a frequency estimate.

- [RGB construction gallery](../../out/diversity_review_106/renders-color.jpg)
- [Geometric-normal gallery](../../out/diversity_review_106/renders-normal.jpg)
- [Semantic gallery](../../out/diversity_review_106/renders-semantic.jpg)
- [Qualification receipt](../../out/diversity_review_106/qualification.json)

Fresh hardware GPU/CUDA replay, full-room RGB inspection, browser execution, concurrent-training throughput and long-run memory were not qualified. No claim is made of photographic realism, memorization prevention or improved downstream transfer. The previous [surface-response qualification](surface_response.md) covers the normal/roughness changes preserved in this checkout.
