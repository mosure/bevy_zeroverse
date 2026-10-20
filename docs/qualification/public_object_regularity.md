# Public furnishings and small-object regularity

This qualification examines accepted geometry and placements, not the number of
possible floating-point seeds. Real products have useful common structure: a mug
needs an opening, a lectern needs a stable base, and an elevator needs a service
core. Diversity should vary those constructions without invalidating the physical
relationships that supervise reconstruction.

## Construction and placement contracts

| Category | Independently sampled construction and relationships |
| --- | --- |
| Ceiling fixtures | Ring diameter no longer inherits the narrow side of a fluorescent fixture. Circuit scale, individual size, aperture and border vary; photometry uses the identical realized emitting area. |
| Mugs, cups and bottles | Curved hollow profiles, belly/waist/shoulder/neck proportions, wall thickness, fill/empty state, faceted or smooth mug bodies, attached handle curves, saucers, fitted sleeves/lids and bottle ribs. Metric proportions vary before surface admission. |
| Vases and bowls | Connected foot, belly, shoulder, neck and mouth profiles; hollow interior, optional fluting/faceting, independent finish and metric dimensions. |
| Notice and exhibition boards | Unequal margins, panel elevation, panel width/offset, sparse or dense compositions, rotated attachments, differing image/text content and support construction. Notes respect frames, dividers and each other. |
| Information stands and lecterns | Plate, elliptical, radial-foot and twin-sled supports; A-frames, pedestal, paneled and braced lecterns; variable panel dimensions, inclination, height, lip and microphone. |
| Teaching boards | Variable panel width/height/elevation, single or multiple leaves, tilted/braced frames, wheeled or fixed feet, optional tray, separate content on both faces. |
| Display cases and tray returns | Category-specific metric heights, base/glazing proportions, open or closed structures, irregular shelf intervals and supported contents. Miniature products use category-appropriate dimensions. |
| Plinths and sculpture | Low platforms through tall pedestals; tapered, circular-column, leg-frame and twin-slab structures. Nine sculptural construction branches with continuous bend, phase, breadth, taper, element and base parameters. Exhibits are sampled conditionally on their real support footprint. |
| Drink dispensers | Single/multiple gravity tanks, thermal urns and recessed fountain stations; variable reservoirs, shoulders, necks, fill, taps and materials. Only powered constructions expose an electrical port. Required service equipment is placed before optional countertop equipment. |
| Queue barriers | Webbing, sagging rope, rigid rails and infill panels; varied feet, posts, height, span, sag and finishes. |
| Elevators | Recessed, pocketed sliding door banks inside a masonry service core. Prefer floor-to-ceiling cores; low overhead installations can require a shorter enclosed core. Car headroom, controls, opening pockets and the clear approach remain part of admission. |
| Service counters and extraction hoods | Counters vary between open prep frames, paneled/slatted storage carcasses and tapered bodies, with unequal bay widths, independent tops and optional attached footrails. Hoods have box, tapered, curved and glass canopies with varying flues, filters and lights. Counter tops retain their entire rectangular usable area. |
| Adjacent furnishings | Locker row/column partitions, metric floor-lamp dimensions, banquette dimensions and public furniture heights have category-specific distributions. Banquette seat height remains independent of overall back height. |

Programs live in category-specific modules beneath
`src/scene/procedural_indoor/objects/public/`; vessels and beverages have separate
builders. The manifest records admitted metric envelopes and seeds. Material
choices remain separate from structural streams where coordinated furniture sets
are appropriate. Geometry changes advance the capture identity and generator
version; replay requires both identity and seed.

## Reproducible qualification

The library test `public_object_regularity_population_qualification` writes local
artifacts in `out/public_regularity/`. It is explicitly ignored in routine tests
because it performs an artifact-producing study. Run it with:

```sh
cargo test --locked --lib public_object_regularity_population_qualification -- --ignored --nocapture
```

The study checks 21 categories at 64 seeds each, with fixed per-category envelopes
and finishes. Six orthographic 24 × 24 depth maps measure geometry independently
of texture, color and raw triangle count. Ordered greedy coverage at normalized
union-pixel depth differences of 0.005, 0.015 and 0.03 is a diagnostic, not an
embedding-space volume or an estimate of training-set capacity. A mostly planar
notice board can have low depth coverage despite varied attachment/content layouts;
its appearance and layout must be assessed separately.

A second stage generates 320 rooms: all 20 concrete profiles, 16 seeds per
profile, three density settings, three cameras and nonzero human frequency.
It validates all layouts, every reviewed object mesh, and full-room geometry for
the first two seeds of every profile. It records accepted size distributions,
elevator-core counts and CPU construction/preparation timings. Routine tests also
cover determinism, category semantics, transformed bounds, winding/normals,
plinth support height, rotated exhibit containment and emitting-area agreement.

## Current bounded qualification

The current accepted-distribution qualification and its measured results are in
[Procedural regularity and shortcut audit](procedural_shortcut_audit.md).
Its routine test covers 320 rooms and checks complete meshes in 40 of them.
The metric-extreme/replay test covers 21 categories × 64 seeds (1,344 cases).
A separate fixed-envelope service test checks counters, hoods and floor lamps
at 64 seeds each, including the counter's exact top support area.

Local current artifacts are `out/regularity_review/accepted.json`,
`service-geometry.json`, `service-geometry.png`, and `schematic-scenes.json`.
The service PNG is a CPU geometry preview. The depth-map coverage statistic
deliberately omits texture and material. The current twelve-sample counter/hood
PBR review is in `out/regularity_review/renders/`, with the local combined gallery
at `out/regularity_review/index.html`; it is an illustrative rendering check, not
a photographic-realism benchmark.

The broader explicitly ignored study described above can regenerate the complete
21-category depth-map grid and per-room reviewed-mesh timing. Its output is only
current when its recorded generator identity matches the compiled source.
The earlier illustrative PBR gallery under `out/public_regularity/` retains its
own source provenance; it must not be presented as a render of subsequently
changed construction or placement programs. No local study is automatically
included in the published project page or paper.

## Interpretation and remaining limits

These measurements can expose repeated envelopes, low shape variation and placement
bottlenecks. They cannot prove that a neural network cannot memorize the generator,
or establish transfer to photographs. Use compositional construction/appearance
holdouts and correct-view versus repeated/mismatched-view ablations downstream.
Do not multiply parameter ranges, count dormant branch parameters, or treat color
changes as new geometry. Room-program topology, bounded material atlases and
manufactured regularities remain additional factors to audit independently.

CPU construction timings do not establish GPU throughput or concurrent-training
performance. Local study artifacts are not automatically included in the project
page, paper or release publication.
