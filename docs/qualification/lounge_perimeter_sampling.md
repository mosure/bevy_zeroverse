# Lounge and perimeter sampling qualification

The current generator samples conversation groups in furniture coordinates and
supplemental service objects in the actual envelope's wall coordinates. Construction,
placement and finish parameters have separate seeded programs. This broadens the
admitted population without rotating fixed architectural attachments arbitrarily.

## Active parameters

| Program | Sampled properties and constraints |
|---|---|
| Conversation groups | Sofa width/height/depth, group offset and small heading offsets; proportional coffee-table dimensions, lateral offset, access gap and yaw; one to three proposed seats with continuously varied angular spread and setbacks. A symmetric proposal mode remains at 18%. Rejected furniture never anchors seats. |
| Collections | 58% of lounge seating proposals share the sofa's finish seed. Other seats retain independent finishes. Geometry remains independently sampled; sharing a finish does not imply identical chair construction or substrate. |
| Supplemental storage | Cabinet width 0.65–1.85 m, height 0.66–1.28 m, depth 0.33–0.59 m; bookcase width 0.65–1.55 m, height 1.05–2.08 m, depth 0.26–0.46 m. Heading stays close to the sampled wall normal. |
| Supplemental plants | Density-dependent occupancy and ordering; width 0.43–1.03 m, height 0.68–1.98 m, depth/width 0.78–1.13, free yaw and wall setback. Existing botanical species selection and other supported-plant programs remain active. |
| Ottomans/poufs | Width 0.40–0.97 m and height 0.30–0.50 m; near-square/elliptical or elongated proportions. Feet, sled, plinth and low upholstered bases vary in span, section and height. Rounded boxing varies independently from its optional top cushion. |
| Upholstery finishes | Independently seeded fabric/leather body and top panels, with wood, metal, chrome or polymer frame substrates. Existing bounded PBR slots and metric UVs are reused. Material assignment may change texture phase; it cannot change positions, normals or metric UV scale. |

These are proposal distributions. Ordinary footprint, floor-support, ceiling,
opening, furniture and circulation admission determines the realized population.
There are at most four supplemental placement attempts per product and at most
eight products. No room-wide unbounded packing search or per-frame sampler is added.
Archived manifests without a spatial program retain their furnishing path.

Upholstered bodies have broad planar support patches and rounded edges. Top cushions
contact the body instead of touching at a single ellipsoid pole. Smooth normals
cross the boxing rings and UV seams. The rounded-shell primitive caches its angular
profile on the stack; manufactured-shell positions, normals, UVs and indices remain
bitwise identical in the fixed-fixture comparison.

## Local evidence

- 64 fixed-envelope ottoman meshes: 64/64/60 representatives at normalized six-view
  depth-distance thresholds 0.005/0.015/0.03. This measures sampled geometry, not
  perceptual novelty, photographic realism or effective training dataset size.
- 288 ottoman dimension/seed cases check bounds, grounding, contact patches,
  continuous cloth normals, semantic ownership and a 2,400-triangle ceiling.
- 768 finish interventions check identical shape/normals and constructor metric
  UVs, with deterministic replay after ordinary material binding.
- A 60-room population checks realized storage dimensions, plant widths and
  furniture-relative seat poses. An additional 80-room study covers every one of
  the 20 concrete profiles, four densities, three cameras and 25% human density.
- The existing 1,024-room mixed-profile seating/contact regression passes, including
  seeded replay. Constrained activity and reported ceiling/geometry seeds remain
  covered by their existing tests.
- A 160-room, three-camera CPU probe, repeated in both execution orders, separately
  measures manifest construction, camera placement and geometry validation/building.
  It does not measure GPU rendering, texture synthesis, readback or training throughput.

The local gallery, exact inputs, construction metadata, per-room timings and mesh
fingerprints are stored under `out/lounge_regularity/`. They are not part of the
published project page or paper. To reproduce the portable geometry/placement
study, run `cargo test --lib lounge_regularity_qualification -- --ignored`.

The fixed-fixture mesh comparison covers eight seeds for all 106 object categories.
The 105 categories outside this ottoman construction change match exactly, including
material bindings and UVs. Native workspace Clippy and the WebGPU/WASM viewer with
human motion are checked separately; this qualification does not claim a new
hardware-rendered photometric or concurrent-training benchmark.
