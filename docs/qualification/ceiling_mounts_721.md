# Ceiling mounts, life-safety fixtures and floor schematics

The 0.63.4 generator uses identity 120 / `capture-v145`, with
`ceiling_camera=8` and `equipment=26`. The release keeps Burn 0.22.1.

## Rejected training rooms

The registry 0.63.3 generator reproduced the rejection of 72154853 and
72111767 under the training fixture: eight moving views, mixed room/ceiling
mounts, mixed fisheye/pinhole lenses and baseline 0.5. Both scenes already
passed layout and triangle validation. Neither could place its first camera:
all 3,072 initial proposals failed placement or clearance.

Their sloping ceilings contain structural ribs. The planner attached cameras
to the nominal roof behind those ribs, while collision tests reserved the
real framing. Mounting now uses exposed planar undersides of solid beams,
I-section flanges and the flat portions of fluted ribs. Open-web trusses retain
collision reservations and do not become fictitious mounting plates. The
existing conservative body, near-lens, optical-axis and swept-volume tests
remain active. Rail qualification checks every support-boundary crossing;
a dedicated fixture rejects a 2 mm unsupported gap between temporal samples.

Both reported seeds now pass eight-view placement, complete layout/geometry
validation, camera-role/lens/motion constraints, directed-overlap constraints,
and deterministic replay. Four neighboring seeds and schematic seed 72184257
also pass. Camera search for the two repaired seeds took approximately 0.14 s
and 2.28 s on the shared native host. These are bounded CPU placement timings,
not isolated renderer or concurrent-training throughput measurements.

## Fixture and schematic correctness

Alarm strobe-lens rear caps and pull-station hinge rear caps previously shared
positive surface area with their housing fronts. Both now seat 1 mm inside
the housing; visible front extents, part counts and semantic labels remain.
The regression checks triangle intersections across 192 devices spanning all
three life-safety categories, four variants and multiple metric scales.

Seed 72184257's large unlabeled schematic rectangles were the allocation
boxes of string-light strands and a ceiling fan. Floor schematics now omit
these and other ceiling-only fixtures while retaining floor/wall furnishings,
clutter, architectural levels, cameras and predictions. Actual floor levels
show their metric elevation in both live and exported diagrams. This is a presentation
filter; it does not remove scene geometry or change capture annotations.
The shared schematic document feeds both SVG/PNG exports and the live Bevy
2D viewer. Tests verify footprint identities, recorded root transforms,
serialization and raster export.

The editor camera-glyph default is 4× in native/WASM configuration, CLI and
sparse URL sharing. Schematic glyphs retain their 1× default. Explicit scales
remain supported without changing camera calibration or metric trajectories.

## Reproduction and qualification boundary

Run the core test filters `production_721`, `cameras::ceiling::surface::`,
`objects::equipment::signals::`, `annotation::schematic::`,
`camera::gizmo::` and `app::query::tests::` with the pinned toolchain.
Local diagnostic inputs, diagrams and release receipts are under
`out/review_721` and `out/release_0634` (excluded from crate packages).
Hardware CUDA replay, browser runtime and a concurrent-training memory or
throughput plateau are not requalified by these CPU geometry tests.
