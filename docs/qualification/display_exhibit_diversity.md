# Display and exhibit diversity

The generator samples construction, content composition, and finish independently
for elevators, displays, phones, boards, artwork, and public-space exhibits.
Generator revision 54 records the changed seeded construction contract. It keeps
the existing semantic classes, placement reservations, material slots, texture
resolution, and annotation paths.

## Construction and content

| Category | Sampled controls and relationships |
|---|---|
| Elevators | Center-opening and telescoping banks, leaf count, portal width/offset/height, recess, parked and intermediate positions, independent surround/door/car/floor finishes, handrails, lining bays, lighting, call-panel side/height, and numeric floor indicators. Fully parked leaves fit their pockets. |
| Monitors and televisions | Aspect, curvature, tilt, panel depth, bezel, unequal bottom chin, casing, stand footprint, and offset rear attachment. Stand/cable inlet connectivity and reserved bounds are preserved. |
| Phones | Rounded housing sections, corner shape, unequal display borders, side controls, body finish, front/back presentation, and camera islands. Screens remain separate from the casing. |
| Whiteboards | Framed/frameless construction, frame finish and section, marker palettes, free clusters or divided compositions, handwritten content, occupancy, erasure and finish. |
| Pinboards | Continuous paper position, rotation and scale, bounded clusters, divider clearance, pins and writing. Rejection preserves visible backing and prevents attachment overlap. |
| Artwork | Palette harmony, light/dark paper, composition, focal point, orientation, negative space, contour drawings, poster layouts, cut-paper shapes, existing landscapes/still life/botanical subjects, substrate and stepped frames. |
| Sculptures | Connected lobes, welded loops, faceted columns, turned forms, and cantilevered discs, with continuous curvature, breadth, phase, taper, thickness, element count and independently sampled finish. |
| Poster stands and exhibition panels | Pedestal, A-frame and rail supports; fitted panel angle, proportions and border; independent front/back content; irregular framed-art placement on display partitions. |

Desktop, phone and television content have different composition priors and
full-image frequencies. Software panes include plans, calendars, images,
documents, plots and existing application families. Header presence, margins,
panel layout, palettes and content seeds vary. Full-image content uses the same
bounded illustration program without baking unused material maps.

These controls remain conditional: a door bank must fit its pocket, a display
must fit its mounting envelope, and pinups must clear their frame and divider.
Realized placements and geometry are measured rather than multiplying nominal
parameter counts into an asserted dataset size.

## Local qualification

```sh
cargo test --locked --lib display_exhibit_population_qualification -- \
  --ignored --nocapture --test-threads=1
```

This opt-in CPU study writes `out/display_diversity/report.json` and local PNG
contact sheets. It checks 64 seeds for each of 11 object categories at a fixed
metric envelope and fixed finish seed, 64 content maps for each of five surface
categories, and 64 complete rooms. Half the rooms use Mixed; the remainder cover
elevator landings, galleries, showrooms and conference rooms. The ordinary tests
also cover 128 seeds per object category at dimensional extremes, sliding-panel
clearance through the complete opening range, replay, attachment placement,
visible controls, floor seams, semantic ownership and material mip contracts.

Shape coverage uses six 24-by-24 orthographic depth maps, with ordered greedy
representatives at normalized mean-distance thresholds 0.005, 0.015 and 0.03.
It excludes colors, texture coordinates, triangle counts and object scale from
the diversity distance. The content diagnostic reports exact pixel hashes and
nearest-neighbor distances of 16-by-16 RGB area means, which suppress fine grain.
Both metrics are threshold-dependent screening tools; neither measures semantic
embedding volume, photographic realism, or downstream generalization.

CPU timings cover individual mesh construction and content synthesis. Complete
room mesh-validation time includes checking triangles and is not a generation
throughput benchmark. The geometry sheet uses a CPU flat-shaded schematic;
material sheets show generated base-color content. They do not substitute for
PBR render or training qualification. GPU work remains deferred during active
training, and the matched study and generated artifacts remain local.
