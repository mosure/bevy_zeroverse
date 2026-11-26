# Functional indoor spaces

The indoor generator has explicit programs for hallways, stair halls, courtyards, elevator landings, cafes,
cafeterias, restaurants, fitness rooms, cycling studios, volleyball gyms, grocery and clothing stores, classrooms,
lecture halls, showrooms and galleries, alongside kitchens and kitchenettes.
`PublicSuite` samples compatible adjacent functions. These programs use the
existing `ProceduralIndoor` scene and capture pipeline. Cornell, object,
primitive and semantic-room scenes remain available.

```sh
cargo run --bin bevy_zeroverse -- --scene-type procedural-indoor \
  --indoor-layout lecture-hall --indoor-seed 10000
cargo run --example indoor_validate -- --layout gallery --seed 20000 \
  --audit-seeds 1024 --audit-geometry --renders 64 --cameras 3 \
  --width 512 --height 512 --labels --co-visibility --no-raw \
  --output out/gallery_qualification
```

The native/WebGPU controller's activity selector uses the same enum as CLI,
URL and Rust configuration. URL spelling follows the existing enum parser,
for example `indoor_layout=lecture-hall`. Changes use the existing explicit
Apply/Next controls. `Mixed` draws uniformly from the full list of room programs.

Courtyard metrics include the open roof fraction, slab thickness, column count, soffit and fascia materials. The public-program audit also records rooted beds, container species, paving, seats and direct and diffuse metered daylight.

Bike proportions use commercial indoor cycles as a reference. Treadmill placement uses the clearance specified in the Life Fitness Integrity operator manual. Camera proposals can frame the body of an admitted station alongside broader room and person views. The workout audit records accepted bikes, treadmills, mats and benches per room, aisle width and the instructor station where present. [Keiser M3i dimensions](https://www.keiser.com/shop/cardio-equipment/m3i-indoor-bike) · [Life Fitness Integrity operation manual, page 17](https://fitnesssuperstore.info/pdfs/Life%20Fitness%20Integrity%20Series%20Treadmill%20CLST%20Operation%20Manual.pdf)

The public-program audit records the table count, seats, supported objects, and each dressed table’s object ID, weave and maximum hem drop for restaurants and cafés. Rust clients can recover the same measurements from an archived table with `objects::tables::dining_linen`. The cloth retains its table’s semantic class and bounding box.

Grocery stores can include a chilled department. The open side of each cold case faces its reserved customer aisle. Cabinet ends can be insulated or glazed independently, and the rear intake has a recessed grille. Its shelves step back above the return grille and stock dairy tubs, gable-top cartons and bottled drinks in product runs with variable fill. The public-program audit records ambient, produce and refrigerated fixture counts per zone and the construction parameters of each cold case. Rust clients can read the same parameters from a manifest with `objects::public::refrigerated_case`. The case section uses commercial open multidecks as a reference. [Hussmann ID5SU cross-section and merchandising layout](https://www.hussmann.com/ns/Data_Sheets/DS_ID5SU_0539599_EN.pdf)

## Construction and use

| Program | Construction and placement |
| --- | --- |
| Hallway | A straight passage with rooms on one or both sides, fitted doors, glazed or opaque frontages, optional wall guards and waiting seats. Door approaches and the continuous walking aisle are reserved before furnishing |
| Stair hall | A double-height foyer with a lower entrance, guarded staircase, wall-backed upper gallery and optional intermediate landing. Flight width, direction, framing, tread finish and balustrade vary with the seed |
| Courtyard | A paved open-air garden enclosed by four covered walks. The entrance walk passes between columns. Off-axis raised beds combine a tree or shrub with lower planting in soil, gravel or mulch. Container gardens and outdoor benches occupy the remaining bays |
| Elevator landing | Telescoping door banks, recessed jambs/thresholds, call controls, indicators, bounded illuminated car cavities, handrails, waiting benches and information stands |
| Cafe | Modular service counter, glazed display case with real shelves/contents, coffee equipment and beverage dispensers, small tables, banquettes and contextual dishware. Some venues fit their dining tables with coordinated linens |
| Cafeteria | Serving counters, shelf or slide-rail tray returns, beverage taps, optional guarded hot-food wells, larger dining tables, reserved serving queue and through circulation |
| Restaurant | Table-service groups, paired booths, upholstered banquettes and a waiter station with staff access. Restaurants can dress all, some or none of their tables. Tables share the venue’s woven stock, with individual cuts and hem depths. Dishes rest on the textile at tabletop height |
| Fitness room | Free-weight benches, dumbbells and mats, with optional barbell stations, stationary bikes and treadmills. Full cages and half racks have bored uprights, adjustable catches and bare or symmetrically loaded bars. Each treadmill reserves 2 m of clear floor behind the belt and 0.5 m on either side |
| Cycling studio | Adjustable bikes from a shared equipment line face an instructor in staggered banks. The front of the class can occupy either end of the room. Each station reserves mounting space and the central aisle stays open |
| Gymnasium | A full-size or community practice volleyball court, a tensioned open net on padded posts, inlaid court lines, sealed hardwood, high lighting and optional loose benches or tiered spectator seating outside the run-off |
| Grocery store | Open refrigerated multidecks, double-sided dry-goods shelves, partitioned produce islands, a staffed conveyor or basket register, optional self-checkout stations and clear browsing routes connected to crossing aisles |
| Clothing store | Sales rails with shirts, dresses and outerwear, folded-stock tables, a cash wrap, optional self-checkout stations and single or paired fitting rooms facing a shared aisle. Collections share cuts and dye lots |
| Classroom | Single, banked or folding teaching boards, lecterns, lockers, individual writing surfaces, tablet-arm seating and continuously varied seating fields |
| Lecture hall | Curved audience rows, linked-frame folding seats, explicit floor tiers, intermediate stair treads, teaching platform and guards at exposed drops |
| Showroom | Product plinths, information stands and vitrines; coordinated content collections and shared finish identity |
| Gallery | Angled exhibition panels with framed artwork, sculptural plinths and vitrines; separate viewing/circulation space and exhibit lighting |
| Public suite | Compatible function pairs with sampled split direction/position, open shared boundaries and one authoritative primary camera region |

In `gymnasium`, full-size volleyball courts measure 18 × 9 m, with 50 mm lines, a 3 m free zone and a centre net height of 2.24 or 2.43 m. Smaller practice courts vary both dimensions and net height. The manifest records the playing area, clearances, orientation and court centre. Floor paint shares the floor semantic. Padded posts, tape and individually modelled cords share an object box and the `sports_equipment` semantic. Depth and co-visibility can see through the net openings. [FIVB Official Volleyball Rules, sections 1–2](https://www.fivb.com/wp-content/uploads/2025/01/FIVB-Volleyball_Rules2025_2028-EN-v05.pdf)

Where the sideline is deep enough, a gymnasium can fit a freestanding bank of two to five tiers. Timber benches or individual moulded seats flank a stepped aisle. Row pitch, rise, bank length and aisle position vary within each installation. Placement leaves at least 0.70 m between the front of the bank and the court run-off and checks the door approach. Room cameras can look diagonally across the playing surface toward the seats as well as along the net. Each bank has one object box and the `bleacher` pixel label. The public-program audit records its orientation, construction and spectator capacity separately from playing courts. The dimensions and seat profiles draw on commercial telescopic gym seating. [Hussey MAXAM dimensional guide](https://www.husseyseating.com/content/MAXAM/MAXAM-Brochure.pdf)



For a hallway, the primary camera and O-voxel region is the passage. Adjacent offices and lounges are visible through glazing and doorways. The topology export identifies which two regions each door connects.

A stair hall exports the ground, upper deck and individual tread tops as metric support regions. The upper gallery can hold facing seats, a freestanding exhibit, a directory or open standing space. Its route reservations connect the ground entrance to the gallery and keep the arrival clear of furniture. It represents a single storey of rise.

A courtyard archives the bounds of its rectangular sky opening, canopy thickness and finishes. The light meter projects both edges of the opening onto the floor for sun and diffuse sky. Ceiling cameras and luminaires attach to the covered walks. A moving camera’s full mounting rail must stay on the canopy. Sky seen through the opening uses the same finite depth and sky label as the other exterior views.

Fitness rooms fit the dumbbell shelf and its approach first. In a narrow bay the shelf can turn lengthwise or use a shorter commercial size. After fitting a bench, either side of the central aisle can receive a barbell station. Strength-oriented rooms request one more often. Placement reserves 0.72 m past each sleeve and 0.91 m in front of the rack. Each station counts once in workout capacity. The public-program audit records its pose, frame, J-cup and safety heights by pin index, plate stock and paired loads. All components share the station’s instance and `fitness_equipment` pixel class. The camera planner can see through the cage. The bar is racked and static. Bar lengths, sleeve diameters, plate profiles and rack proportions draw on commercial equipment. [Training bar](https://media.eleiko.com/admin/download-product-sheet.aspx?articlecode=3002551&language=en-us&view=true) · [Half rack](https://media.eleiko.com/admin/download-product-sheet.aspx?articlecode=3060249-03&language=en-us) · [Training plates](https://eleiko.com/en-us/equipment/plates/weightlifting/3085440-20-eleiko-iwf-weightlifting-training-plate-20-kg)

Retail capacity counts installed merchandise fixtures. Their stock shares the fixture’s bounding box and rigid transform. Pixel semantics distinguish fixture structure (`retail_fixture`), food, drink containers and clothing. Checkout structure and terminals use `counter` and `control_panel`.

Checkout forms follow the merchandise. Grocery lanes have an inset scanner and a separate receiving or conveyor surface. Apparel wraps provide a continuous folding surface, cashier-side storage and an optional suspended wrapping roll. Self-checkouts have a shopper-facing terminal, a lower bagging platform and an independent card reader. One attendant can serve a single or paired bank within 6.2 m of the staffed till. The placement validator checks each customer approach and the cashier's working side. The public-program audit exports these forms, poses, dimensions and finishes. The Rust accessor is `objects::public::checkout_station`. For a mesh audit at the installed size ranges, run `cargo run --example audit_objects -- --category CheckoutCounter --sample-dimensions --output out/checkout_audit`. Checkout screens use static, shop-specific sale or payment pages. [Commercial scan and bagging station reference](https://www.ncrvoyix.com/dam/retail/docs/NCR-Voyix-Hardware-Fabric-Datasheet.pdf)

Fitting rooms occupy shop perimeter after the checkout and two merchandise fixtures have clear access. Remaining stock fits around the changing area. A paired unit shares a partition and dye lot. Each bay independently samples curtain draw, pull direction and fold count, a supported bench and a backed dressing mirror. An opaque silvered finish reflects the existing room probe. The entrance faces a clear approach connected to a public cross-aisle. The cubicle is solid for camera and person placement even when the curtain is open. The instance box covers the assembly; pixel labels distinguish the curtain, mirror, bench and retail structure. The public-program audit records each bay’s clear opening in metres.

Dimensions, row pitch/curvature, aisle width, density, group rotation, level
rise, display spacing, construction state, contents and finish identity are
seeded parameters. A room name selects functional requirements; it does not
just bias the old four office activity weights. No external mesh or image
assets are required by the new object builders.

Required anchors are admitted before optional content. A bounded search finds
usable space within the actual polygonal envelope. An unavailable required
function causes validation to fail; unrelated furniture cannot satisfy it.
Physical object bounds, supported contents, mounted equipment and reserved
circulation are checked separately. Doors and folding seats sample static
states within their pre-admitted bounds. Folded auditorium seats are excluded
from seated-person sampling.

Lecture tiers and treads are ordinary authoritative `FloorPatch` geometry;
construction, camera clearance, human placement and annotations consume them.
Exposed drops have guards in both meshes and collision queries. The current
ARDY planner excludes unsupported stair traversal; it does not claim to
synthesize stair-climbing motion. Static people and safe level-floor motion
remain supported without loading a motion model when motion is disabled.

Dining areas flank the public aisle. The seated area on each side starts beyond
its service counter, display or queue, and stops at any cross aisle. Separate
sides can have different spacing and table headings. Broad areas can fit two
runs of tables. Venue density controls how many of the available places
receive a table. Cafeterias also sample communal tables in rows. Dataset
metrics report tabletop-to-aisle clearance in metres and table-centre position
across each bay (zero at the aisle, one at the outer edge) for independent
dining groups alongside the aisle.

At café booths, the usable cushion width and the facing edge of the actual
tabletop determine how many places can be served. Settings fit the tabletop,
stay within 90 cm of their diner and share the normal clutter budget. A booth
may be empty or partly served. `SeatsAt` links a dining banquette to its table,
and `Serves` links each placed dish to its chair or banquette. Perimeter waiting
benches have no table assignment. The `dining_banquette_places` and
`dining_banquette_served_dishes` metrics count places and dishes per paired
banquette. The room's seating-station capacity counts the whole bench once.
Cutlery sets can share a metal finish and a timber or moulded grip.

Served bowls sample a house soup and individual orders. Stock, tomato, squash and
pea soups vary in pigment and surface roughness. Herbs, beans, diced vegetables
and croutons can float in loose clusters or across the bowl, and some servings
are plain. Object metadata records the recipe, fill elevation and individual
ingredients. Dataset metrics expose `dining_soup_recipe`, `dining_soup_garnish_count`
and `dining_soup_fill_height_m`.

Cafeteria hot lines use 530 × 325 mm Gastronorm openings. Each well accepts a full pan,
two halves, three thirds or a half with two quarters. Independent pans hold soup, roasted
vegetables, potato gratin, a lid or an exposed steel basin. Dry sides favor shallow pans. A gratin can be freshly baked or scooped away along one edge. Rear controls face the staff aisle. Installation
requires a supported counter, an unobstructed guest-side landing and clearance in front of
the beverage taps. The archive distinguishes food, controls and appliance geometry. Dataset
metrics include `hot_food_pan_contents`, `hot_food_pan_depth_m` and `hot_food_glass_guard`.
[Bartscher pan dimensions](https://www.bartscher.com/int/en/product/711150/gn-container-1-1gn-d150-basic-line) ·
[Two-half-pan countertop unit](https://www.bartscher.com/pl/en/product/200232/bain-marie-212-150) ·
[Vollrath guarded serving equipment](https://www.vollrathfoodservice.com/products/countertop-equipment/merchandising-display-solutions/display-cases-cabinets/heated-display-cases/40732)

Café and cafeteria queue barriers follow the edge of the service lane. Hand-set
posts can sit at a slight angle when the full base and the public aisle have
clearance. Retractable webbing runs from a reel to a latch, ropes sag between
metal end sockets, and rigid dividers use round or square pickets or a panel
gripped on both faces by four brackets. Dataset metrics record post spacing,
installed yaw and connection height. Sag applies to flexible spans only.

## Export contract

`IndoorManifest::functional_programs()` exposes required and accepted members,
capacity with an explicit unit, sampled parameters, rejected candidate counts, reserved spaces,
focus regions and service relationships. `assembly_diagnostics.mounts` records
ceiling equipment and its physical host. `room_topology()` includes both real
partition portals and open shared boundaries. IDs refer to the ordinary object
instances, preserving individual OBBs and semantics.

`layout::functional::support_regions()` returns named support surfaces with
metric elevations and exact clipped triangles. Bounds are only an acceleration
structure: the ground surface excludes tier patches, while an overhead deck
can overlap a lower floor in X/Z. Capture metadata exports these relationships
for functional scenes; ordinary scene preparation does not construct this
reporting triangulation. These are architectural surfaces; furniture rugs retain
their exact contours and elevations separately in `IndoorManifest::rug_supports`.
The live `floor_height()` query includes those rug overlays for placement.

The semantic taxonomy includes elevator, control panel, display case, tray return,
queue barrier, lectern, locker, plinth, sculpture, information stand and
extraction hood. Existing semantic IDs and colors stay stable. Embedded
contents retain their own semantic classes and the assembly's object instance.
The elevator car's context pieces carry explicit voxel-exclusion tags that
survive material-finish tagging. Depth and semantic capture still see them,
preventing open doors from exposing a void.

RGB, depth, normal, position, semantic, co-visibility, flow, OBB, pose and
schematic data use the production meshes and existing glass policy. Floor and
camera coordinates remain metric. Voxel exports retain the existing static,
single-timestep restriction. The new programs do not change GPU-to-Burn
transport or introduce per-room model initialization.

## Reproducible qualification

- `audit_public_programs` runs every requested seed, optionally building all
  actual meshes and validating camera/people placement. Its report retains
  failures and capture-engine provenance.
- `audit_public_cameras` sweeps 1–12 cameras and baseline 0, 0.5 and 1 over a
  declared population. It exports overlap-oracle samples, geometric baselines,
  placement diagnostics and metric schematics.
- `indoor_validate` captures production RGB and aligned annotations, writes a
  schematic from recorded capture calibration, and verifies rendered geometry.
- `bevy_zeroverse_publication program-study --input ... --output out/...`
  verifies complete, current-source native cohorts and builds a local gallery
  containing every room and annotation. It cannot write a matched study into
  the published project-page directories.

The report distinguishes requested capacity, admitted content, geometric
validity, rendered visibility and elapsed capture time. Seed diversity and
embedding spread are diagnostics; neither establishes photographic realism
or downstream reconstruction gains. Physical reference comparison, dedicated
hardware-browser qualification and fixed-budget downstream training remain
separate acceptance gates.

## Qualification scope and limits

The final native gallery contains every requested room: 64 per profile, including
PublicSuite, for 640 rooms and 1,920 calibrated 512×512 views. All aligned
RGB/depth/normal/position/semantic/co-visibility captures passed. Views contain
4–30 semantic classes; maximum geometry reprojection error is 0.00180 pixels.
The Rust report includes all rooms, 108 camera-count schematics, 90 fixed-camera
object examples, source-bound receipts and a generated paper figure/table.

The local campaign uses 1,024 consecutive seeds per profile for the nine named
programs and PublicSuite (10,240 requests). These are stratified, seed-matched
profile populations, not 10,240 IID draws from a proposed default mixture.
Every request passes layout constraints and actual constructed-mesh validation.
A separate 1,280-request boundary sweep uses density 0/1, three cameras and full
human density. Camera qualification covers 16 seeds per named profile, counts
1–12 and baselines 0, 0.5 and 1: 5,184 configurations. All requests, including
any failures, remain in the audit receipts.

Manifest generation P50 ranges from 0.36–1.18 ms across these profiles; P95 is
0.68–2.90 ms. These measurements exclude cameras, people, material synthesis,
rendering and export. Actual geometry construction **plus validation** has
P50 16.6–31.0 ms; it is not a production preparation timing. Native capture
latency and GPU tensor delivery are measured separately on shared hardware.

Capacity counts have units: chair stations (including folded stations), display
stations or elevator cars. Folded stations do not admit seated people. Kitchen
service-run receipts use their existing separate schema; an empty public-program
capacity array is not evidence that the kitchen has no seating.

The fixed-envelope object audit samples 64 constructions for each of the 15
new categories. Six orthographic 24×24 depth maps measure shape variation
without rewarding scale or material changes. Elevators, lockers, queue barriers
and sculptures vary substantially at this resolution. Several simpler fixtures
have only one separated representative; their dimensions and finishes vary,
but broad fixed-envelope shape diversity has not been demonstrated. Thin and
internal details can be under-resolved by this diagnostic.

Two matched linear-radiance Cycles comparisons at 512 samples, with independent
reference repeats, expose a lighting limitation. Mean native/reference luminance
is 1.006 for cafe seed 14000 and 2.976 for gallery seed 26000. Native relative RGB
MAE is 0.341 and 2.130 respectively. The gallery ceiling is particularly bright.
Disabling GI reduces its mean ratio to 1.660 but worsens much of the local
illumination match; GI remains enabled. These are targeted diagnostics, not a
population-wide physical accuracy score or evidence of photographic equivalence.

The WebGPU viewer compiles with optional motion enabled; an M4/Chrome runtime
and responsiveness qualification has not been performed on this host. GPU tensor
parity covers two frames, shifted principal points, both glass annotation policies,
and retained allocations. Motion-planning checks cover safe level-floor routes,
not ARDY stair climbing or a new model-quality evaluation.

Mixed suites currently compose compatible pairs across a sampled open boundary.
Arbitrarily oriented structural partitions, selectable unions of reconstruction
regions, multi-storey motion, richer wet-service routing and default-mixture
promotion remain separate work. Existing irregular exterior envelopes are still
supported. Do not interpret the room catalog, exact structural fingerprint count
or an embedding spread as a claim of ten-million-sample learning utility.

The headless CLI also emitted two elevator-landing datasets with GPU O-voxels
at resolution 64 and schematic sidecars. Their 67,921 and 67,137 voxel coordinates
are sorted, unique and in bounds; voxel AABBs equal the annotated reconstruction
AABBs, and the semantic palettes contain no exterior sky. A 32-room lecture-hall
JIT probe delivered three 512×512 views plus depth/normal/position/semantic tensors
at 0.75 rooms/s with completed GPU consumers, full shadows and diffuse GI. This
shared-device run overlapped rendering and training; it does not establish an
isolated throughput regression bound.
