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
| Activity recovery | Sparse-room repairs introduced fixed coffee/side-table, storage and bin dimensions, centered-first proposals and fixed gaps. A library could meet its generic count without shelving. | Repair products sample ergonomic metric dimensions before admission. Seeded lateral order, radial phase, offsets and gaps retain support/clearance checks and bounded attempts. Missing library storage is repaired independently of the generic count. |
| Projector admission | An exact-center first proposal won before random proposals were evaluated. | All proposals use continuous support-local positions; one product size is retained through a support's attempts. Optical-axis, support and beam-clearance checks remain active. |
| Screen convergence | Monitor/laptop normals pointed exactly at the associated chair, including after chair tucking. | A seed-stable ergonomic yaw offset survives seating adjustments and physical admission. Monitor offsets are bounded by 0.19 radians, laptop offsets by 0.18 radians. |
| Public anchors and recovery | Movable teaching/presentation furniture always faced +Z; recovery started at the same corner of an exact lattice. | Category-appropriate orientation and offsets, shuffled stratified recovery proposals, and rotationally sampled local alternatives. Service counters and elevator approaches retain their operational orientation. |
| Gallery/showroom fields | Station centers repeated two narrow rails, with a small amount of jitter. | Continuous placement across the actual aisle-bounded available fields, with depth stratification to limit clumping and unchanged circulation reservations. This broadens fields, not aisle topology. |
| Floor lamps | Fixed dimensions and room-axis yaw survived richer lamp construction. | Metric footprint/height distributions in both initial furnishing and activity repair, with independent yaw. Emitter placement still derives from the same physical lamp program. |
| Service construction | Counter and hood geometry varied little at a fixed admitted envelope. | Open preparation frames, storage carcasses, slatted and tapered counter bodies; box, tapered, elliptical and glass hood canopies with variable flues and filter/light layouts. Counter tops preserve their full rectangular support contract. |
| Desk storage | A pen holder used one metal cylinder and exactly seven pens; organizers used equal compartments and repeated contents. | Hollow tapered/faceted cups with independently sampled substrates and collision-separated contents; unequal organizer compartments and tray spacing, optional supported paper stacks and varied occupancy. Work-surface task sampling includes pen holders. |
| Kitchen supplies | Bread, folded towels, soap bottles and cutting boards occupied only a few fixed-envelope shapes. | Bounded loaf/cut profiles, fitted cloth folds, bottle shoulders/pump construction, curved cutting-board outlines and attached handles. Built-in sink dispensers use the same construction program. Food pigments vary within substrate-appropriate color ranges. |
| Cabinet contents | Boxes and vases reused miniature fixed constructors centered in each bay. | Shared product builders, independent dimensions/finishes, and bounded support-local position/yaw proposals. Every transformed vertex remains within its physical bay. |
| General cabinet construction | Equal compartment widths, equal shelf/drawer intervals and repeated handle coordinates survive metric-scale variation. | Unequal seeded bays and shelf intervals, sampled mixed-front membership, independently coordinated frame/shelf/front substrates, variable hardware and slats. Compartment boundaries drive contents, fronts and opening reserves together; the full top supports its declared footprint. |
| Wall clocks | Square metric envelopes, a fixed depth, seed-modulo casing materials and fixed dial proportions recur. | Square, wide and tall metric sampling before mounting; tapered rounded cases, independent casing/dial finishes, varied indices, hands and fitted digital typography. Rear attachment and relief geometry share the sampled case depth. |
| Whiteboard accessories | Equally spaced marker rows, alternating cap colors and one tray material impose repeated local geometry. | Supported irregular marker/eraser composition, variable occupancy and dimensions, independent cap/tray/frame finishes, fitted magnets and a connected tray with sampled lips/end caps. |
| Wall shelves | A few bracket counts and fixed proportions repeat even when the mounting position changes. | Dimension-conditioned support counts, sampled span/offset/reach, straps, rods, curved braces and hidden supports, independent board/hardware substrates, and a full supported top. Braces connect to the physical wall plate and load strap. |
| Projection screens | Repeated casing/borders and a nominal optical target disconnect placement from the actual rolled sheet. | Sampled sheet width/offset/drop, frame/case/caps and mounting proportions. Projector targeting and throw ratio use the same white-face rectangle that is rendered. |
| Portable power | Nominal bounds concealed a nearly constant body and socket deck. | Sampled casing height, bevel, socket span/phase, inserts and switching. Both cable routing and geometry derive ports from the same program. |
| Finish coupling | Every assembly chair shared a finish seed; unrelated substrates shared the same atlas index. | Collection-level coherence allows matching and replacement products. Independent substrate streams preserve related wood/cloth faces without locking wood grain to upholstery structure. Atlas capacities remain bounded. |
| Content diagnostics | Screen/art/whiteboard statistics used the small finish palette rather than the actual content slot. | Metrics follow the material slot attached to the built mesh. Back-facing phone configurations without a screen mesh do not emit screen-content statistics. |
| Benches | A small number of fixed supports, slat pitch and seat constructions recur; a tall input envelope can become an implausibly tall backless seat. | Independent ergonomic seat height, splayed/sled/pedestal/panel supports, continuous rail spans and slat spacing, divided upholstery, raked backs and optional arms. Placement and metrics read the same bench program. |
| Drawer units | Drawer sizes follow a monotonic ramp, hardware always uses one bar, and feet have fixed offsets. | Unequal sampled drawer-height weights, caster/leg/plinth bases, varied fronts and bar/knob/finger pulls. The top remains a full level support. Hardware and drawer interiors retain their common motion ownership. |
| Projectors | Lens variation hides a nearly fixed housing, button row and power-port position. | Continuously varied rounded/tapered enclosures, feet, controls and vents. Layout beam clearance, mesh optics, cable ports and metrics derive from one physical construction program. |
| Knife blocks | Uniform handles occupy a single fixed row with equal blade exposure. | Fitted two-dimensional socket layouts with optional empty sockets, variable handles and exposure, independent block profiles and plausible substrates. Bounded proposals enforce clearance between handles. |
| Bound literature | Volumes share a cover material and a predictable descending size/yaw sequence. | Independent volume dimensions, small supported offsets and yaw, cover finishes, clipped cover corners, binding pitch and elastic widths/positions. Open notebooks retain page-surface-aligned writing. |
| Upright shelf books | Every partial book run starts at the left divider. | The complete collision-free run samples its legal horizontal interval; full shelves remain naturally constrained. Shelf containment and grounded contents are checked on actual transformed meshes. |
| Product coatings | Cabinets, service equipment, presentation furniture and book covers can inherit the architectural wall-paint recipe and its wall treatment. | Manufactured products use the existing polymer/laminate, metal, plating, timber or binding materials and their independent finish slots. Elevator enclosures retain architectural paint where appropriate; operated door skins use manufactured finishes. No new shader or palette capacity is required. |
| Rolling carts | Equal shelf spacing, fixed post/caster spans and a centered paper block recur across different products. | Height-conditioned shelf counts, unequal clearances, independent frame/tray substrates, varied grounded casters and push rails, and bounded lower-bay placement of actual boxes/books. Empty and occupied bays coexist. The full level upper support and rigid cargo motion remain intact. |
| Room dividers | One foot/post arrangement and one fabric slab dominate the category. | Unequal panel widths, three base constructions, variable mounting height, frame/infill materials and optional upper slats with sampled pitch, width and angle. Actual infill height and base span are exported to metrics. |
| Kettles | A fixed handle/spout assembly overwhelms the small amount of vessel-profile variation. | Continuous body, shoulder, neck, lid and spout proportions; side or overhead handles attached to the actual body profile; open pouring bores; independently sampled metal, chrome, polymer and ceramic shells and grips. |
| Desk fans | One vertical stand, fixed guard counts and a few blade counts impose a repeated silhouette. | Seeded head pitch/yaw fitted before construction, independent base/pedestal or yoke, guarded blade geometry, radius, depth, blade sweep/count/phase and guard spacing. The housing and supports share the tilted head frame. |
| Printers | Nearly constant scanner/body blocks, centered output paper and two screen positions recur. | Independent occupied body proportions, real output cavities, tray widths and offsets, optional scanners/feeders, control layouts and paper occupancy/yaw. Tray and paper contact planes are separated from the body to avoid coplanar overlap. |
| Desktop speakers | Driver centers and size ratios repeated; all sampled loose speakers were upright boxes. | Independent cabinet proportions and substrates, horizontal/vertical metric envelopes, unequal driver cells, varied cone radius/depth/dome, true baffle apertures and optional textile grilles. Exposed drivers are ray-checked against the real cabinet mesh. |
| Headphones | One upright arch, fixed cup orientation and generic rounded solid ear pads. | Adjustable band/cup proportions, independent upright swivels, hollow cushion openings and a continuous padded ribbon. Flat storage swivels both backs onto the support; yaw can vary without balancing the object on one point. Sampled envelopes reserve the resting pose; uniform fitting stays inside them. |
| Table microphones | A seed-modulo branch and fixed width/depth/controls made the two forms readily predictable. | Metric boundary/gooseneck/capsule forms, fitted head axes, varying base profiles, heights, bends and controls. Table admission samples full metric dimensions instead of a fixed footprint. |
| Staplers | The same stacked blocks and hinge dominate the category. | Independent base, axle, lever profile, nose taper and opening constrained by actual height clearance. Metal magazine and front anvil stay attached to the lever/base, with independent product finishes. |
| Tape dispensers | One radius/base/sidepost arrangement and no connection between roll and cutter. | Variable roll/core/width, weighted base, open/closed cradle sides, cutter teeth and occupancy of a tangent tape web. The core clears the cheeks; web, roll and blade have a shared geometric construction. |
| Storage boxes | Solid capped bodies, correlated lid/hardware branches and narrow loose-item dimensions recur. | Actual open cavities or seated lids, continuous wall/floor/taper/roundness, attached curved handles, independent shell/lid/hardware finishes, and broader dimensions sampled before support admission. The shared constructor also serves cabinet bays and cart cargo. |
| Coat racks | Identical angular spacing and parity-based tip heights couple hooks to the feet; every object has a narrow upright envelope. | Independent hook angles/heights/reach, independent stable base counts and phase, and garment rails with separated, irregularly occupied hangers. Wide and compact metric envelopes are sampled before room clearance. |
| Waste bins | Wall thickness, rim height, lid scale, rib count and pedal dimensions use repeated constants. Recovery often reintroduces round equal-axis envelopes. | Dimensioned cavity profiles, variable rims/seams/lids/pedals/dividers/ribs and independent hardware finishes. Main and recovery placement use the same metric sampler, including unequal footprint axes. |
| Loose floor belongings | Side-storage orientation is almost always the wall heading plus a small offset. | A mixture retains aligned service arrangements while allowing free yaw for bins, storage boxes and coat racks. Every proposal still passes the ordinary bounds and collision checks. |
| Constrained recovery | Searching many positions for one sampled bin/side-table size can leave usable narrow floor space unfilled. | At most three independently sampled products, with a compact final proposal, retain immutable dimensions during each bounded position search. Independent per-anchor/category streams keep one failed product from controlling another's proposals. Population, support and clearance requirements are unchanged. |
| Kitchen dishware | The same few radial profiles recur in loose dishes, shelves, sinks and appliances. | A shared hollow program varies occupied diameter/height, oval aspect, foot/shoulder/mouth, wall thickness, faceting, handle reach and substrates. Plate stacks have a constant-thickness nesting profile; handle thickness is bounded by bend curvature. |
| Pantry and refrigerator contents | Three pantry shapes and two refrigerator shapes repeated; admission used a different approximation from rendering. | Six dimensioned constructions (carton, jar, can, bottle, pouch and lidded tub), independent geometry/finish streams, varied shoulders/necks/feet, fill levels, caps and label bands. Shelf admission and rendering consume the identical sampled program, including labels and handles. Wrapped label UVs remain local to each package. |
| Drying racks | Exactly seven rails and plates concentrated at one end. | Dimension-conditioned pitch and rail count, shuffled partial occupancy, individual plate dimensions/yaw/lean, independent tray/hardware/dish finishes. Actual plate bounds fit each wire slot and touch the tray. Empty appliance contents remain empty. |
| Utensil holders | A parity-alternating ring with monotonically increasing tool heights. | Hollow tapered/faceted holders and independently sampled spoon, spatula, slotted-spatula and fork heads. Bounded proposals vary tool heights, lean, twist, position, dimensions and materials while checking shaft/rim clearance and conservative separation. Metadata records accepted tools. |
| Finish-key convergence | Applying a parent's automatic finish could overwrite a child mesh already using the same material/semantic key. | Convergent keys merge geometry without losing vertices or triangles. Vacant keys move the existing geometry without an extra allocation/copy. |
| Lounge conversation groups | One centered sofa, a nearly centered table and two mirrored seat locations survived the spatial planner. | Sofa offsets, front clearances, table proportions, seating occupancy, angular spread and individual setbacks are sampled in furniture coordinates. An occasional symmetric arrangement and coordinated finishes remain plausible. Only accepted sofas/tables anchor seats, and all placements retain ordinary support/collision admission. |
| Supplemental perimeter furnishing | Fixed-size bookcases, fixed-depth cabinets and equal-width corner plants were still appended to modern spatial programs. | Independent metric dimensions, occupancy and ordering; bounded proposals on actual envelope edges, with category-appropriate yaw and sampled setbacks. Archived layouts without a spatial program retain their existing behavior. |
| Ottomans and poufs | Four wooden feet, two cushions and a fixed vertical crown dominated the category. | Continuous footprint/aspect, height, cushion crown/roundness, cap fraction and base dimensions; feet, sled, plinth and low upholstered constructions. Independent fabric/leather panels and wood/metal/chrome/polymer base finishes use existing bounded PBR slots. Finish interventions preserve geometry, normals and metric UV scale. |


The lounge/perimeter programs and their qualification are documented in
[Lounge and perimeter sampling](lounge_perimeter_sampling.md).

The implementation does not resample materials or geometry independently per view.
The capture identity records the furnishing, tabletop and public program contracts.

The category-specific construction, scale, service-core and supported-content
qualification is documented in [Public furnishings and small-object regularity](public_object_regularity.md).
It measures actual fixed-envelope meshes and accepted room dimensions separately.

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
- Monitor/laptop facing error relative to the actual associated chair, with explicit linked-target and within-2-degree counts. Unlinked screens do not enter this denominator.
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

The local object audit has three complementary scopes:

- `examples/audit_objects.rs`: all catalog categories at fixed metric envelopes;
  real triangles, finite normals/UVs, winding, bounds and semantic vocabulary.
- `objects::audit::regularity_qualification`: a bounded catalog sweep, seeded
  replay and fixed-envelope shape coverage for changed products, cabinet-bay
  containment/placement statistics, and full geometry/layout checks over every
  concrete room program with three cameras and nonzero human frequency.
- Category tests exercise supported-size extremes, pen/holder separation, crust
  attachment, power-port/deck agreement, actual content-slot metadata, and
  independent substrate sampling. Existing botanical, cable, support and semantic
  regression tests remain applicable.
- Portable-product tests exercise independent dimension extremes, real lower-cart
  bay containment/separation, current rest-mesh equality with rigid motion,
  kettle handle attachment and an independent ray oracle through the spout bore.
  Constructor programs and reported fields reference the same geometry inputs.
- Kitchen supply tests qualify hollow interiors, nested contact, yawed footprints,
  rack slots/tray contact, utensil cavity clearance, refrigerated shelf packing,
  local label UVs and vertex/triangle conservation during finish assignment.
- Audio/stationery qualification uses independent metric extremes, exposed-driver
  ray checks, tape tangency and roll clearance, stable headphone contact, separate
  geometry/finish streams, and actual accepted table-placement distributions.
  Physical admission and rendering use the same reserved product dimensions.
- Container/rack tests exercise independent metric extremes, triangle rays into
  actual cavities, lid extents, handle attachment, rack ground contacts, hanger
  separation, separate finish streams and rigid-motion ownership.
- [Wall presentation equipment qualification](presentation_equipment.md) covers
  accessory contact/separation, independent metric extremes, actual shelf-top
  support, sheet ray hits, material independence and 80 distinct complete-room
  seeds. Accepted transforms and construction programs accompany the local
  geometry gallery and timing report.

Set `ZEROVERSE_OBJECT_REPORT=out/object_regularity_current` while running the
regularity qualification to save `catalog.json`, `geometry.json`, `rooms.json`,
and a fixed-envelope `objects.png` contact sheet. Cabinet contents are checked
against actual sampled compartments by `objects::storage::tests`; the local
`storage_clock_regularity_population` study records current cabinet/clock
parameters and accepted placements in `out/storage_clock_regularity`. The
source identity and generation configuration are recorded in the receipts. The
contact sheet is a flat-shaded CPU rasterization of actual triangles, not PBR.
Study results and galleries stay local; they are not automatically included in
project-page or paper publication.

Shape coverage uses six orthographic 24 × 24 depth maps and ordered greedy
representatives at normalized depth distances 0.005, 0.015 and 0.03. It does not
count color, UV offsets or triangle duplication as geometry diversity. Low depth
variation is expected for some flat panels and standardized products; content
and material measurements must be evaluated separately.

For example, a notice board's frame or a regulated exit sign can have similar
fixed-envelope depth maps while its physical dimensions, contents and placement
vary in actual rooms. A low geometry score alone is not a reason to deform a
standardized product. Review those categories with their separate content and
accepted-placement distributions.

CPU construction measurements exclude rendering, material baking, GPU transfers
and concurrent training. They are bounded preparation-cost diagnostics, not an
end-to-end throughput claim. Qualify GPU/PBR appearance and browser performance
separately when the device is available.

These measurements do not establish a count of independent training examples or
prove that a model cannot learn the generator's rules. Evaluate compositional
holdouts and correct-view versus repeated/mismatched-view ablations downstream.
