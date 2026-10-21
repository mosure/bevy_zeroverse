# Room services and floor coverings

Room services use ordinary procedural object identities and the existing PBR,
semantic, instance-box, depth, normal and visibility paths. They do not introduce
new texture atlases or per-bulb shadow maps.

| Feature | Sampling and constraints |
| --- | --- |
| Exit signs | Green or red illuminated EXIT letters, variable enclosure dimensions and finish. Mounted above the actual primary entry only when solid backing and headroom permit. These are visual fixtures, not a building-code compliance simulation. |
| Fire safety | Distinct wall horns, strobes, combined horn/strobes and round metal bells; ceiling smoke/heat detectors; reachable pull stations and manual call points beside actual entrances. Enclosure proportions, vent patterns, test buttons, protective guards and red/white/ivory/dark/metallic finishes vary. |
| Speakers | Existing tabletop speakers plus one to three matched wall units; continuous enclosure dimensions include soundbars and tall multi-driver columns. Real backing and installation clearance are required. |
| String lights | Wall-mounted and room-spanning ceiling-suspended runs with real support wires and canopies; one to three asymmetric sagging bays, transverse bends, jittered spacing/drop/size, and globe, pear, capsule or miniature bulbs. Coherent white, alternating, festive and pastel palettes use at most six emissive materials per run, shared by raster rendering and static transport. No individual shadow lights or texture uploads are added. |
| Exposed ducts | Circular spiral-seam ducting, square and rectangular sections, reducers, offsets, registers, hangers and folded seams. Service lane ordering and offsets vary independently of cross-section selection. |
| Angled ceiling fixtures | Adjustable downlights/tubes and modestly inclined suspended fixtures. Visible apertures, shadowed spotlights, CPU/GPU transport, reflection geometry and reservation bounds share one orientation. |
| Area rugs | Existing small mats plus larger furniture-bearing rugs up to 7.5 × 6.5 m before fit rejection. Exact rug contours are archived as support footprints. Fully contained furniture groups and supported props are raised by the actual 4–9 mm pile height. Partial support, floor-level breaks, entry paths and existing floor details reject the proposal. People are sampled afterward against the same support surface. |
| Wallpaper | Seeded stripes, diamond lattices, oval, ogee and leaf-like motifs blend with paper/fiber/weave substrates. Metric relief and roughness remain independent of printed pigment. Existing feature-wall material slots and CPU/GPU synthesis kernels are reused. |

Semantic classes include `exit_sign`, `fire_alarm`, `fire_detector`,
`fire_pull_station` and `string_lights`; speakers retain `speaker` and rugs retain
`floormat`. Taxonomy version 10 appends colors
without changing existing class colors. Construction seeds, dimensions and
transforms remain in each object record. `rug_supports` stores the rendered
floor-covering contours so support queries do not regenerate geometry.

Ceiling-suspended strands require backing at every canopy and reserve cable,
hanger and bulb volumes against architecture, ceiling services, fixtures,
furniture and people. Their empty interior volume remains navigable. Ceiling
supports currently require a locally level mounting patch; unsuitable sloped
patches reject the proposal. Manual controls avoid door/glazing/frame extents.

Installation retries, bulb counts (at most 36 per run), rug count and service module counts are
bounded. Rejected installations are omitted rather than moved through glazing,
architecture or existing objects. A low-headroom room can legitimately have no
exit sign above its entry. These fixtures do not model powered audio, flashing
alarms or an emergency evacuation system.

Crowded rooms can reject whole-group rugs. A bounded alternative proposes a small
lounge rug with a supported coffee table or ottoman on adjacent clear floor; it
must preserve support and clearance of the existing furniture too. Rug sampling
does not guarantee a floor covering in every room.

Rolling/swiveling furniture uses the rug contact plane and rejects sweeps beyond
its support edge. ARDY navigation currently assumes a single architectural ground
plane: raised rugs are navigation barriers, and people placed on them retain
static poses. This avoids silently sinking motion clips through their pile.
