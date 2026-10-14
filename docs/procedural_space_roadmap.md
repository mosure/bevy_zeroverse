# Procedural indoor realism and diversity roadmap

The next substantial gain should come from **coherent construction, functional
arrangements, and better light transport**. Increasing the number of seeds or
independently randomized parameters alone will not make the rooms more realistic
or establish useful diversity for ten million training samples.

This is a review and implementation plan for the current generator, dated
2026-10-13. The reviewed checkout is `91c5c96361970cc38867e2e78f27813fb92d833c`,
bevy_zeroverse 0.39.0, generator 37, source
`760448699a3a7edb6ef25b2cd8623614dcd2bf53d3871ed9225a28e9c8ec8130`.
This document preserves the planning baseline; the implemented scope, measured
results and remaining qualification limits are recorded in
[the execution report](procedural_space_implementation.md). Existing Cornell,
object Zeroverse, and other room modes remain part of the compatibility contract.

## Current capabilities and evidence

The source contains 63 object kinds, 36 base material roles, 14 botanical growth
habits, ten chair selectors, and 19 hair selectors including baldness. Many
selectors contain continuous construction programs. These are inventory counts,
not estimates of perceptually distinct scenes.

| Area | Current implementation | Most useful next improvement |
| --- | --- | --- |
| Architecture | Concave and oblique envelopes, chamfers, multi-wall glazing, arches, pillars, sloping ceilings, floor levels and mezzanines | More spatial topologies and a common architectural construction model |
| Windows | Variable opening sizes, mullions, transoms, reveals and shades; shared interior/exterior glass recipes | Per-opening construction and optics, partial frosting, operable geometry and consistent reflections |
| Furnishing | Ten activity profiles bias four main composition patterns; continuous workstation fields, support-aware clutter and chair insertion | Functional groups with explicit relationships, usable space and accepted-state coverage |
| Objects | Multipart furniture, curved tops, bent supports, cushions, screens, services, routed cables and small props | Manufacturing logic, articulation, coherent assemblies and placement states |
| Materials | Metric UVs, correlated albedo/normal/roughness maps, layered finishes, anisotropy, clearcoat and filtered mipmaps | Object-scale material identity, seam continuity and context-dependent surface variation |
| Lighting | Broad continuous photometry, physical units, sun, fixture circuits, approximate diffuse GI and scene-derived reflections | Occlusion correctness, spatial reflections and finite emitter response |
| Cameras | Primary-room paths, count-aware baseline, surface-target variation, overlap policies and metric handheld motion | Better accepted trajectory coverage and calibration/overlap qualification across conditions |
| People | Anny bodies, morphology, fitted garments, procedural grooms and optional batched ARDY motion | Garment silhouette, hair scattering, facial construction and visible interaction quality |
| Execution | Bounded native preparation, cooperative browser work, hybrid GPU material synthesis and same-device Burn capture | Preserve these contracts while extending quality; measure completed generation and training |

The current publication protocol provides a 512-room structural distribution
audit, 32 rendered rooms, four cameras and two times: **256 calibrated views**.
The separately emitted actual-mesh audit covers **16 rooms**, not all 512.
The current camera study covers **2,304 requested sets** across 64 seeds,
1–12 cameras and baselines 0, 0.5 and 1; all sets passed placement qualification.
Its native count sweep covers two room families, so it is a spot check rather
than a population estimate. See [publication.toml](../publication.toml) and
[camera sampling qualification](camera_sampling_review.md).

Reanalysis of the existing current-generation capture files gives:

| Observation | Result and denominator |
| --- | --- |
| Structural invalid rooms | 0 / 512 under the recorded protocol |
| Concave / oblique envelopes | 221 / 512 and 458 / 512 |
| Sloping ceilings / pillars | 354 / 512 and 425 / 512 |
| Archways / mezzanines | 194 / 512 and 46 / 512 |
| Raised / depressed floor regions | 90 / 512 and 103 / 512; features can co-occur |
| Main-envelope plant presence | 476 / 512 rooms; mean 4.44 plants |
| Main-envelope phone / cable presence | 445 / 512 and 283 / 512 rooms |
| Conference-profile rooms containing a `Table` | 15 / 44; mean 7.07 desks per conference-profile scene |
| Accepted monitor-plus-laptop work areas | 11 / 4,817 desk/coffee-table observations; includes neighboring rooms and non-office surfaces |
| Accepted multiple-monitor work areas | 55 / 4,817 under the same denominator |
| Rejected object placements per room | Median 238; P95 645; includes intended rejection during constrained sampling |
| Ceiling fixtures per room | Median 13; P95 36; maximum 57 |
| Camera path length | P05 0.060 m, median 0.331 m, P95 2.978 m across 2,048 cameras |
| Vertical field of view | P05 33.0°, median 64.1°, P95 102.4° across 2,048 cameras |
| Camera height | P05 0.98 m, median 1.76 m, P95 2.68 m across 2,048 cameras |
| Semantic classes per captured view | Minimum 7, median 17, maximum 28 across 256 views |
| Visibility shared with any peer | Median 84.3%, minimum 43.5% of valid pixels across 256 views |
| Rooms with people in the primary viewing zone | 19; all show the person class somewhere in their eight views |
| Reprojection P99 | Worst view 0.000612 pixels across 256 views |

The low docked-workstation count warrants conditional analysis; its denominator
does not support saying that almost all office desks lack screens. Likewise,
people elsewhere in the envelope need not appear in primary-room captures.
Class-level visibility does not establish that every individual is visible.

I visually inspected one initial view from each of the 32 current rooms and
full-size examples from rooms 5, 9 and 22. The images show substantial geometric
variation, but repeated partition arrangements, sparse functional groupings,
large simple surfaces, conspicuous procedural surface patterns, simplified
exterior context and stylized people remain apparent. These are qualitative
observations from those views, not estimated population failure rates. Static
images cannot establish temporal stability or absence of flicker.

The local review assets are under `out/procedural_space_review/`: two contact
sheets and `current_capture_summary.json`, including hashes of the input capture
and manifest files. They reuse the current captures; this review launches no
render benchmark or training run. Historical qualification folders are not used
as current-generator measurements.

## Priorities and sequence

| Stage | Deliverable | Reason for the order |
| --- | --- | --- |
| 1 | Functional-placement diagnostics and lighting correctness fixtures | Establish whether sampled intentions survive construction and whether light reaches only valid surfaces |
| 2 | A complete conference/workstation assembly slice | Improve room organization, furniture relationships, clutter and cables together |
| 3 | Spatial reflection, patterned glazing and material-scale qualification | Improve surface response, partial frosting and glass appearance with explicit optical contracts |
| 4 | Polygonal architecture, room adjacency and window assemblies | Expand spatial structure and window geometry while sharing construction and collision definitions |
| 5 | Botanical, human and object-state refinement | Add richer silhouettes and interactions on the improved placement foundations |
| 6 | Camera trajectory and optional optics extensions | Build on the existing baseline rather than replacing its working guarantees |
| 7 | Larger distribution, rendering and downstream qualification | Establish useful coverage and performance before changing training defaults |

Stages 1–3 are the highest-return first tranche. Architecture design can proceed
alongside them, but it should consume the same support, opening and placement
contracts. Every stage must leave a usable generator; none requires a wholesale
rewrite before benefits can be measured.

## Functional placement and room composition

The ten activity profiles currently choose among workstations, meeting groups,
soft seating and learning groups through `ActivityMix`. `furnish_program` then
uses a few major branches and fills remaining space with amenities and recovery
placements. Spatial fields already add curvature, stagger, fan, jitter and
occupancy gradients. Extend this foundation into **relationships among objects**.

1. Sample room use and required activities before choosing furniture. Represent
   relations such as faces, supports, serves, seats-at, powers, reaches and
   shares-circulation. A meeting requires a usable shared surface and seats; a
   workstation requires reach and sight lines; a presentation station requires
   an audience orientation and an unobstructed display or projection region.
2. Construct a continuous furniture group around those relations. Meeting
   capacity, table aspect, segments, seat spacing, incomplete occupancy,
   attached returns and clearance vary jointly. Workstations span wall-facing,
   opposed, clustered, curved, peninsula and shared arrangements without
   encoding each finished layout as one fixed template.
3. Place and locally adjust the complete group transactionally. Try rotation,
   translation, table proportions or another compatible support construction
   before abandoning its function. Keep bounded retries and deterministic tie
   breaks. Do not achieve acceptance by deleting difficult objects or weakening
   doorway, circulation or mesh-clearance rules.
4. Distinguish necessary circulation, operational clearance and allowed contact.
   A seat may enter the clear volume beneath a table; a hand may touch its top;
   a book may rest on another book. A drawer needs an opening sweep. These are
   different constraints from a universal inflated bounding box.
5. Sample state at the group and person level. Recently used workstations can
   have pulled-out chairs, a turned display, a mug within reach and a connected
   charger. Tidy/shared/personal work areas should vary in meaningful ways;
   independently rotating every object produces disorder without use.

Extend `IndoorObject.support` and `interaction_target` rather than creating a
second scene hierarchy. Add operational volumes, sockets and contact regions
to construction recipes. Separate proposal counts from accepted counts and
record rejection reasons by group, object kind, room shape and activity.

The first slice should guarantee the **declared function** of an accepted
conference assembly. It should not require every mixed conference-profile room
to contain an arbitrarily named class. Export the actual assembly role so a
shared arrangement of desks can be distinguished from a failed meeting-table
proposal. Measure intended versus accepted capacity, reachable seats, work
surface coverage, empty-space connectivity and wall-alignment distributions.

Code homes: `program/activity.rs`, `program/furnishing.rs`, `layout/programs.rs`,
`layout/programs/recovery.rs`, `layout/surfaces.rs`, `layout/seating.rs`,
`footprint.rs` and `metrics/content.rs`, under `src/scene/procedural_indoor/`.

## Architectural diversity

The existing envelope is already nonrectangular. Its topology nevertheless
comes from a constrained construction: a tapered/chamfered footprint with an
optional notch on one edge, clipped axis-aligned recursive partitions, a single
planar ceiling slope, floor patches and at most one mezzanine. Rotating the scene
does not add new adjacency or circulation structures.

Introduce a bounded architectural program whose output is consumed consistently
by mesh construction, placement, camera navigation, GI, annotations and exports.
Start with simple polygons and a room/portal graph rather than unrestricted CAD.

- Compose convex cells into offset bays, wings, alcoves and re-entrant corners.
  Vary the number and placement of features, attachment edges, angles, lengths
  and opening sizes. Preserve minimum feature dimensions and positive wall
  thickness. Interior courtyards or polygon holes can follow only when the
  triangulation, containment and export contracts support them explicitly.
- Let partitions follow polygon edges or independently sampled directions.
  Sample useful room adjacency, corridor width, open connections and glass
  partitions before decoration. Doors need jamb depth, lintels, reveal geometry
  and a usable approach on both sides; doors with leaves need sweep clearance.
- Replace the sole ceiling plane with compatible bounded patches: flat fields,
  lean-to slopes, ridges, tray recesses, soffits, exposed beams and clerestories.
  Join them with closed surfaces. Fixture attachments must use the actual local
  ceiling, not a nominal room height.
- Connect columns, beams and ceiling modules to plausible construction rhythms.
  Preserve unusual configurations as explicit distribution tails; the current
  83% pillar and 69% sloped-ceiling frequencies should be reviewed against the
  intended training domain rather than increased automatically.
- Model steps, landings, depressions and mezzanines through connected support
  regions with headroom and navigation constraints. A full multilevel case
  needs separate vertical occupancy intervals, not just one height per X/Z.
- Give glazing a construction hierarchy: openings, frames, mullions, gaskets,
  panes, sill and shade mechanisms. Vary full-height spans, high windows,
  grouped bays, corner glazing and recessed openings subject to wall support.
  The dedicated window program below connects these shapes to operating states,
  glazing treatments, reflection behavior and annotation policy.
- Generate exterior parcels with coherent ground level, facade scale, horizon
  and distance. Preserve the finite sky proxy as an annotation fallback; its
  depth must remain distinguishable from physical buildings and room geometry.

The primary reconstruction region should remain an explicit selected space.
When zones become polygonal, an AABB alone cannot define membership. Retain the
exported AABB and add an exact region identifier/membership representation for
masking, O-voxel selection and camera admission. A feature expansion must not
silently turn the task into whole-building reconstruction.

Acceptance requires the same analytic geometry to drive triangulation,
containment, openings, support, clearance and visibility. Test nearly collinear
edges, very thin valid polygons, sloping joints, door/window intersections,
stairs, mezzanine headroom and UV seams. Keep the historical thin-polygon and
reported-seed regressions; do not solve these cases by relaxing tolerances in
only one subsystem.

Code homes: `envelope/`, `program.rs`, `architecture/facade/`,
`architecture/details.rs`, `architecture/outdoors.rs`, `validation/` and
`cameras/navigation.rs`.

## Windows and glazing diversity

Window diversity should combine opening shape, construction, pane treatment,
operating state and the environments on both sides. These factors must remain
related: a privacy band has a useful height, a sliding sash needs a track, and
a corner window needs a plausible corner joint. Changing glass color alone
does not create a different window system.

The current facade program already varies exterior sides, opening dimensions,
columns, transoms, frame material, frame depth, recess, sill projection, Venetian
blinds and roller shades. Construction uses rectangular panes; the `operable`
flag currently adds a handle without articulating the sash. Main architectural
glazing samples one exterior and one interior optical recipe per scene, with
uniform pane roughness and fixed role thickness. This leaves considerable room
for useful variation without duplicating existing opening-size controls.

### Construction and opening geometry

Extend the aperture program with pane boundaries, frame profiles and attachment
coordinates that construction, support, collision and export all consume.
Preserve continuous dimensions within compatible constructions rather than
selecting complete windows from a small finished-mesh catalog.

| Construction axis | Proposed variation | Geometric constraints |
| --- | --- | --- |
| Opening silhouette | Punched openings, horizontal ribbons, floor-to-ceiling spans, clerestories, arched heads, rounded corners and roof-following trapezoids | Positive wall/header dimensions, a valid aperture contour and closed reveals |
| Assembly footprint | Flush and recessed units, projecting bays, faceted bow windows, corner returns and continuous glazing around adjacent walls | Shared corner joints, supported projections and consistent interior/exterior surfaces |
| Pane subdivision | Unequal panes, offset transoms, grouped lights, asymmetric grids, narrow sidelights and opaque spandrel panels | Mullions actually meet; pane edges stay captured by frames; no overlapping duplicate bars |
| Frames and edge details | Extruded metal sections, painted timber, polymer frames, glazing beads, gaskets, handles, hinges, tracks and drainage/sill details | Profile thickness fits the assembly; material direction and joinery agree at corners |
| Opening state | Fixed, hinged, awning, tilt, pivot and sliding arrangements with continuously sampled opening angle or travel | Mechanical limits, sash/frame contact, opening sweep and clearance from blinds, walls and furniture |
| Overhead glazing | Sloped roof lights and clerestory strips integrated into ceiling patches | Continuous curb/flashing/reveal construction, roof support and proper light/visibility openings |

Initially support closed and statically opened states. An opened sash moves its
pane, frame, handle and seals as a coherent assembly; there must be no hidden
closed pane left in the aperture. A window is not automatically a walkable portal:
navigation depends on sill height, usable opening and the task's boundary rules.
If animated opening is added later, the same transforms must drive flow, bounds,
shadows and collision over the capture interval.

Faceted projections provide a bounded first implementation of bay/bow geometry.
Continuously curved glass and complex roof intersections should follow only
after their surface parameterization, tessellation and optical approximations
pass qualification. Every construction must retain meaningful inside/outside
normals and avoid coincident trim, sill, film and pane faces.

### Glass identity and tint

Sample glass properties through a hierarchy: building or installation family,
window assembly, pane, then sparse local variation. Adjacent panes from the same
installation should usually share thickness and finish; occasional replaced
panes or added privacy film can differ. Independent random tint on every pane
would make a coherent curtain wall look accidental.

Proposed recipe fields should distinguish dielectric IOR, physical pane
thickness, absorption color/distance, surface roughness, surface treatment and
explicit coating response. Use a calibrated optical response for any reflective
coating rather than arbitrarily raising ordinary glass metalness. Dielectric
reflection/transmission varies with incidence through Fresnel behavior; the
[PBRT dielectric reference](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF)
provides the isolated reference model for these tests.

Include continuously varied clear/low-tint glazing, green edge absorption,
neutral smoke, bronze and restrained blue/green solar-control appearances.
Treat tint strength, transmitted brightness and reflected color as distinct
properties. Verify absorption against thickness and angle using known backdrop
colors. Painted or opaque spandrel panels need their own material behavior,
even when they belong to the same window assembly.

Laminated and insulated units can add edge spacers, seals, multiple panes and
air gaps. Build those layers only when the selected renderer can represent their
transport adequately, or expose a declared effective single-pane approximation.
Adding several transmitting slabs to the current screen-space path must not
silently imply correct multilayer refraction. Keep thin surface film separate
from bulk glass absorption in the recipe even when a qualified shader combines
their effects.

### Partial frosting and patterned surfaces

Add a seedable, composable treatment field over pane-local metric coordinates.
Its primitives should include height bands, gradients, repeated stripes,
dots/frit, bounded geometric motifs, rounded patches and signed-distance shapes.
Continuous coverage, pitch, orientation, edge transition, irregularity and
scattering strength create variation within these programs.

- Sample lower privacy panels, eye-level bands, upper diffusion zones, clear
  borders, clear viewing apertures, alternating panes and whole-pane frosting.
  The complete range should include untreated panes as well as fully treated
  ones; partial treatments must have meaningful accepted probability.
- Anchor related bands at installation/world height so they align across unequal
  panes, doors and sidelights. Support deliberately staggered or pane-local
  treatments as separately sampled design choices, not accidental UV mismatches.
- Distinguish etched/sandblasted roughness, scattering privacy film, opaque
  ceramic frit and printed markings. A translucent rough region and an opaque
  printed region need different optical and annotation behavior.
- Mix the appropriate roughness, scattering, transmission and absorption fields.
  Do not implement all frosting as a flat white tint or reduced alpha; the
  blurred background, reflected lobe and retained light should be evaluated
  separately. Where the renderer cannot express distinct surface/film responses,
  record and qualify the approximation.
- Apply very thin treatments in the existing pane material or a nonoverlapping
  surface partition. Avoid coplanar transparent overlay planes. Only substantial
  relief or an actual separate layer should introduce additional geometry.
- Filter treatment edges and fine patterns consistently through the mip chain.
  Keep opaque coverage and roughness statistics explicit so distant dots do not
  shimmer, disappear unpredictably or become an incorrectly solid sheet.

Add subtle, contextual surface variation after the clean material works:
cleaning streaks, handle-adjacent fingerprints, sparse dust at seals, mild
surface waviness and condition-dependent condensation. Preserve clean glass as
a common case. Avoid arbitrary strong normal noise that makes every office
window resemble distorted decorative glass.

### Reflections and the surrounding scene

Qualify windows under bright-exterior/dim-interior, overcast, balanced light,
and lit-interior/night-exterior conditions. Reflection appearance should respond
to these environments and viewing angle. A privacy-mirror appearance must arise
from a supported coating and illumination contrast, not a direction-dependent
visibility trick.

Use local reflection context that includes neighboring windows, frames,
furniture and actual exterior geometry. Extend the planned spatial probe work
to both sides of glazing and to grazing views, with explicit handling of probe
boundaries and screen-space misses. Optional planar reflections need a bounded
budget and same-scene consistency across dataset cameras. Record which
contributors are static and which moving objects the method cannot reflect.

Test reflection and transmission together: a stronger reflection must not be
obtained by adding unaccounted energy. Check tint/frosting against daylight,
direct shadows, indirect illumination and emissive sources. A visibly clear
pane should not acquire an opaque shadow simply because its material path is
absent from the light-visibility implementation. Colored or diffused transmitted
light requires a supported transport approximation and its own reference tests.

### Annotation contract and efficient implementation

Keep the existing semantic taxonomy and stable window IDs. Add optional
assembly/pane IDs and treatment metadata or masks for consumers that need
material-aware supervision. Track frame, opaque insert, clear/frosted region
and exterior fallback separately without requiring new top-level semantic
classes for every tint or finish.

Solid-glass mode should retain the pane's physical first surface. In
transmitted-surface mode, define exactly which material regions the ray skips:
frames, opaque inserts and opaque frit/printing must remain occluders. Frosted
transmissive regions can retain the existing through-glass convention, but
must be marked as having diffuse/ambiguous RGB correspondence. Record that
convention explicitly; frosting strength must not silently switch a pixel's
geometry target. Keep geometric co-visibility separate from optical
correspondence through refractive or scattering regions.

Use one authoritative treatment field for rendering and any treatment-aware
annotation decisions. Define annotation sampling at the documented pixel/ray
convention and report fractional coverage separately where necessary. Match
this behavior across native, WASM, CLI, CPU exports and same-device Burn tensors.
Open-window apertures must reveal supported exterior/neighbor geometry or the
labelled finite sky proxy; neither whole-scene bounds nor O-voxel export should
expand because new backdrop geometry becomes visible.

Compile treatment fields into bounded shared material maps or compact analytic
parameters, choosing by measured cost. Bake static fields once and reuse them
for all cameras/times. Include pane scale, orientation, treatment anchoring,
optics and seed in cache identity; share installation recipes while preserving
distinct instance placement. Bound extra materials, draw calls, refraction passes
and reflection updates. Do not require a new reflection render or large unique
texture for every pane. Keep CPU material/export oracles and readiness barriers.

### Window qualification and delivery

Use a fixed-room local gallery that changes only the window assembly/treatment,
plus unmodified sampled rooms to measure the distribution. Controlled views
should include normal and grazing incidence, close-ups, both sides, multiple
camera baselines, slow motion and day/night illumination. Include unequal panes
sharing a privacy band, partially open sashes, corner glazing, doors beside
windows, patterned film and layered units. Compare the same setups with the
Cycles reference and separately report unsupported transport effects.

Report realized opening-area/wall-area ratio, exposed-wall count, sill/head
height, frame/pane proportions, pane count, operating state, treatment coverage,
pattern scale, roughness and optical-transmission diagnostics. Condition these
on room use and window construction. Also measure projected clear/frosted/frame
fractions in the primary-room views: abundant glass elsewhere in a building is
not evidence of visible window diversity in the dataset.

Acceptance tests must cover construction fit, minimum edge separation, opening
sweeps, aligned treatment boundaries, correct normals from both sides,
angle-dependent reflection, absorption, roughness/mask mips, shadow leakage,
temporal flicker and both annotation policies. Preserve all requested scenes
and report rejected assemblies, unsupported effects and capture cost.

Deliver in three increments:

1. Per-opening optical recipes and shared treatment coordinates; clear/tinted
   panes, partial frosting and opaque pattern masks with annotation fixtures.
2. Richer frame/pane construction and genuinely opened sashes, plus integrated
   local reflection qualification and completed generation benchmarks.
3. Bay/corner/roof assemblies and more demanding layered or curved glazing,
   admitted only after architectural and transport contracts are qualified.

Code homes: `architecture/facade.rs`, `architecture/facade/geometry.rs`,
`envelope/`, `program.rs`, `materials/glass.rs`, `materials/variants.rs`,
`materials/environment.rs`, `materials/gpu/`, `shading/transmission.wesl`,
`metrics/exterior.rs`, `render/annotation_material.rs` and
`render/co_visibility/`. The `render/` paths are relative to `src/`; the other
paths are relative to `src/scene/procedural_indoor/`.

## Objects and botanical construction

Prioritize objects that change silhouette, occlusion and interaction at the
training resolution. Fine details should also improve close views, but they
must not be counted as room diversity if they consistently project below a pixel.

| Construction family | Planned extensions | Required relationships and checks |
| --- | --- | --- |
| Tables and desks | Modular surfaces, returns, connected segments, variable edge profiles, aprons, modesty panels, cable trays, adjustable columns | Load-support assembly, knee volume, leg/top clearance, usable surface and matched wood direction |
| Chairs and soft seating | Seat/back articulation, independent lumbar and headrest dimensions, mesh tension, curved shells, upholstery seams, sectional junctions | Floor contact, stable base, seat occupancy, arm/table clearance and no intersecting cushion faces |
| Storage | Drawers, doors, cubbies, mixed open/closed sections, irregular book groups and stacked containers | Hinge/slide sweep, contents supported inside usable volume, distinct lid and rim surfaces |
| Electronics | Monitor arms and stands, screen assemblies, laptops with consistent hinge/keyboard construction, phones, docks and peripherals | Device ports, reachable inputs, readable viewing direction, cable destinations and support contacts |
| Small objects | Bags, folded fabric, paper stacks, food/drink accessories and containers with variable fill | Task-conditioned grouping, gravity, stacking and contact; do not place every class on every desk |
| Wall and ceiling systems | Panels, rails, outlets, switches, HVAC elements, suspended fixtures and acoustic treatments | Attachment to actual walls/ceilings; regional electrical conventions remain scene-consistent |

Cable generation should extend its existing routed/catenary geometry into a
connection graph: a device socket, slack segment, constrained support/run and
power/data destination. Sample hanging loops, cable trays, bundles, clipped
runs and floor drops in context. Check bend radius, connector orientation,
endpoint attachment and furniture/door interference. Report powered devices
with feasible connections separately from raw cable counts.

For plants, frequency is already substantial: 93% of audited main envelopes
contain plants. Focus on crown construction and placement conditions. Couple
branch order, internode spacing, leaf area, growth direction and petiole length;
vary these by habit and age rather than applying one density multiplier.
Broaden canopy occupancy and asymmetry while keeping roots, stems and leaves
connected. Add paired upper/lower leaf properties, realistic thickness and
bounded transmission, damaged/young leaves as sparse correlated changes, and
pot/soil/root proportions suited to stature. Measure canopy projected area,
leaf overlap, empty-crown fraction, pot fit and render cost across viewing angles.

Keep plant collisions conservative enough for navigation, with finer geometry
checks for visible hard intersections. A coarse crown bound must not force every
plant into identical isolated wall positions.

Code homes: `objects/`, `geometry/slab.rs`, `geometry/sweep.rs`,
`geometry/upholstery.rs`, `objects/equipment/wires/`, `plants/`,
`layout/botany.rs` and `layout/equipment/`.

## Materials and surface appearance

The current implementation already uses PBR maps and metric scale. It has
explicit wood, textile, leather, mineral, ceramic, coating and botanical
programs. A flat-looking result can originate in geometry, reflection context,
texture scale or lighting; adding bump amplitude indiscriminately is not a fix.

Use a hierarchical material program: scene design intent → manufacturer or
furniture group → object part → localized use/aging. Share a finish where real
objects are manufactured together, and vary other finishes independently.
Retain bounded texture banks, but vary object-level cut position, orientation,
panel identity and wear fields so repeated structures do not become identical
surfaces. Current banks have twelve finish slots; only two wood structures or
three upholstery/ceramic structures per role are shared within a room.

- **Wood:** construct related face, edge and end grain from a common cut field;
  preserve grain at connected edges and part boundaries. Add boards, veneer,
  laminate, exposed substrate and stain/finish interactions without arbitrary
  high-frequency stripes. Separate timber figure from damage and dirt.
- **Fabric and leather:** qualify weave scale at room and close-up distances,
  seam direction, panel orientation, leather grain, edge finishing, tension and
  compression. Test sheen/anisotropy before adding finer noise. Clean smooth
  leather and tight upholstery are valid samples.
- **Paint, drywall and concrete:** separate substrate shape, application marks,
  joints, repairs and coating. Most ordinary paint should remain subtle.
  Formwork, troweling and aggregate need related directions and physical size;
  structural seams should not be random texture lines.
- **Ceramic:** correlate glaze pooling with lip, foot and curvature where
  appropriate. Keep body and glaze roughness/color distinct, with unglazed
  bases, thickness variation and restrained defects.
- **Metal:** join manufacturing direction, bevels, seams, brushing and surface
  contamination to the part. Chrome requires credible reflected surroundings;
  increasing metalness cannot restore missing local reflections.
- **Screens, boards and art:** make content follow a scene's activity and object
  use. Reuse a coherent document or presentation across related devices while
  preserving independent content elsewhere. Improve composition, typography,
  stroke/erase construction and screen luminance before increasing pattern count.

Material validation should vary illumination and view independently, inspect
metric swatches and constructed objects, and retain all mips. Check albedo color
space, scalar reflectance, tangent orientation, anisotropy, normal energy,
roughness filtering, seams and grazing-angle aliasing. Keep the current coupled
normal/roughness mip reduction. Where new shading lobes are necessary, qualify
them separately; the [Filament material reference](https://google.github.io/filament/main/filament.html)
describes why standard, coat, anisotropic and cloth responses need distinct treatment.

Code homes: `materials/program.rs`, `materials/variants.rs`, `materials/layers.rs`,
the substrate modules, `materials/filter.rs`, `materials/gpu/` and `shading.rs`.

## Lighting and transport

The sampled lighting range is already broad: the current audit spans roughly
0.11–99,405 solar lux, 3.1–998 electric target lux and EV100 3.01–8.87.
The next goal is coherent illumination and transport across that range.

**First, qualify direct-light occlusion.** In `architecture::spawn_lights`, only
ceiling fixtures with index below eight enable shadows. Later fixtures still
emit. Test bright and dark rooms separated by solid walls, large rooms with more
than eight fixtures, mezzanine decks and independently switched circuits. The
code establishes a potential leakage path; this review does not quantify its
rendered error. A replacement must preserve visibility for every meaningful
contributor. Spatial light influence restrictions, cached static visibility and
shadow allocation are candidates, but proximity ranking alone cannot prevent
unshadowed light leaking through a wall.

Then qualify the differences between visible emitter geometry and lighting.
Ceiling panels currently use spotlights with a fixed radius; changing panel
size does not create a true rectangular area emitter. Evaluate a bounded finite
emitter approximation that preserves flux and produces appropriate highlights
and penumbrae. Account for occlusion of its samples. Increasing the number of
point lights without considering visibility and cost is insufficient. The
[PBRT area-light reference](https://pbr-book.org/4ed/Light_Sources/Area_Lights)
provides the reference transport contract.

Improve reflection locality before spending heavily on microtexture. Current
specular IBL uses one static 64-pixel-per-face scene probe; it has no object-local
parallax and does not reflect moving people. Evaluate bounded room/zone probes
with parallax correction and explicit transition handling. Reserve expensive
planar reflection paths for justified surfaces. Measure errors at multiple
locations and roughnesses, not only at the probe center.

Native diffuse GI is an approximate irradiance volume. Test energy, corners,
partition leakage, glass treatment, thin geometry and dark rooms against the
existing Cycles reference export. Browser Auto currently retains shadows and
refraction but omits this GI and native SSAO; record that capability difference
and evaluate a bounded browser GI strategy independently.

Sample sun elevation, sky distribution, color temperature, clouds, exterior
occlusion, fixture use and exposure as a coherent environment, with explicit
tail mixtures for difficult lighting. Current broad independent priors are
useful for stress, but should not be mistaken for a calibrated distribution of
real offices. Keep exposure separate from illumination and record both.

The window program above defines per-opening optics, spatial treatments,
reflection context and annotation rules. Its transport qualification must also
cover depth, position, normal, semantic, flow and co-visibility together. Neither
screen-space refraction nor a straight through-glass annotation ray should be
presented as a complete multilayer optical solution.

Use two reference tiers: small material/light fixtures that isolate scattering
and calibrated complete rooms that include interactions. Match geometry, maps,
camera, emission, exposure and color transforms. Compare scene-linear radiance
before tone mapping, then inspect display images. Test Cycles convergence with
independent seeds/sample budgets so reference noise is not attributed to Bevy.
Track shadow leakage, contact response, specular localization, glass bias/noise
and temporal stability by region. Path-tracer agreement alone cannot prove that
the generated furniture or architecture resembles real buildings.

## Camera and trajectory diversity

Preserve the current count-aware baseline as the default starting point. It
already expands spacing and aim diversity while enforcing shared visibility.
At baseline 1, the audited median largest optical-axis angle grows from 66.1°
for two cameras to 157.3° for twelve. The cameras are not universally constrained
to one target or a 90° arc. Related path topology remains a separate issue.

The next camera work should address accepted distributions and task utility:

1. Record requested versus accepted path length, route family, height, lens and
   overlap stratum, including retry/fallback causes. The current median path is
   only 0.331 m despite a broad allowed range; bounds are not distribution proof.
2. Sample trajectories from free-space routes and camera intent: translating
   past a surface, partial orbit, room traverse, approach/retreat and independently
   oriented handheld movement. These should be continuous route programs with
   speed/acceleration and orientation constraints, not a bank of fixed curves.
3. Test whether additional cameras increase shared reconstructed surface
   coverage. Keep the reference-connected policy, and evaluate an optional
   connected overlap graph for broader coverage where requiring every view to
   share the same reference becomes restrictive. Do not silently change the
   existing star-policy contract.
4. Report per-pair directed overlap and triangulation, nearest pose distance,
   optical-axis spread, common surface tracks and normalized path similarity.
   Two distinct starting positions can still follow nearly identical motion.
   Measure primary-room geometry separately from windows and sky proxy surfaces.
5. Condition intrinsics on coherent camera families and aspect ratios. Full K,
   dimensions and pixel conventions already exist; the renderer currently uses
   centered pinhole lenses. Qualify off-center projection first, then a versioned
   distortion model only with matching projection, inverse, flow, bounds and
   visibility oracles. Warping RGB alone would invalidate geometry supervision.
6. Retain metric handheld motion and physical time. Correlate intrinsics and
   exposure across an actual rig/sequence when appropriate, while sampling
   different devices across sequences. Keep difficult/negative pairs in declared
   mixtures; do not discard them after downstream model inference.

Depth of field, rolling shutter and motion blur introduce multiple contributing
surfaces/times per pixel. Defer them until their annotation semantics and
training need are explicit. Existing CPU RGB sensor augmentations in the Burn
crate are separate post-tonemapping effects; their presence does not imply an
optical camera model or an efficient same-device augmentation path.

## People and interaction realism

Keep the `burn_human` Anny body, rig and morphology as the anatomical source.
More appearance selectors are less valuable than better fitted constructions.
The current 19 groom selectors already include long, tied, braided and curly
forms, while garments remain fitted/skinned shells with local crease fields.

- Separate sewn panels, garment ease and drape from body shape. Improve shoulder,
  chest, sleeve, crotch and seated folds with constrained fitting and pose-aware
  corrections. Add layered/open garments only with clear attachment and collision
  rules. Preserve uncovered hands and anatomical facial ownership.
- Construct hair from connected roots, guides and clumps with parting, curl,
  coherent tip density and scalp coverage. Avoid disconnected tails and excessive
  fine fragments. Treat the hairline, face, eyewear and beard boundaries as
  geometric and anatomical constraints. Evaluate view-dependent scattering
  separately from silhouette; do not rely on transparent layers to hide poor shape.
- Make behavior interact with the accepted room: sitting/standing transitions,
  reaching a supported object, looking at a screen, writing on a board, stepping
  aside and navigating to a destination. Existing ARDY text diversity must be
  evaluated through generated motion, not prompt-string counts.
- Keep motion optional with no model load on the static path. Preserve cached
  loaders, batching and per-actor randomness. Evaluate root paths, foot sliding,
  contact error, jerk, joint speeds, collisions and visible moving-person pixels;
  report infeasible plans and static fallback separately.

A small controlled portrait/body/interaction gallery should cover body-size
extremes, side/rear views and seated/standing poses under the same lights. Room
captures then establish whether those improvements survive realistic distance,
occlusion and mixed illumination.

## Measuring useful diversity

Treat diversity as a set of conditional distributions, not one scalar space
size. Continuous parameters have unlimited mathematical cardinality but can
produce near-identical images. Ten million samples is an output scale, not a
qualification threshold or a diversity measurement.

Use four complementary levels:

1. **Structure:** canonical room/portal and assembly graphs, support relations,
   coarse occupancy, silhouette descriptors and contact patterns. Factor out
   global yaw, IDs and irrelevant seed changes. Report duplicate/family counts
   and occupancy effective sample size at explicitly stated quantization scales.
2. **Appearance:** fixed SigLIP2 embeddings of whole views, room-weighted
   aggregates and object crops. Preserve model hash, preprocessing, resolution
   and color encoding. Measure cross-family nearest neighbors, duplicate tails,
   effective rank and coverage of held-out real-scene clusters. The
   [SigLIP2 paper](https://arxiv.org/abs/2502.14786) supports its use as a learned
   visual representation; using its distance as a realism score would be an
   additional unvalidated assumption.
3. **Geometry supervision:** observable primary-room surface coverage,
   track length, triangulation, depth range, occlusion, texture ambiguity and
   viewpoint/motion novelty. Distinguish true geometry variation from new camera
   views of the same room and RGB-only recoloring.
4. **Learning utility:** controlled downstream reconstruction/calibration runs
   with identical sample/compute budgets and fixed family splits. Report real
   validation generalization, stability and throughput. No generator metric alone
   establishes better training.

The existing Rust embedding study already computes cross-room nearest neighbors,
effective rank and shared-PCA volume. Extend it rather than introducing a separate
script pipeline. Its paired volume diagnostic currently accepts at most 256
rooms and uses ranks 4/8/16. For larger cohorts, introduce bounded streaming
statistics and a validated approximate-neighbor index, with exact checks on a
held-out subset. Aggregate by independent room family, not correlated views.

Calibrate perceptual duplicate thresholds using pairs with known relationships:
identical replay, camera-only change, illumination-only change, material-only
change, small geometric changes and unrelated rooms. Preserve both canonical
captures and ordinary distribution captures. Bootstrap rooms/families and keep
the projection basis fixed when comparing embedding volumes. Volume can grow
because of unnatural colors or noise; quality gates and conditional coverage
must accompany it. A “10×” claim requires a predeclared, stable measured meaning;
no product of selector counts or volume ratio is sufficient by itself.

Keep matched before/after studies local. Published page/paper snapshots should
describe the current generator and its protocol, with complete denominators,
failures and provenance. Reuse `bevy_zeroverse_publication` to build both surfaces.

## Qualification and performance gates

Use escalating checks so inexpensive failures are found before costly rendering.

| Scope | Proposed protocol | Promotion requirement |
| --- | --- | --- |
| Each construction change | Analytic fixtures, adversarial dimensions and deterministic replay | Finite/wound geometry, actual support/contact/clearance, no unintended coincident faces |
| Routine distribution gate | 1,024 consecutive rooms with actual geometry audit plus explicit extreme cases | No silently dropped seeds; valid topology, layout, labels and declared function |
| Local visual iteration | 64 fixed rooms, 3 × 512² cameras, 2 times; matched geometry when testing appearance | All rooms retained; annotated gallery, artifact triage and measured cost |
| Release qualification | 512 rendered rooms: 256 consecutive, 128 prespecified factor strata, 128 held-out; same 3 cameras and 2 times = 3,072 views | Separate stratum/population estimates, current provenance, annotation/quality gates |
| Camera qualification | Baseline 0/.25/.5/.75/1, N=1…12, several aspect ratios and density/lighting strata | CPU distributions plus stratified rendered checks; record every failed request |
| Lighting/material reference | Isolated fixtures plus prespecified whole rooms, independently converged Cycles references | Bounded radiometric/transport errors established by fixtures, no unexplained regional regressions |
| Window qualification | Matched opening/treatment fixtures, both sides and grazing angles, open/closed states, day/night, sampled complete rooms | Construction and optical consistency, filtered partial masks, both glass policies, no flicker or unlabelled exterior gaps |
| Larger coverage | 10,000+ manifest audit, sampled full construction and staged render/embedding expansion | Conditional tail coverage and collision estimates with uncertainty |
| Runtime | Repeated completed captures and actual consumer steps, native and hardware Chrome | Quality held fixed, bounded memory, stable numerical replay and measured throughput |

Hard annotation gates cover depth/position consistency, normals, semantics,
bounding boxes, timestep-based optical flow, co-visibility masks and primary-room
O-voxel bounds. Test both glass policies. Keep geometric normals distinct from
material normal maps. Structural changes must update all annotations; RGB-only
changes must not change their geometry. Retain the current range-aware numeric
tolerances rather than relaxing them to accommodate a new effect.

For coplanar failures, combine construction-level face overlap checks with a
slow-camera temporal fixture. Detect bright/dark or semantic ownership changes
at nearly fixed poses; image similarity at one frame cannot detect z-fighting.
Permit intentional contacts through declared ownership rules, not arbitrary
depth bias that changes geometry labels.

For performance, maintain separate budgets for layout, construction, CPU
materials, GPU material baking, transport preparation, GI/reflection work,
uploads, rendering, tensor handoff and consumer training. Report P50/P95/P99
latency, completed rooms/views/steps per second, CPU time, device utilization,
peak memory and steady-state memory slope. A faster isolated shader or higher GPU
utilization is not sufficient if completed training slows down.

The native preparation pool is already bounded at four threads and WASM work
cooperatively yields around a four-millisecond budget; browser yielding is not
parallel execution. Preserve deterministic output ordering and bounded successor
residency. Off-main-thread browser preparation should be evaluated only if
profiling justifies its worker/asset-transfer complexity.

GPU material synthesis already covers fabric, knit, carpet, leather, metal,
chrome, plastic, rubber and paper. Extend costly remaining substrates according
to measured CPU time, retaining CPU recipe oracles, full normal/roughness mips,
reference-export behavior, capture readiness and standard published WebGPU APIs.
Keep generated maps cached for rendering rather than recomputing them per
fragment/view. Shared-device training contention is part of the benchmark.

Do not move the entire constraint solver to GPU preemptively. Irregular topology,
branching placement and reproducible admission may benefit more from batched
CPU preparation and better proposal efficiency. GPU geometry kernels are useful
only after profiling identifies sufficiently regular, expensive work. Do not
trade shadows, material detail, people, texture resolution or annotation
correctness for a throughput headline.

## Modular delivery and first implementation tranche

Keep a versioned manifest as the source of scene intent, followed by typed
construction recipes, accepted relationships, actual render assets and shared
annotation geometry. Sampling, validity checks, rendering and metrics should
remain separate modules. Use independent RNG domains and stable identities;
adding one prop or finish must not accidentally resample unrelated cameras,
lighting or people.

Add new policy fields with validated defaults and explicit provenance. Store
sampled and realized settings when rejection or recovery changes the outcome.
Expose useful dataset controls such as activity mixture, coverage intent and
appearance variation; internal cache sizes and worker counts should not become
mandatory downstream tuning. Make native UI, URL serialization, CLI and Rust
API derive their controls from the same definitions.

The first implementation tranche should contain four reviewable changes:

1. **Diagnostic contract:** distinguish primary viewing zone, other main-envelope
   zones and neighbor; record assembly intent, accepted relations, rejection
   causes, operational coverage and candidate-to-accepted distributions.
2. **Lighting fixture tests and fix:** reproduce the eight-shadow limit in
   partitioned rooms, select an occlusion-correct strategy, and verify the same
   scene on native and hardware WebGPU. Preserve full-quality defaults.
3. **Conference/workstation assemblies:** complete the group-placement slice,
   including pushed-in occupied chairs, mixed tabletop use, usable screens and
   connected cables; evaluate true silhouettes and clearance rather than counts.
4. **Reflection/material qualification:** add spatial reflection fixtures and
   physical-scale material crops, including per-opening tint and partial-frosting
   fixtures with both annotation policies. Implement the smallest measured fix
   for flat chrome, overlarge patterns or incoherent surface identity. Window
   construction then expands through the dedicated three-increment plan.

Promote that tranche only after the local 64-room gallery, geometry/annotation
checks and completed generation/training measurements agree. Use the results to
choose the next architectural or human-quality tranche. Maintain an explicit
list of tested capabilities and remaining limitations; photographic realism
and ten-million-sample utility remain empirical outcomes to demonstrate.
