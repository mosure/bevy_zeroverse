# Indoor room and object expansion plan

Expand procedural_indoor through functional spatial programs: transit, food
preparation, service, dining, learning, presentation, exhibition and product
display. Room names should select distributions over these functions. Geometry,
furniture, services, lighting, people and cameras should then be sampled together
subject to construction and use constraints.

The [kitchen and kitchenette implementation](procedural_kitchens.md) provides
service runs, appliances, storage, supported accessories and dining assemblies.
The [public-space implementation](procedural_public_spaces.md) adds the remaining
room families, functional receipts, support exports and qualification tools.
The contracts and qualification gates below define the completion criteria;
implementation alone does not satisfy the platform or downstream-learning gates. Existing scene modes and capture APIs remain supported. New programs
should initially be opt-in; changing the default mixed distribution requires
qualification and a generator-version change.

**Current foundation**

The working source has 100 ObjectKind entries, twenty named indoor profiles plus
Mixed, and 36 material roles. These are catalog counts, not estimates of useful
scene diversity. The main opportunities are visible in the current contracts:

| Foundation | Current boundary | Expansion |
| --- | --- | --- |
| [ActivityMix](../src/scene/procedural_indoor/program/activity.rs) | Four shared weights plus dedicated kitchen profiles | Typed functional requirements and compatible activity mixtures |
| [SceneDomain](../src/scene/procedural_indoor/domain.rs) | Shared room-size sampler, with kitchen-specific bounds applied during layout | Purpose-conditioned dimensions, capacity and height, including compact service rooms and larger halls |
| [Zone and Partition](../src/scene/procedural_indoor/program.rs) | Rectangular zone bounds and axis-aligned partitions; an archived furnished zone defines the primary viewing region | Polygonal regions, arbitrary wall orientation, explicit program ownership and reconstruction regions |
| [EnvelopeProgram](../src/scene/procedural_indoor/envelope/mod.rs) | Simple polygon, floor patches, planar outer roof, soffits and an optional mezzanine | Connected support regions, tiered seating, vestibules, multiple ceiling fields and explicit vertical connections |
| [FunctionalAssembly](../src/scene/procedural_indoor/layout/assemblies.rs) | Workstation/meeting receipts and kitchen service-run membership | Explicit dining groups, teaching rows, display groups and transit clearances |
| [IndoorObject](../src/scene/procedural_indoor/layout.rs) | Existing support, interaction, cable and finish relationships | Attachment sockets, operational volumes, nested support and articulated state |
| [MaterialRecipe](../src/scene/procedural_indoor/materials/program.rs) | Correlated material layers and role-specific finishes | Construction-scale continuity, additional substrates and room-specific finish assignments |
| [Preparation](../src/scene/procedural_indoor/preparation.rs) | Bounded workers, staged assets, GPU material requests and cooperative browser work | Reuse these paths for new assemblies; measure cost by accepted object/program |

Kitchen builders reuse the existing Counter, Sink and Refrigerator semantic
classes, and append appliance, dishware, utensil, food and cleaning-supply
classes. Stools, banquettes, curved counters and different cupboard fronts
should be geometry programs or subtypes where their identity permits, rather
than one semantic class for every visual style.

**Spatial and construction contracts**

Evolve the existing manifest and program, with versioned conversion for archived
data. Avoid a second independent layout description beside the one used for
collision, cameras and exports.

1. Give each program a set of required functions, optional functions, capacity
   targets, compatible neighbors and spatial relationships. A kitchenette needs
   an accepted preparation surface and its configured service equipment. A
   gallery needs accepted exhibit surfaces and a connected viewing route.
   Counting arbitrary chairs or cabinets must not satisfy these requirements.
2. Extend the region/portal graph with wall-local frames, passage types,
   connectivity and boundary ownership. Resolve shared walls once. Build
   openings, collision volumes, navigation connections and annotation ownership
   from the same accepted opening. Support offset rooms, alcoves, irregular
   vestibules, splayed walls and loops without requiring arbitrary CAD or
   unrestricted boolean operations.
3. Introduce named support regions with a surface frame and height model.
   Furniture, feet, camera paths and floor details must bind to the actual
   region. A lecture tier, its adjacent stair tread and an overhead deck cannot
   be represented adequately by one floor height at each X/Z coordinate.
   Retain the current flat-floor path as the inexpensive special case.
4. Make the reconstruction region explicit. Usually it is one functional room;
   it can be an explicitly requested connected collection of gallery bays or a
   cafe dining area. Its geometry membership, primary camera domain, schematic,
   annotated AABB and O-voxel scope must agree. Do not automatically include a
   kitchen behind a service opening, an elevator shaft or the exterior backdrop.
5. Add assembly roles for service runs, dining groups, audience rows, display
   stations, waiting groups and equipment stations. Store required members,
   accepted members, capacity and rejection reasons. Allow nested groups without
   losing stable individual object identities.
6. Distinguish physical occupancy, support/contact, opening sweep, reach space,
   viewing space and circulation. A drawer's operating volume is different from
   its closed mesh; a stool can fit under a counter overhang while its back
   cannot intersect the worktop. Clearance rules should describe the relevant
   relationship rather than inflate every object equally.
7. Reserve service anchors and routes before dressing surfaces: electrical
   outlets, plumbing endpoints, ventilation runs, AV connections and fixture
   mounts. Route visible wires and pipes between actual endpoints. Concealed
   infrastructure can remain a logical connection until it becomes visible.
8. Represent state explicitly: door opening, drawer extension, folding seat
   angle, screen angle, appliance contents and shade deployment. Static samples
   can use these states immediately. Time-varying state requires the existing
   timeline, motion-vector history, readiness and annotation contracts.

Each new function should first work within current room envelopes. Generalized
partitions and vertical supports should enter when a concrete room program needs
them, with conversion tests and a complete rendered example at each step.

Compact rooms need particular care: the present default size sampler clamps
horizontal dimensions to 5.6–21 metres, and program validation requires zone
dimensions of at least 3 metres. Those are existing generator limits, not
universal architectural requirements. Replace them with purpose-specific
feasibility checks while retaining the clearance actually needed by an accepted
assembly. Larger halls also need scale-aware light placement, texture density
and geometry budgets. Permit cycles in the region graph explicitly; current
partition-count assumptions must not accidentally reject valid circulation loops.

Calibrate each purpose using a small reference sheet of real room photographs,
layouts, proportions and object relationships. These guide priors and visual
review; they are not runtime mesh/image prerequisites. Sample a shared building
context first, including construction era, service intensity, maintenance,
finish palette and institutional versus personal use. Then sample local
variation. This lets adjoining rooms differ without looking unrelated.

**Room programs**

**Elevator landings and connected lobbies.** Generate a bank of door assemblies
along one or several compatible walls, with recessed jambs, reveals, sills,
sliding panels, call buttons, floor indicators, directory signs and wall
protection. Vary bank count, panel split, recess depth, wall facing, ceiling
height, corridor angle, lobby aspect, waiting area and distance between doors.
One small landing with a side corridor and a broad shared lobby should emerge
from the same transit program.

Required geometry includes a plausible closed shaft boundary; when a door is
open, generate a bounded car interior, aligned floor, door pockets and rear
surface. An open door must never reveal missing geometry. Sample metal,
painted-panel, stone or timber surrounds; brushed stainless doors; inset
lighting; entrance mats; benches and plants outside approach zones. Model
queuing/waiting locations separately from through-circulation. Keep door sweeps,
landing access and the corridor connected. Elevator travel simulation can remain
outside the first delivery; static car/door states already provide useful variety.

**Kitchenettes.** Build fitted counter runs from compatible cabinets, drawers,
end panels, toe kicks, worktops, splashbacks and optional wall cupboards.
Add sinks with actual bowls, drains and taps; compact refrigerators; microwaves;
dishwashers; and existing coffee makers/kettles. Support a variable subset, so a
dry refreshment station is distinct from a sink-equipped kitchenette.

Vary run length/depth, straight and turning segments, corner treatment, worktop
overhang, upper-cabinet coverage, appliance grouping and shelving. Reserve
appliance openings and standing space before placing a small dining group.
Generate supported cups, dishes, utensils, towels, drying racks, containers,
coffee supplies and waste/recycling bins with context-dependent quantities.
Cut the sink opening in the top; do not place a bowl through a solid slab.
Close unused cabinet ends and avoid intersecting handles at inside corners.

**Kitchens.** Extend the kitchenette system with islands, peninsulas, parallel
runs, cooking stations, ovens, cooktops, extraction hoods, pantry shelving and
larger preparation areas. Sample domestic-style communal kitchens and compact
institutional preparation areas through equipment and service-intensity priors.
Do not force every configuration to follow one work-triangle layout: use
storage/preparation/washing/cooking adjacency and workable routes.

Counter segments need compatible corner joints, finished edges, appliance
cutouts, wall junctions and material continuity. Doors and drawers may be partly
open only when their state passes clearance checks. Add bounded procedural
plates, bowls, pots, pans, cutting boards, utensils and simple food/container
geometry where visible; detailed food simulation is unnecessary for the first
scope. Electrical, wet and heated equipment should occupy plausible service
locations. Ventilation geometry should connect to a ceiling or wall route.

**Cafes.** Combine a service counter, order/pickup zones and flexible dining.
New equipment includes configurable espresso-machine assemblies, grinders,
pastry/display cases, menu boards, a payment terminal, condiment stations and
tray or dish collection. Reuse the vessel, screen, lamp, chair, table and plant
systems while adding booths, wall banquettes, window bars and linked seating.

Sample counter curvature and segment count, dining-group spacing, seating
orientation, queue depth, window frontage, overhead lighting and degree of
personal occupation. Make cups, bags, phones and laptops reflect a plausible
table use state. A person should face their companion, work surface or service
point as appropriate. Keep staff space behind the counter distinct from the
public route. Window seating should relate to actual sill and glazing geometry.

**Cafeterias and food courts.** Compose multiple serving stations, tray pickup,
payment and return areas with broad dining fields. Add serving counters with
insets, sneeze guards, beverage dispensers, trays, cutlery stations, clearing
trolleys and multi-stream bins. Shared builders should support both compact
staff cafeterias and larger institutional rooms.

Vary station ordering, number of queues, dining islands, long communal tables,
fixed bench/table combinations, columns and acoustic ceiling fields. Use
curvature, local alignments and incomplete occupancy within the dining field;
retain enough organization for cleaning and circulation. Distinguish public,
staff and tray-return paths. Dense populated variants need clear admission
limits and cost receipts, rather than unbounded crowd or prop counts.

**Classrooms and teaching studios.** Add individual/student desks, paired desks,
tablet-arm seats, teaching benches, lecterns, storage cubbies, lockers and
mobile teaching stations. Reuse whiteboards, projectors, displays, shelving,
books and stationery, with course/context-dependent contents. Desk height,
seating height and occupant morphology must agree.

Sample directed rows, curved rows, seminar rings, horseshoes, collaborative
islands and hybrid arrangements through continuous row/group fields. Vary
orientation, row pitch, gaps, local fan, desk attachment, teacher-front
orientation, writing-surface area and occupancy. Preserve a teaching area,
access to boards/storage and connected aisles. Adult education is a complete
initial population; other age groups require independently validated body and
furniture priors rather than simply shrinking adult meshes.

**Lecture halls and auditoria.** Build on classroom seating with curved and
segmented rows, linked seats, foldable seat pans, writing tablets, shared
supports, raised teaching platforms and tiered or gently raked floors.
Add stair aisles, landings, guard geometry where needed, railings, wall acoustic
panels, lecture consoles and appropriate lighting circuits.

Parameterize row curvature, fan angle, pitch, tier height, aisle count/location,
front distance, stage height, seat widths and interrupted rows. Seating must
follow its support elevation; the observer and seated head sightlines should
be evaluated toward the teaching region, with their intended limitations
recorded. A person walking an aisle must follow stairs instead of interpolating
through tiers. Use per-seat instance identity even when identical row components
share GPU resources. This family depends on the support-region work and should
not be approximated by chairs placed on a decorative sloped slab.

**Showrooms.** Combine product islands, demonstration stations, wall displays,
plinths, freestanding racks, information stands and sales/reception furniture.
Start with coherent product collections made from existing furniture, lighting,
electronics, cabinets and plants. Add representative appliances through the
kitchen work; avoid requiring a large external product mesh library.

Vary exhibit spacing, island size/height, pedestal outline, wall-bay rhythm,
group composition, staff desk placement and demonstration accessibility.
Objects from a displayed collection should share selected manufacturing and
finish characteristics while differing in size or configuration. Keep deliberate
product alignments and asymmetry, rather than applying random yaw everywhere.
Add labels, screens and spotlights oriented toward the corresponding exhibit.
Glass cases need frames, base support and interior objects, with separate
material and annotation ownership.

**Galleries and exhibition rooms.** Generate wall-hung work, freestanding panels,
framed canvases, prints, sculptural objects, plinths, vitrines, labels, benches
and optional barrier posts/ropes. Procedural art should include shape,
composition, brush/ink marks, palette and frame/mounting variation. Sculptures
can use connected sweeps, solids, reliefs and assembled forms without depending
on large source assets.

Sample exhibit spacing, height, size hierarchy, wall composition, internal panel
angles, viewing distance, alcoves, openings and loops. Pair exhibits with
viewing regions and aiming lights. Avoid blocking an exhibit with its label,
bench or neighboring work. Include quiet, sparse galleries as a legitimate
domain, while still preserving geometric learning signal through views of
multiple surfaces and depth layers. Sparse rooms should not fail an office
furniture-count rule.

These nine programs should also compose naturally: elevator landing to gallery,
cafe beside a classroom, kitchenette inside a coworking suite, or showroom with
a small consultation area. Compatibility and service relationships determine
valid combinations; uniform sampling of every cross-product would create
implausible rooms.

Sample dimensions and group parameters continuously within conditional feasible
domains; integer counts and discrete construction choices remain legitimate
parts of the program. Use broad distributions with common interior mass and
bounded tails, and record requested versus realized parameter distributions
after placement. Interpolation is appropriate within a construction family,
such as row curvature or counter depth; it need not interpolate between
incompatible constructions. Dataset manifests should preserve the sampled
parameters, program version and domain-separated seeds for replay.

**Reusable object and geometry work**

| Builder family | Continuous construction controls | Shared correctness requirements |
| --- | --- | --- |
| Fitted cabinet/counter system | Module widths, panel thickness, door/drawer subdivision, corner angle, handle profile, toe kick, worktop outline and openings | Finished exposed ends, joined seams, reachable storage, exact support surfaces |
| Hollow fixtures and containers | Bowl profiles, wall thickness, rim curvature, basins, faucet bends, drains, tray lip and handle attachments | Closed visible shells, no caps over openings, bounded contents and supported stacking |
| Appliance enclosure | Housing aspect, front-panel partitions, vents, controls, hinge/pivot, feet, door windows and internal shelves | Clear opening state, material separation, visible internals when open |
| Seating and audience system | Curved shell/back, seat height/tilt, linkage, bench segmentation, tablet arms, banquette corners and shared supports | Human fit, under-surface clearance, aisle access and per-seat identity |
| Display and exhibit system | Plinth profiles, mounts, shelves, framed panels, glass enclosure, hangers, sign stands | Support/attachment, view access, case interior containment and clear labels |
| Door and service opening | Sliding/hinged leaves, recess depth, sill, jamb profile, aperture, pocket and opening fraction | Mesh/portal agreement, finite geometry behind openings, valid swept volumes |
| Circulation and vertical support | Landings, tread/riser dimensions, tier pitches, ramp surfaces, rails and edge returns | Continuous routes, local headroom, feet/camera support and correct primary membership |
| Architectural lining | Tile bands, wainscot, battens, acoustic fins, soffit edges, cornices and wall protection | Metric spacing, complete corner joints, noncoincident boundaries and stable surface labels |
| Small functional props | Dish stacks, cutlery, trays, dispensers, labels, menus, cleaning/storage supplies and teaching equipment | Context-conditioned occurrence, stable support, plausible scale and bounded detail |

Add generic sweeps, rounded profiles, offset panels and annular/open-shell
builders only where multiple objects share a need. Prefer analytic construction
with explicit thickness and cutout topology to repeated whole-mesh booleans.
Compound collisions should represent occupied parts where a single bounding box
would reject valid under-counter seating or open shelving.

Existing plants, personal electronics, cables, switches, outlets, bins and
lighting remain important across all families. Their placement probability
should depend on use and services: trailing plants near suitable shelves,
tabletop foliage outside work areas, phones at occupied seating, and powered
equipment near actual connections. Large floors should receive meaningful
zones, border treatments, mats and wear fields instead of uniform repetitions.

**Material and appearance expansion**

The material system should separate substrate, fabrication, finish and use.
Shared finish identity should give a row of cabinets or a showroom collection
coherence, while local grain phase, cut direction, seams, wear and contamination
provide bounded differences. Reuse current Surface roles and typed recipe
extensions; do not multiply Surface entries for every finish style.

| Material family | Additional procedural dimensions | Important checks |
| --- | --- | --- |
| Stainless steel and coated metal | Directional brushing, polish, panel grain alignment, stamped relief, fingerprints and handling wear | Dielectric coatings stay separate from metal; reflections and filtered roughness must remain plausible |
| Ceramic, porcelain and glazed tile | Body/glaze layers, glaze pooling, speckle, tile format, grout depth, edge rounding and chipped tails | Continuity at corners, correlated maps and restrained roughness/relief |
| Stone, terrazzo and solid worktops | Aggregate size/composition, directional veining, slab cuts, edge polish and seam placement | Plausible world scale; cut edges relate to the same material volume |
| Laminate, veneer and engineered panels | Edge banding, face/edge construction, cut orientation, wood stain and finish sheen | No stretched end grain or identical grain phase across a complete run |
| Resilient floors | Linoleum/vinyl/rubber patterns, flecks, sheet seams, tile boundaries and local polish | Stable detail at distance and distinct reflection behavior from actual metal |
| Painted plaster, enamel and wallcovering | Roller/stipple relief, gloss, pigment, paper/fabric backing, joins and bounded contact wear | Paint uses dielectric response; relief survives filtering without specular aliasing |
| Acoustic materials | Felt/fabric, cork, perforated timber/metal and mineral panels | Large perforations are geometry; distant microdetail filters without false holes or shimmer |
| Upholstery and curtains | Woven/coated fabrics, leather, patterned panels, piping and directional texture | Geometry follows seams/tension; verify all mips and grazing response on native and Metal WebGPU |
| Art, signage and display content | Paper/canvas relief, print grain, ink coverage, menus, teaching diagrams, information layouts and screen themes | Content and geometry IDs are distinct; emission never substitutes for a physical light calibration |
| Glass and transparent covers | Thickness, transmission, tint, roughness, partial frosting, coatings and distinct pane/frame construction | Correct glass policy across annotations; stable temporal filtering and finite geometry behind clear openings |

New optical effects must fit the renderer's supported material model and
qualification. Do not represent physically unsupported reflection/refraction
behavior as ground-truth photographic accuracy.

Wear should follow contact and exposure: countertop work zones, cabinet pulls,
traffic paths, chair contact, shelf edges and window exposure. Sample clean and
maintained spaces as well as used ones. Avoid decorating every surface with
independent grime or mapping one noise field into every PBR channel.

Lighting should follow functions and fixtures: under-cabinet task lights,
counter pendants, display spots, wall washers, audience/teaching circuits,
elevator indicators and general illumination. Sample circuit activation,
illuminance, color temperature, daylight admission, sun direction and exposure
with consistent fixture geometry. Include daylight, mixed light, evening and
low-light conditions. Keep exposure and RGB sensor transforms separately
recorded from material reflectance and illumination.

Window placement should respond to function: broad cafe frontage, kitchen
windows above compatible work surfaces, glare-controlled teaching areas,
clerestories and selectively daylighted galleries. Vary aperture, reveals,
frames, mullions, operability, tints and partial treatments. Extend exterior
geometry to cover every new visible opening; a glass door, skylight or open
elevator cannot expose the background sentinel.

**Placement, people and use**

Use a deterministic dependency order:

1. Sample room function, capacity, scale, style and use intensity.
2. Resolve the envelope, service walls, portals and reconstruction region.
3. Reserve main circulation, queues, operational clearances and teaching/viewing
   areas.
4. Fit large functional assemblies transactionally.
5. Resolve supports, fixed services and mounted fixtures.
6. Place movable furniture around the accepted assemblies.
7. Apply articulated furniture/appliance state and validate its swept volumes.
8. Populate supported props using station/person use state.
9. Route exposed cables and services around final geometry.
10. Place people and fit contacts/gaze/gestures; plan optional motion through
    accepted routes.
11. Admit cameras and complete trajectories against the realized geometry and
    occupied human envelopes.
12. Validate, materialize assets, wait for readiness and capture.

Circulation reservation should protect useful potential camera space without
removing legitimate difficult layouts. Record any adjustment or fallback.
Keep bounds on retries; if a required function cannot fit, report an explicit
failed program or a declared bounded repair. Never quietly replace a rejected
kitchen with generic office clutter.

Sample placement at several scales: a coherent room organization, correlated
variation within a group, then small use-related offsets. Chairs can be tucked,
turned or pulled out; cups and phones can be near a user's hand; trays can queue
at return points. Correctly aligned cabinets and lecture rows are realistic.
The aim is to avoid accidental uniform alignment across unrelated objects.

Continue using the Anny body, fitted clothing and current static pose system.
Add support-aware activities such as waiting at a lift, reaching to a call
button, preparing a drink, carrying a tray, pointing to a board, writing,
viewing exhibits and demonstrating a product. Fit hand, seat, foot and gaze
targets to actual object sockets and people. Task, dominance and interaction
distance should bias continuous gestures and torso/head turns.

Optional ARDY sequences can cover approach, interaction and departure, including
stairs only after route/support validation exists. Preserve static people and
zero model initialization when motion is unspecified. Captured poses should
continue through the [existing pose API](human_poses.md), with optional
relationship labels such as hand-to-handle contact. Model prompts alone do not
establish contact or collision correctness.

**Capture, semantics and multi-view training**

Extend the authoritative taxonomy deliberately. Reuse existing cabinet, counter,
sink, refrigerator, chair, bench, lamp and appliance classes; add stable labels
for genuinely new distinctions such as lecterns, display cases, elevator
controls or tray-return equipment only where useful. Keep semantic class,
object subtype, assembly role, part role and material identity separate.
Publish a coarse mapping and a taxonomy version; never change existing colors
or IDs silently.

Every new builder must preserve instance ownership through batching and export
boxes, actual geometry, poses where applicable and relationship metadata.
Support RGB, depth, normals, position, semantics, optical flow, co-visibility
and schematic capture through the existing paths. Maintain single-timestep,
motion-free O-voxel restrictions and exact reconstruction-region membership.
Opaque bodies, glass frames and transmissive panes need the existing glass
annotation policy. Screen-content changes are appearance changes, not rigid
surface motion; retain that distinction in temporal supervision.

Use the current baseline, path and overlap controls. New room priors should
condition camera admission on usable surfaces, depth layers and circulation,
without forcing all cameras to one optical-axis intersection. A cafe can share
table/counter geometry across views; a lecture hall can connect aisle views with
oblique teaching-area views; a gallery can connect adjacent exhibit bays.
Per-request primary membership remains explicit.

For one camera, measure surface/depth coverage. For 2–12 cameras, measure directed
rendered overlap, shared primary geometry, metric baseline, triangulation
angles, view-graph connectivity and trajectory similarity. Where space is too
tight for the requested baseline/count, return admission diagnostics or the
declared fallback instead of nearly duplicated views with an implied guarantee.
Keep approximate placement overlap separate from measured rendered overlap.

Retain explicit high/low/no-overlap training strata sampled before rendering.
Do not reject difficult or negative pairs after evaluating a downstream model.
Keep topology/content family IDs stable across frames, alternate cameras and
appearance changes, so dataset splitting can prevent leakage. Store physical
seconds separately from normalized trajectory progress.

**Configuration and code organization**

Keep procedural_indoor as the scene type. Add room-purpose presets and weighted
composition through typed configuration rather than a separate renderer or
top-level scene implementation for each family. Extend ActivityMix and the
assembly contracts; preserve the interpretation of old serialized values.

Simple controls should cover purpose/mix, scale/capacity, furnishing/occupation,
disorder, daylight/artificial light, people and camera baseline. Advanced
controls expose validated parameter ranges, required assemblies and conditional
weights. A purpose preset must expand into an inspectable sampled program.
Record the effective settings, accepted realization and any capacity shortfall.
Native UI, WASM query round trips, CLI, library and Burn consumers should share
the same schema. Apply scene edits explicitly so slider interaction remains
stable; serialize the active seed and effective configuration into shareable URLs.

Suggested code ownership follows the existing modules:

| Responsibility | Existing home and proposed extension |
| --- | --- |
| Program sampling | program/activity.rs and new focused modules for services, audience and exhibition programs |
| Spatial topology/support | program/topology.rs, envelope/ and their shared queries |
| Transactional placement | layout/assemblies.rs and focused service, dining, audience and exhibition placement modules |
| Object construction | objects/storage.rs, hospitality.rs, seating.rs and new appliance, exhibit and transit builders |
| Material families | materials/ typed recipes, GPU kernels, CPU oracles and mip qualification |
| Semantics/capture | render/semantic.rs, annotation/, sample metadata and crates/capture |
| Metrics and reports | metrics/, validation.rs and the Rust crates/publication pipeline |

Extract small shared primitives when the second use appears. Avoid one huge
match statement containing all room behavior or a general-purpose constraint
language that must be completed before any new room works.

**Performance design**

The expansion should retain the efficient default native/Burn path and a
responsive browser. Extra configuration should primarily describe desired
content, not require downstream users to discover worker/prefetch tuning.

| Potential cost | Planned mitigation | Qualification |
| --- | --- | --- |
| Dense furniture/prop constraints | Spatial indexing, assembly-local checks, early analytic rejection and bounded repair | Accepted distribution, rejection reasons, layout P50/P95/P99 and worst seeds |
| Repeated rows/cabinet modules | Share exact meshes/materials for repeated components; vary instance transforms and logical identity | Per-instance annotations remain distinct; no forced quantization of requested dimensions |
| More unique geometry | Independent deterministic construction jobs in the existing bounded pool | Serial/parallel manifest and mesh parity; allocation and service-time profiles |
| Expanded material catalog | Select only referenced recipes; retain cached GPU synthesis and mip readiness | All-map/all-mip CPU/GPU parity, residency bounds and native/Metal images |
| More fixtures and glass | Coherent fixture groups, existing light/probe strategy and measured shadow/reflection cost | Same render quality and glass policy; no hidden disabling of shadows |
| Multi-room lookahead | Bounded existing preparation/staging with complete request identity and cancellation | No stale room, unbounded cache or lifetime leak |
| Browser long tasks | Cooperative jobs with bounded mesh/construction chunks; no assumption of browser worker support | Real WebGPU adapter, input responsiveness, long-task and memory traces |
| JIT training transport | Preserve GPU-resident WGPU/CUDA routes and metadata-only capture helpers | Completed tensor parity, ownership/lifetime and concurrent optimizer throughput |

Cache topology or exact immutable parts, not entire randomly sampled rooms
indefinitely. Cache keys must include all geometry/material/state determinants.
Derive object detail once for a capture set so views and annotations agree.
Do not achieve throughput by changing silhouettes, reducing diversity, removing
objects or weakening clearance.

Measure completed three-view 512×512 delivery, preparation critical path,
GPU busy/submission gaps, CPU time, GPU memory, process RSS and warmed/cold
latency separately. Compare identical accepted manifests and quality settings.
Report cost as scene complexity grows; a dense cafeteria naturally has different
work from an empty landing. Existing workloads should show no repeatable
regression beyond a proposed 5% measurement allowance before promotion.
New programs need measured complexity budgets and a reference implementation,
not an unsupported universal rooms-per-second promise.

**Qualification and diversity analysis**

These are proposed execution gates, not results of this planning review.
All required failures remain in the denominator with source identity, requested
program and failure reason. Recovery must not hide missing program coverage.
Promotion requires zero invalid emitted scenes or misaligned annotation records
in the qualification campaign. An explicitly rejected infeasible request is a
different outcome; report it by program and parameter stratum, and set admission
targets before tuning so the sampler cannot improve its score by excluding
difficult cases.

| Gate | Proposed coverage | Required evidence |
| --- | --- | --- |
| Builder correctness | Boundary/adversarial dimensions and deterministic samples per new builder | Finite meshes, valid triangles/winding/normals, correct bounds, attachment, thickness and noncoincident visible surfaces |
| Functional placement | Each program at sparse/dense, compact/large and difficult-envelope settings | Required functions accepted; support, reach, opening, circulation and service constraints satisfied |
| Structural campaign | 1,024 independent requests for each of the nine new families plus 1,024 mixed programs: 10,240 total | Manifest and actual constructed-geometry audits; documented admission/fallback rates |
| Incremental render review | 16 rooms per family during development | Aligned RGB/annotation views, schematic overlays, close crops and distance/temporal inspection |
| Final room galleries | 64 independent rooms per family, three 512×512 cameras: 1,728 views | Complete gallery, per-room/per-view metrics and no selection based on visual appeal |
| Camera sweep | 16 seeds per family × 1–12 cameras × baselines 0, 0.5, 1: 5,184 requested sets | Coverage, baseline, overlap proxies, admitted counts and duplicate/path metrics; paired rendered subsets check proxy accuracy |
| Time and interaction | Selected door/drawer/seat states, static contacts and optional motion at multiple times | Physical time, pose/instance alignment, contact, flow, co-visibility and collision consistency |
| Materials and lighting | Each new recipe under controlled illumination, grazing views, distance and exposure sweeps | Linear/color-space conventions, mip response, PBR channels, clipping and targeted Cycles comparisons |
| Portability/performance | Native, hardware Chrome WebGPU including M4/Metal, and same-device Burn routes | Full-quality delivery timings, numerical annotation checks and memory/responsiveness traces |
| Regression | Existing Cornell/object/indoor fixtures and archived settings | Preserved scene modes, API compatibility and explicit versioning of intended generator changes |

After fixing a family-specific failure, expand around that seed and the failing
parameter boundary before rerunning the whole campaign. Keep expensive image
and physical-reference tests targeted until structural correctness is stable.
Record randomized qualification seeds before implementation tuning; keep a
separate held-out set for final acceptance.

The diversity report should cover accepted distributions, not just available
controls:

- Room topology, adjacency, degree/loop counts, area/aspect, vertical levels,
  portals, service lengths, window area/orientation and circulation coverage.
- Required/accepted capacity, object count by class and subtype, assembly
  frequencies, support relations, yaw distributions, wall distance, clustering,
  empty regions and person/contact distributions.
- Material recipe/finish usage by object part, world texture scale, palette,
  roughness, metalness, glazing, exposure and illumination. Report conditional
  coverage so one family is not always tied to one material or time of day.
- Projected class/instance size, occlusion, depth layers, overlap and trajectory
  similarity across camera count, baseline and room family.
- Duplicate structural fingerprints, near-duplicate geometry and image
  neighbors, alongside SigLIP2 coverage and distances stratified by room family.
  Use a fixed embedding model/preprocessing and fixed comparison population;
  bootstrap at room level rather than treating views as independent rooms.
- Controlled factor sweeps showing that a sampled parameter actually changes
  visible geometry or PBR response. Include failure/admission rates so a rich
  proposal distribution cannot conceal repetitive accepted outputs.

A larger determinant or spread in an embedding projection can come from lighting,
color or artifacts. Interpret it alongside structural diversity, human visual
review and a held-out reconstruction experiment at fixed training budget. A
tenfold increase in raw parameter combinations is not evidence of tenfold useful
diversity or improved ten-million-sample training.

Use the Rust publication pipeline to produce category grids, room schematics,
material grids, camera-count heatmaps and current-source metrics from the same
capture receipt. The paper and project page should describe the instantaneous
qualified generator. Matched development comparisons remain local. A new
program should not be advertised as supported until its geometry, annotations,
UI/API and rendered evidence meet its acceptance gates.

**Implementation and acceptance status**

The nine named programs and compatible public suites are implemented as opt-in
activities. The current [implementation contract](procedural_public_spaces.md)
documents their typed parameters, required members, capacity units, circulation,
service links, support surfaces, semantics and capture exports. Native structural,
boundary, camera and tensor-parity campaigns have passed; the complete local
render gallery is generated from the same source identity.

The delivery table below remains an acceptance checklist, not a declaration that
all research/platform gates passed. General oriented partitions and reconstruction
region selection, stair-climbing motion, M4/Chrome runtime qualification and
broader physical-lighting agreement remain open. Targeted Cycles comparisons
found a substantial gallery-ceiling luminance mismatch. The established default
Mixed distribution is therefore retained; no default-mixture promotion or
photographic-equivalence claim is made.

**Delivery sequence**

| Phase | Complete deliverable | Exit condition and dependency |
| --- | --- | --- |
| 1 | Functional program/schema extension and one kitchenette with a dining group | Sink/counter/cabinet service relationships work end to end; native/WASM configuration and aligned capture pass |
| 2 | Kitchen, cafe and cafeteria programs | Jointed counters, appliance states, service/queue/return paths and dining placements pass; material/lighting additions qualified |
| 3 | Classrooms, galleries and showrooms | Audience fields, exhibit groups, purposeful contents and view/access checks work; contextual semantics and collection identity exported |
| 4 | Elevator landings and connected circulation | Shared wall/portal ownership, open-car geometry, approach regions and nonrectangular vestibules pass |
| 5 | Lecture halls and vertical occupancy | Named support regions, tiered rows, stair/aisle navigation, headroom and seated/moving people pass |
| 6 | Mixed functional suites and broader architecture | Compatible adjacent programs, region selection, oriented partitions and looped routes agree across all spatial consumers |
| 7 | Distribution, physical/reference and throughput qualification | Full campaigns, current-state report, matched GPU transport, browser checks and explicit unresolved limits |
| 8 | Default-mixture promotion | Evidence supports the accepted mixture; schema/generator versions, UI, CLI and docs agree |

Materials, annotations, humans and performance checks belong inside each
delivery, rather than being a cleanup phase after the catalog grows.
Phases 2–4 share substantial builders; each can ship independently once its
dependencies are ready. Broad vertical complexity follows the support model.

The first implementation slice should therefore be one parameterized counter
run with cabinet, sink, tap, optional refrigerator, supported coffee equipment
and a nearby table/chairs. Its acceptance review should include compact and
spacious rooms, several counter shapes, pushed-in seating, open cabinet states,
cable/service connections, material close-ups, three calibrated views and all
annotation modes. That slice tests the shared mechanisms needed by several
families before the catalog expands.
