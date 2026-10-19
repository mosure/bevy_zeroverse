# Furniture and botany construction qualification

Plants, chairs and tables use seeded construction programs. The object ID,
semantic class, placement reserve and furniture relationships remain the same
contracts used by capture and annotations. Construction introduces no models,
external assets, GPU initialization or per-object texture atlases.

## Realized variation

Plants include 17 growth habits, with bird-of-paradise paddle leaves, palmate
fan palms and ZZ-style paired leaflets alongside the existing trees, shrubs,
ferns, vines, palms, succulents and cacti. Crown spread, leader count, node
spacing, shoot phase, angular irregularity, leaf-tip shape, curling and lobing
vary continuously within botanical constraints. Leaf attachments follow the
emitted stem segments. Planters vary their neck, belly, rounded base, shoulder,
faceting, fluting and saucer proportions. Plant selection is constrained by the
available height and canopy width, for both floor and supported placements.

Chairs combine tension mesh, spindles, shaped shells, two/three-part upholstery
backs and horizontal slats with independently chosen seat, back and frame
substrates. Shells have continuously varied waist and crest contours, with
normal vectors transformed along with their surfaces. Separate back panels
are attached to continuous structural rails. Stools have rounded/rectangular
or elliptical seats, three/four splayed legs, pedestal bases or tubular sled
frames. Footrests connect to the actual supports. Low-back and backless stools
retain the 0.47 m seating datum used by static people and furniture placement.
Swivel bases sample both casters and grounded glides; conference-style bases
favor glides while task-chair bases favor casters.

Tables combine superellipse, capsule and rounded polygon support functions.
Polygon side count, phase, rounding and blend vary independently of top size,
thickness, edge treatment and finish. All interpolated contours remain convex;
clutter and seated-person clearance use this same contour. Ten support
constructions include hairpin frames and waisted drum pedestals. Rendering and
collision emit from one shared structural program. Mounts are fitted to the
actual top, rather than its rectangular envelope.

The diagnostic object-program API records these sampled controls. Dataset
metrics include plant crown/leaf/planter values, polygon controls, pedestal
profiles, shaped chair backs and stool bases. Sampled controls are not assumed
to be statistically independent, and dormant conditional controls must not be
multiplied together to infer dataset capacity.

## Local verification

The ignored Rust test `plant_chair_table_population_qualification` writes its
report and CPU contact sheets to `out/furniture_diversity/`. It checks 688 object
meshes (17 plant habits and ten chair families with 16 seeds each, and four
table categories with 64 seeds each), deterministic replay, semantic classes,
metric bounds, finite normals, winding and triangle budgets. It also constructs
64 consecutive mixed-room seeds with three cameras and people, and validates
complete geometry and placement. Ordinary tests check geometric clearance,
shared seating height, tabletop convexity and actual supported placements.

Within-group shape comparisons use six 24-by-24 orthographic first-surface depth
maps. The fixed object envelope and finish seed remove scale and color as
explanations for variation. Ordered greedy coverage is recorded at explicitly
defined distances; neither mesh hashes nor this geometric descriptor estimate
SigLIP volume, photographic realism or downstream learning capacity.

Contact sheets are flat-shaded CPU geometry previews. CPU construction timings
are not rendered throughput. GPU appearance, browser runtime and concurrent
training throughput require a separate qualification window.
