# Functional indoor spaces

The indoor generator has explicit programs for elevator landings, cafes,
cafeterias, classrooms, lecture halls, showrooms and galleries, alongside the
kitchen and kitchenette programs. `PublicSuite` samples compatible adjacent
functions. These programs use the existing `ProceduralIndoor` scene and capture
pipeline. Cornell, object, primitive and semantic-room scenes remain available.

```sh
cargo run --bin viewer -- --scene-type procedural-indoor \
  --indoor-layout lecture-hall --indoor-seed 10000
cargo run --bin indoor_validate -- --layout gallery --seed 20000 \
  --audit-seeds 1024 --audit-geometry --renders 64 --cameras 3 \
  --width 512 --height 512 --labels --co-visibility --no-raw \
  --output out/gallery_qualification
```

The native/WebGPU controller's activity selector uses the same enum as CLI,
URL and Rust configuration. URL spelling follows the existing enum parser,
for example `indoor_layout=lecture-hall`. Changes use the existing explicit
Apply/Next controls. The established office `Mixed` distribution remains
unchanged until the default-mixture qualification gate is satisfied.

## Construction and use

| Program | Construction and placement |
| --- | --- |
| Elevator landing | Telescoping door banks, recessed jambs/thresholds, call controls, indicators, bounded illuminated car cavities, handrails, waiting benches and information stands |
| Cafe | Modular service counter, glazed display case with real shelves/contents, coffee equipment and beverage dispensers, small tables, banquettes and contextual dishware |
| Cafeteria | Serving counters, tray-return shelving, beverage service, larger dining tables, reserved serving queue and through circulation |
| Classroom | Teaching board, lectern, lockers, individual writing surfaces, tablet-arm seating and continuously varied seating fields |
| Lecture hall | Curved audience rows, linked-frame folding seats, explicit floor tiers, intermediate stair treads, teaching platform and guards at exposed drops |
| Showroom | Product plinths, information stands and vitrines; coordinated content collections and shared finish identity |
| Gallery | Angled exhibition panels with framed artwork, sculptural plinths and vitrines; separate viewing/circulation space and exhibit lighting |
| Public suite | Compatible function pairs with sampled split direction/position, open shared boundaries and one authoritative primary camera region |

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
reporting triangulation.

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
