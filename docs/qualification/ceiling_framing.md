# Exposed ceiling structure and service layers

Exposed ceilings have a recorded framing program in addition to service and
acoustic installations. Rafters, secondary members, roof ribs and bearing
connections provide depth beneath the roof. Their sections follow the roof
slope; the closed roof surface remains present for RGB, depth, visibility and
voxel annotations. Finished ceilings retain their acoustic, tile and coffer
programs.

## Parameterization

| Component | Sampled relationships |
|---|---|
| Primary framing | Timber, painted steel or metal; solid, I-section or open-web construction; both span directions; variable span, phase, section width and depth |
| Structural depth | Rafters target 0.28–0.75 m, subject to available headroom; secondary members target 0.08–0.18 m and roof ribs 0.025–0.065 m |
| Spacing | Primary pitch 1.55–3.1 m, secondary pitch 0.55–1.15 m, rib pitch 0.26–0.62 m; actual bay spacing fits each zone |
| Roof appearance | Wood, concrete or metal, with matching rib geometry and existing procedural PBR finish variants |
| Services | Round and rectangular ducts, conduit bundles, cable trays, sprinkler runs and ventilation terminals; independent run directions, lengths and vertical layers |
| Supports | Explicit body-to-structure rods; luminaires and services attach to the nearest actual framing surface or the roof |

Exposed construction is sampled in 85% of Industrial-style rooms and 32% of
other styles before placement constraints. This is a style-conditioned prior,
not a guarantee of admitted structural density. Small, low, partitioned or
mezzanine rooms can admit fewer members. A zone shares structural spacing and
materials deliberately: beams should form a coherent system, rather than
unrelated objects with independent noise on every dimension.

The structure is bounded to 160 members and installations to 96 modules per
room. Existing material-role batches are reused. Framing, supplementary services
and original ceiling layouts have separate seeded random streams. Detailed
plans, including sections, geometry, materials and suspension endpoints, are
serialized in the room manifest.

## Clearance and annotation contracts

Service bodies and thin hangers reserve their actual occupied regions
separately. This permits perpendicular runs to cross at different elevations
without reserving an entire empty column up to the roof. Body/frame and
body/hanger collisions remain rejected. Ventilation necks retain their full
vertical reservation. Open-web attachments use the actual triangles so a
hanger cannot terminate in a gap between chords.

Frames respect the room polygon, soffit breaks, partitions, luminaires and
upper-floor headroom. Later rugs, props, furniture and camera trajectories use
the same reservations. A bounded contact graph checks that every member connects
to a roof rib or bearing, with a 5 mm joint allowance. Local heights on the roof
plane prevent overlapping bounds from disguising a gap between sloped members.
Partial bays receive checked roof brackets; members without a valid connection
are omitted. This is a geometric connectivity check, not a load-bearing analysis.
Camera visibility acceleration includes the actual
frame triangles, including open truss webs, rather than treating every frame
bounding box as opaque. Rendered parts retain the `ceiling` semantic label.
This applies to room cameras as well as mounted cameras. Moving ceiling cameras
reserve their body, lens near field and optical-axis sweeps separately. A broad
lens must not enlarge the narrow mounting body into adjacent ribs. All three
components are checked over the complete trajectory; requested travel bounds,
calibration, mounting probability and overlap checks remain authoritative.

Absent optional `framing` and `suspension` fields preserve the previous manifest
construction and reservation paths. The framing contract is `ceiling=7`; the
local qualification report records its generator and complete capture identity.
Dataset metrics export frame roles, sections,
materials, dimensions, member counts and suspension counts.

## Qualification protocol

The ignored Rust test `exposed_ceiling_population_qualification` samples 256
consecutive Mixed room seeds, validates their layouts and builds every ceiling
mesh. Exposed rooms among the first 64 seeds also receive complete scene
geometry validation. It records admitted distributions, layer crossings,
manifest time and ceiling mesh construction time in
`out/ceiling_framing/report.json` and produces an eight-room underside gallery.

Regression tests cover deterministic replay, legacy fields, bounded mesh
geometry, finite normals and UVs, winding, triangle budgets, exact attachment
hits, different-height service crossings and malformed metadata. Complete
room and mixed ceiling/room camera tests exercise the shared placement rules.

The local 256-room qualification contains 113 rooms with framing and 3,111
connected members, covering all four roles and all four section types. Forty-one
rooms contain service runs crossing at separate elevations. Ceiling mesh
construction measured 0.29 ms median and 1.19 ms p95 on the shared native host;
these are descriptive CPU timings, not an isolated throughput comparison.
Full plans and per-room measurements are retained in the local report. The
associated qualification receipt records source hashes, test results and limits.

The gallery is a CPU flat-shaded view of actual mesh geometry with the roof slab
omitted for inspection. It does not qualify lighting, PBR appearance or GPU
performance. Construction timing is separate from rendering and concurrent
training throughput. These geometric distribution diagnostics do not establish
downstream generalization or a numerical effective dataset size.
