# Wall presentation equipment qualification

Wall-mounted products retain physical alignment with their support. Diversity
comes from product construction, dimensions, finishes and supported contents;
arbitrary wall-normal rotation would create detached or embedded fixtures.

## Construction and placement contracts

- Whiteboards sample case thickness, border, corner rounding, mounting caps,
  tray length, lateral offset, depth, retaining lips and end caps. Trays meet the
  board rather than floating in front of it. Marker bodies, caps and erasers
  have independent dimensions and bounded support-local positions/yaw. Actual
  rotated footprints must fit and remain separated. Occupancy includes empty
  trays; magnets are separated and fitted inside the writing face.
- Wall shelves sample board thickness, bracket span, lateral offset, support
  count, reach, strap dimensions and curved-brace shape. Concealed, angle-strap,
  rod-braced and curved-brace constructions retain a full horizontal support
  top. Visible braces terminate inside their wall plate and load strap. Layout
  samples widths 0.42–1.70 m, heights 0.13–0.38 m and depths 0.17–0.40 m before
  the ordinary architectural and object-clearance checks.
- Framed and retractable projection screens sample housing/caps, mounting
  positions, sheet width/offset, border proportions and drop. Screen dimensions
  span 1.00–2.70 m width, 0.70–1.70 m height and 0.050–0.12 m reserved depth.
  A shared construction program supplies the actual white-sheet rectangle to
  projector placement and throw-ratio/architecture checks. The equipment is
  powered off; these checks do not simulate a projected image or beam.
- Substrates use independent finish streams. Geometry and optical targets stay
  unchanged when only the finish seed changes. Timber, polymer and metal
  components reuse the bounded PBR material system and semantic classes.

Construction modules are `objects/presentation/{board,contents,projection}.rs`
and `objects/equipment/wall_shelf.rs`. Placement and metrics consume these
programs rather than approximating the visible product independently. Normal
generation performs bounded local proposals; audit collection and mesh-distance
measurements only run when explicitly requested.

## Qualification protocol

Routine tests exercise independent dimension extremes, finite attributes and
winding, semantic vocabulary, exact wall attachment datums, accessory support
and separation, full shelf-top ray hits, first-surface rays into actual screen
triangles, deterministic replay and finish/geometry independence. Coplanar-face,
wall-attachment, projector-clearance, botanical support, complete-room geometry,
motion and metric-export regression checks complement those local invariants.

The ignored `presentation_equipment_population` study builds 64 fixed-envelope,
fixed-finish samples of each category. Six 24×24 first-surface depth maps and
union-pixel normalized L1 distance measure shape coverage at thresholds 0.005,
0.015 and 0.03. These measurements exclude color, UV shifts and triangle ordering;
low coverage can be appropriate for a flat panel and is not an effective dataset
size estimate.

The study also validates 80 complete rooms, using distinct seeds 100000–100079
across all 20 concrete profiles, density 0.65, three cameras and human density
0.15. It archives accepted non-neighbor objects, their dimensions/transforms and
construction programs. These observations include secondary interior spaces and
are not visibility-weighted image statistics.

Outputs stay local under `out/presentation_equipment`: the mesh gallery and tray
close-ups, per-sample JSON, timing/distribution summaries and a source-hashed
qualification receipt. Previews use flat CPU shading of actual triangles. CPU
preparation timings exclude material baking, rendering, readback and concurrent
training. They do not qualify PBR appearance, hardware/browser runtime behavior,
embedding-space volume or improved downstream generalization.

## Recorded local results

Generator 92 / capture-v117 passed 62 distinct native tests, strict workspace
Clippy with CUDA IPC/WGPU fusion and SigLIP CLI/pipeline enabled, formatting, and
the WASM viewer compilation with `web,human_motion`. Routine product checks
include 1,152 meshes at independent metric extremes; the local study adds 192
fixed-envelope products and all 80 complete-room geometry/layout checks.

| Category, 64 seeds | Shape representatives at 0.005 / 0.015 / 0.03 | CPU build median / p95 | Triangle median / maximum |
|---|---:|---:|---:|
| Whiteboard | 61 / 55 / 55 | 0.047 / 0.081 ms | 1,222 / 2,758 |
| Wall shelf | 55 / 51 / 51 | 0.015 / 0.021 ms | 540 / 756 |
| Projection screen | 54 / 50 / 49 | 0.021 / 0.025 ms | 730 / 730 |

Accepted products comprise 90 whiteboards, 58 shelves and 28 projection screens.
All four shelf support constructions occur (18 rod-braced, 12 angle-strap,
16 curved-brace and 12 hidden), with board thicknesses spanning 0.015–0.062 m.
The screen sample includes 13 framed and 15 retractable products; actual white
image widths span 0.781–2.534 m. These counts establish bounded coverage, not
global distribution probabilities.

CPU room preparation, including three-camera placement, measured 25.7 ms median,
193.5 ms p95 and 391.5 ms maximum on the shared machine. No matched throughput
speedup is claimed. The build retains the upstream Burn fusion 0.22.1
future-incompatibility advisory.

The coplanar-face regression caught intersecting screen/bar end faces during
development. The weight bar now extends beyond the sheet within the existing
placement reservation, and the hanging sheet enters its housing without a gap.
The final face-overlap and first-surface ray tests pass. Source/test-binary
hashes, per-sample observations and exact commands are retained in the local
qualification receipt.
