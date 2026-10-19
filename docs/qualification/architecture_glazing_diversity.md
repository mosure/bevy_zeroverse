# Architecture and glazing diversity qualification

Generator revision 53 uses continuous installation parameters for architectural
trim and glazing. Dimensions are in metres and are serialized in the scene
manifest. Sampling uses separate RNG streams, so adding a section parameter does
not advance furnishing, human, or camera random streams.

## Construction contract

- `FinishParameters.trim` controls optional base and crown sections: height,
  projection, shoulder, tip depth, profile, substrate, finish variant, construction
  joint, and corner clearance. Flat, chamfered, stepped, round and coved profiles
  have continuous dimensions. A room shares its section and finish; it does not
  acquire unrelated mouldings on every wall.
- Base trim stops at rough openings and the actual projecting sill envelope.
  Crown trim follows the roof and stops at openings. End cuts account for adjacent
  wall angles. Partition skirting fits the existing collision reservation and the
  available solid backing below glazing. Wall attachments clear the realized base
  and crown heights; fitted kitchen furniture retains its wall clearance.
- `Facade.joinery` controls mullion width/depth, unequal pane widths, sash
  width/depth, seals, applied frame beads, sill thickness/extension/material, and
  frame finish. `WindowOpening.transoms` supplements the existing `transom` field.
  `Facade::pane_rectangles()` exposes the same clear rectangles used by mesh
  construction. Subdivision validation rejects duplicate crossbars and unusable
  narrow lights.
- Exterior openings retain shared datums in some installations and have
  continuously varying sill/header levels in others. Aperture articulation only
  reduces a reserved opening, preserving piers, spandrels and roof clearance.
- `FinishParameters.interior_glazing` varies the neighboring-room partition's
  sill, header, solid piers, opening spacing, mullion pitch, crossbar and frame.
  Interior partition rails consume the same room-wide frame program.

The new optional fields have serde defaults. Recorded manifests without them use
their existing construction branches. Newly sampled seeds intentionally have a
new generator revision. Frame, sill, seal, trim and pane geometry retain their
wall/ceiling/window semantic ownership. Standard glazing annotation policy still
controls whether transmissive panes participate in geometric annotations.

## Verification and measurement

The bounded CPU study uses 64 consecutive Mixed rooms, three cameras, density
0.65, and human density 0.25. It validates room/layout constraints and complete
architecture, object and human meshes. A separate 1,024-seed parameter population
checks replay, physical bounds, optional sections and distribution coverage.
Regression tests check real aperture visibility, wall-complement area, moved
sashes, covering clearance, normals/winding, roof/floor support, coplanar faces,
and acute-corner trim clearance. Native Clippy and a WASM build cover both targets.

Reproduce the local CPU study with:

```sh
cargo test --locked --lib architecture_population_qualification -- \
  --ignored --nocapture --test-threads=1
```

It writes `out/architecture_diversity/report.json` and `parameters.json`.
Per-room timings separate manifest generation from architecture mesh construction.
The optional-detail timing control uses the same current room and apertures with
new joinery disabled; it is not a prior-release or end-to-end renderer benchmark.
Timings collected alongside training are a bounded performance screen.

Standard metrics exports now include base/crown presence and profile/material
counts; their height, projection and shape parameters; interior sill/header,
piers and frame sizes; and exterior mullion/sash/sill dimensions and pane/crossbar
counts. `architecture.jsonl` includes the finish programs. `exterior_windows.csv`
adds `crossbar_count` and semicolon-delimited `transom_fractions`.

GPU captures are deferred during active training. These checks qualify CPU
construction and metadata; they do not establish photographic realism, device
rendering performance or improved transfer to real reconstruction datasets.

## Local sample results

The 1,024-seed parameter population spans all five section profiles. Base trim is
absent in 84 samples (8.2%); crown trim is absent in 321 (31.3%). Dimensions below
are installation parameters; individual pieces are fitted to their backing and
clear pane rectangle. Sills are constructed only where the opening supports one.

| Parameter | 5th percentile | Median | 95th percentile |
|---|---:|---:|---:|
| Base height | 39.0 mm | 98.5 mm | 263.6 mm |
| Base projection | 9.8 mm | 23.3 mm | 36.6 mm |
| Crown height | 37.9 mm | 96.7 mm | 217.9 mm |
| Crown projection | 11.2 mm | 37.9 mm | 61.4 mm |
| Interior glazing sill | 30.9 mm | 313.6 mm | 1122.5 mm |
| Interior frame width | 26.1 mm | 49.6 mm | 70.2 mm |
| Requested exterior sash width | 13.6 mm | 28.1 mm | 43.7 mm |
| Requested exterior sill thickness | 16.8 mm | 40.0 mm | 61.9 mm |

The 64-room population contains 517 exterior rough openings: 234 without a
crossbar, 147 with one, 91 with two and 45 with three. All 64 full-room geometry
audits passed, including finite attributes, valid normals/winding, zero
degenerate triangles, and object/human support and envelope checks.

Architecture mesh construction took 15.80 ms median and 38.51 ms at the 95th
percentile during concurrent host activity. This excludes material preparation,
rendering and readback. The local study includes schematic SVG/PNG examples in
`out/architecture_diversity/construction-study.*`; these are not PBR captures.

The broad CPU suite passed 686 tests plus 14 isolated O-voxel tests. Following the
final trim-clearance adjustments, all 50 focused architecture/envelope tests and
the 64-room audit passed. Strict workspace/all-target Clippy and WASM compilation
with `web,human_motion` also passed.
