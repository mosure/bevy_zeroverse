# Architecture, window sections and trim qualification

Window sections and architectural trim are sampled in metric units and recorded
in each scene manifest. Separate random streams control wall sections, opening
returns, trim and heating fixtures. A room shares its exterior wall thickness and
skirting installation; individual openings retain their own frame and finish.

## Construction contract

- `EnvelopeProgram.wall_thickness_m` samples 0.15–0.72 m with a log-uniform scale.
  Thickness grows outward from the usable room boundary. The shared doorway wall
  retains its existing construction. Window apertures reserve thicker corner
  caps; short edges remain solid when they cannot fit a useful opening.
- `Facade.joinery.reveal` records the wall section, lining thickness/substrate,
  finish variant and optional profiled surround. Frame setback varies within the
  wall, leaving real returns on both sides. Paint, timber, concrete, tile and
  polymer linings are geometry, with 4–14 mm thickness. Surrounds have flat,
  chamfered, stepped, round or coved sections. They remain inside the rough
  aperture and outside the clear pane rectangles.
- Window frame, sash, mullion, bead, seal and sill construction uses the recorded
  opening. Sills span the actual wall depth. Unequal panes, multiple transoms,
  different sill/header levels and independent window treatments remain active.
  `Facade::pane_rectangles()` exposes the same clear rectangles used by meshes.
- `FinishParameters.trim` controls optional base/crown height, projection,
  shoulder, tip depth, profile, substrate, finish, construction joint and corner
  clearance. Base heights span 25–420 mm and projections 6–70 mm. Sampling spans
  slim commercial strips, taller skirting and deeper plinths. Crown dimensions
  remain within the clearance reserved by tall furnishings.
- Base trim stops at apertures, projecting sills and fitted cabinetry instead of
  running through them. Corner cuts account for adjoining wall angles. Partition
  trim remains limited by its actual backing and collision reservation. Wall
  attachments use the realized trim heights.
- Optional `BaseboardHeater` objects add low convector installations: 0.55–3.20 m
  width, 0.13–0.46 m height and 75–190 mm depth. Covers have real inlet/outlet gaps,
  varied vent pitch/angle, end caps, brackets, controls and optional hydronic
  connections. Installation frequency increases smoothly with wall thickness.
  The sampler makes bounded placement attempts after furnishings and before
  humans, cameras and motion. Units use the regular solid-object clearance gates.

Absent optional wall/reveal fields retain the legacy 180 mm wall construction;
old recorded manifests do not acquire heaters. New samples have a new generator
and capture identity. Frames/trim/returns retain their window/wall semantics;
heaters export an explicit object kind, per-instance bounds and the `appliance`
semantic class. Primary-room shell bounds include the thicker wall section but
exclude exterior backdrop geometry. Glass annotation policy remains independent
of these solid opening returns.

## Reproducible checks

Focused tests exercise seeded replay and JSON round trips, finite/wound meshes,
clear pane centres, actual ray visibility through deep openings, invalid section
rejection, fitted cabinet joints, heater bounds/support/collisions and semantic
ownership. Existing architecture checks cover wall-complement area, moving
sashes, treatments, roof/floor support, coplanar surfaces and acute-corner trim.

```sh
cargo test --locked --lib architecture_population_qualification -- \
  --ignored --nocapture --test-threads=1
```

The bounded CPU study audits 64 consecutive Mixed rooms plus 13 regression seeds,
with three cameras, density 0.65 and human density 0.25. It validates layout and
complete architecture, furnishing and human meshes. A 1,024-seed parameter study
records wall/reveal, skirting/crown and interior glazing dimensions. Artifacts are
written to `out/architecture_diversity/report.json` and `parameters.json`.
Per-room timings separate manifest generation and architecture mesh construction;
they exclude material baking, rendering and readback.

The standard metrics exports include exterior wall thickness, return depth,
lining material/thickness, surround profile/dimensions, skirting/crown profiles,
heights/projections and opening subdivisions. The object metrics include heater
counts, dimensions, placement and construction parameters.

For controlled PBR geometry cutaways using the actual builders/materials:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json LP_NUM_THREADS=2 \
  cargo run --example review_window_sections
```

This writes three images and their exact fixture parameters to
`out/window_trim/renders`. These fixtures isolate thin/medium/deep window sections,
skirting and convectors; they are not a random-room distribution or a physical
light-transport benchmark. Software Vulkan checks do not qualify hardware CUDA
throughput, browser GPU performance or photographic realism.

## Local qualification results

The 1,024 parameter draws span all five surround/trim profiles and all five
lining substrates. Percentiles use sorted empirical samples without extrapolation.
The return-depth row uses an 80 mm reference frame, isolating the wall/setback
sampler from frame-depth variation.

| Parameter | 5th percentile | Median | 95th percentile |
|---|---:|---:|---:|
| Exterior wall thickness | 163 mm | 325 mm | 652 mm |
| Room-side frame return | 26 mm | 109 mm | 374 mm |
| Base trim height | 29 mm | 99 mm | 370 mm |
| Base trim projection | 7 mm | 21 mm | 62 mm |

All 64 consecutive rooms and all 13 regression seeds pass layout and complete
mesh validation. The consecutive population includes 29 heating units across 15
rooms. Heaters remain optional, with accepted placements subject to available
solid wall/floor space. The separate 128-room placement test checks support,
collision rejection and semantic ownership.

In this two-worker CPU screen, architecture mesh construction has a 12.17 ms
median and 21.96 ms 95th percentile. Manifest generation, including three-camera
placement, has a 40.06 ms median and 189.78 ms 95th percentile. These are current
local component timings, not an end-to-end render benchmark or a measured
hardware throughput improvement.
