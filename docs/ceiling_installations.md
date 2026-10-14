# Procedural ceiling installations

Ceiling installations are recorded in `IndoorManifest.ceiling_installation`.
Each zone samples a coherent installation rhythm, finish, plenum gap and service
orientation. Modules adapt to the actual roof and floor heights, including
sloped ceilings, raised floors and soffits.

| System | Geometry and variation |
| --- | --- |
| Suspended tiles | Metric grid pitch and tile aspect, recessed tegular faces, T runners, perimeter fascia and suspension |
| Open-cell ceiling | Rectangular cells, variable pitch, aspect and blade depth; the concrete slab remains visible |
| Acoustic clouds | Tapered curved undersides, carrying rails, width and suspension height |
| Baffles / timber slats | Independently phased lower profiles, taper, spacing and depth with a common supported top datum |
| Round / rectangular duct | Smooth lateral and vertical offsets, reducing sections, seam collars, takeoffs, elbows, registers and trapeze hangers |
| Sprinklers | Main and branch pipe sizes, branch reach, spacing, upright/pendent heads, frame, thermal bulb and deflector |
| Conduit | Parallel runs, swept offsets, elevation changes, junction boxes, couplers and supports |
| Cable tray | Sloping ladder tray, side rails, spaced rungs, individual supported cables and hangers |
| Trusses | Warren, Pratt or spatial webs, tapered depth, chord diameter, cell pitch, cross bracing and end plates |

Continuous shape fields live in each module's optional `detail` program; the
module seed determines repeated details. Old modules lacking this field retain
their earlier construction. Capture provenance identifies this geometry as
`ceiling=2`.

All opaque installation geometry uses the `ceiling` semantic class. Hardware,
fabric, wood and metal have separate existing PBR roles and bounded finish slots.
The geometry enters the same RGB, depth, normal, position, flow, co-visibility and
primary-room voxel paths as the rest of the architecture. It uses the existing material pools, with no installation-specific textures,
extra compute pass, physics simulation or per-frame geometry work.

Sampling caches fixture and structural bounds once per room, limits retries,
and caps installations at 96 modules. Body and suspension reserves exclude
walls, window coverings, columns, partitions, fixtures and other installations. Minimum reserved
headroom is 2.35 m, including any mezzanine stair treads inside a module footprint. Shallow rooms compress the plenum and body depth; short duct
takeoffs use directly mounted registers when there is no room for a drop elbow.
Service runs are bounded local assemblies, not a whole-building MEP simulation.

Population export records system counts, body depth, plenum gap, headroom,
length/width, grid dimensions, duct offset/reduction/branches, sprinkler pitch
and orientation, conduit parameters and truss webs. These distributions describe
actual sampled programs; they are not an estimate of independent scene entropy.

## Local qualification

```sh
cargo test --lib architecture::ceiling
cargo run --example review_ceilings -- out/ceilings/details
cargo run --bin indoor_validate -- --seed 0 --audit-seeds 256 --audit-geometry \
  --renders 8 --cameras 3 --width 512 --height 512 --labels --co-visibility \
  --no-raw --output out/ceilings/rooms
```

The isolated review selects two generated modules of each system, without
rescaling their geometry, in fixed inspection lighting. It saves RGB, normal,
depth and semantic images, module programs, assembly construction timings and
capture provenance. Use the complete room captures to judge placement and
production lighting; isolated construction timings are not generation throughput.
