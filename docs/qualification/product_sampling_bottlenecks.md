# Product geometry, finishes and relational placement

The catalog audit covers all 107 object categories. Six orthographic depth maps
per object measure construction differences at a fixed metric envelope. Shape
coverage is considered alongside actual room placements, material bindings and
rendered appearance. Thin sheets, standardized sockets and manufactured products
retain their functional constraints; low depth variation alone is not a reason
to deform them.

## Construction programs

| Category | Sampled construction and appearance | Physical contract |
| --- | --- | --- |
| Beverage cans | Heel, waist, shoulder, neck, height, lid orientation, opened aperture, label coverage and bare-metal cases; broader metric diameters and aspect ratios | Label follows the body profile with a separated surface and one complete UV wrap; lid parts remain inside the envelope |
| Sponges | Footprint, rounded section, waist, height, foam/pad proportions, optional scouring layer and product palette | Flat support contact, fitted pad, rough nonmetallic materials |
| Cutting boards | Body proportions, rounded profile, bevel, offset handle, optional rubber feet and wood/plastic finish | Bread proposals use the actual board body, excluding the handle and rounded corners; feet and handle remain attached |
| Outlets and switches | Plate width/height/depth, gang count, case proportions, control shape/size, USB inserts, finish, screws and switch state | Rear mounting plane stays fixed; controls fit both dimensions; cable endpoints follow the actual first receptacle |
| Framed artwork | Asymmetric mat margins, image bias and frame sections across all six existing frame constructions | Complete normalized picture UVs, positive depth separation and image containment within the rails |

Construction parameters are available through `objects::audit::program` and
their category modules. Dataset content metrics include the effective can,
sponge, board, frame and electrical parameters. Conditional choices such as a
bare can or an unmatted picture do not turn inactive parameters into additional
visible diversity.

## Placement and installation

`layout::electrical_parameters` exposes the seeded room installation program.
Socket convention is coherent within the room. Nominal outlet heights span
0.19–0.62 m, switch heights 0.96–1.43 m, plate heights 0.095–0.137 m and plate
depths 0.016–0.026 m. Instance tolerances and floor-relative placement vary the
admitted transforms.

Worktop outlets are proposed from nearby desks/counters/cabinets projected into
the real wall frame. Placement verifies proximity to the oriented support,
openings, mounting faces, columns and existing objects. This relation is
computed before admission, without post-placement jitter. Ordinary low outlets
remain part of the distribution.

Kitchen runs first try their original sampled dimensions. If no run fits, two
bounded compact preparation-cabinet proposals preserve the required sink and
range dimensions. Cabinet width never drops below 0.48 m; door travel, floor
support and all clearance checks remain active. Optional objects do not replace
the required kitchen program.

## Materials and cost

Sponge foam and scouring layers use eight bounded product palettes, with seeded
hue, saturation, lightness and roughness variation. Bindings reuse the requested
normal/occlusion structure maps. Furniture albedo and glossy plastic roughness
maps are excluded from these rough dielectric finishes. Only requested slots
are created during normal generation; this adds no texture bake or render pass.

Wall installation geometry is modularized under `objects::wall::fixtures`, with
placement in `layout::decor::electrical`. Can and sponge constructors have their
own modules. Construction, placement, cable connection and metrics call the
same seeded programs.

The standalone `IndoorMaterials::build_for_parts` API also completes its
scene-derived reflection cubemaps before returning. Internal staged preparation
continues to reuse its own geometry/transport build. The object-review tool uses
the selected-material API so qualification exercises consumed-map selection.

## Qualification

Local receipts, exact build provenance, comparative descriptor statistics and
RGB/normal/semantic inspection galleries are retained in
`out/diversity_review_102/`. This is a local study, not published site evidence.
The tests cover mesh bounds, normals, degeneracy, coplanarity, deterministic
replay, full-image UVs, actual board-top triangle support, outlet connections,
palette binding and admitted wall/room placement.

The qualified local source is generator 102 / capture-v127,
`2e185a165ae81c90a3ac33fbbf59259d3f5688b87fa85dadaf7dc300617303fa`.
The completed checks are:

- 69 focused tests; strict workspace Clippy including CUDA IPC/WGPU fusion;
  formatting and the WASM viewer with human motion.
- 6,848 mesh audits: 64 construction samples for every category, with no failures.
- 208 complete-room geometry audits: 128 mixed rooms, 16 dense eight-camera
  rooms, 32 kitchens and 32 kitchenettes, with no failures. Seed 127 exercises
  compact-cabinet recovery without resizing its sink or range.
- A separate 96-room electrical-placement sweep: all three socket conventions,
  69 admitted outlet-height bins at centimetre resolution, plate depths across
  16–26 mm, and 33 outlets above real nearby worktops.
- 108 software-Vulkan captures: six representatives of each revised category
  in RGB, normal and semantic modes, at 320 square pixels with automatic SSAA.

Current ordered-greedy depth coverage from the 64 samples is:

| Category | Representatives at 0.005 | Representatives at 0.015 |
| --- | ---: | ---: |
| Soda can | 64 | 58 |
| Sponge | 64 | 64 |
| Wall outlet | 64 | 60 |
| Light switch | 64 | 61 |
| Cutting board | 64 | 61 |
| Wall art | 22 | 6 |

The fixed catalog sweep constructs 19,290,463 triangles in 30,742 material
batches. CPU construction alone took about 0.65 seconds; this excludes descriptor
measurement, room preparation, rendering and readback. Comparative local
receipts record a 0.01% triangle increase, slightly fewer batches and approximately
the same CPU-construction time. This is a bounded diagnostic, not a hardware
generation-throughput claim.

The review now uses consumed-map selection. For the 30 views with unchanged
geometry, normal and semantic images exactly match the preceding full-material
studio run. RGB is not bitwise identical: its maximum image RMSE is 6.418/255
because unrequested room textures no longer contribute to the studio's
scene-derived environment. These review-tool observations are separate from
production renderer performance and image replay.

Descriptor representatives are threshold-dependent shape coverage, not an
effective dataset size or a proof that a model cannot memorize the generator.
Artwork framing changes can be visible in RGB while remaining similar in depth.
Software Vulkan renders qualify the exercised material/annotation paths; they
do not measure hardware-specific viewer behavior, concurrent-training
throughput, photographic realism or downstream generalization.
