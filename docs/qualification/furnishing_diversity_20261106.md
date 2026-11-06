# Furnishing construction and placement diversity

This qualification covers the six requested construction bottlenecks. All geometry
uses the same builders, semantic ownership, placement envelopes and deterministic
seeds as capture; no preview-only replacements are used.

| Category | Sampled relationships | Physical contract |
| --- | --- | --- |
| Rugs | Independent group borders, offset and yaw; small mats through long runners and large area rugs | Exact contour support, consistent furniture lifting, door/circulation clearance and reserved ceiling headroom |
| Waste containers | Independent height/width/depth, tapered round-to-rectangular bodies, pedal/divided/ribbed forms; optional coordinated trash/recycling stations with distinct real apertures | Sample dimensions before admission; at most two peer positions for each of at most three existing bins; bounded hinge geometry and validated partner metadata |
| Sofas/loveseats | Broader dimensions, asymmetric seat widths, independently divided backs, channel pitch, cushion thickness, stepped crest, single/double chaise returns, lumbar-to-square loose pillows | Shared support dimensions for mesh and pillows; fixed chair seating datum; finite back-channel budget; pillows do not intersect each other or the back |
| Mobile whiteboards | Board width/height/depth, inset support span, post splay, foot depth and stretcher height | Load-bearing posts reach the panel; lower stretcher follows the posts and remains below the writing face |
| Ring lights | Broad luminous bands through thin hoops, continuous room/fixture variation | Identical realized aperture in mesh and emitting-area calculations; no added light sources or render passes |
| Office chairs | Seat/back proportions, split-panel ratios and gaps, arm-pad dimensions, headrest width/height/gap and support spacing | Human seat datum stays 0.47 m; headrest reserve follows actual dimensions; geometry remains inside table-clearance envelopes |

`out/furnishing_diversity_20261106/` contains local receipts and CPU flat-shaded
mesh galleries. These are construction previews, not photographic or PBR evidence.
The opt-in `furnishing_diversity_qualification` test produces the gallery, object
programs, measured build time, realized room placements and errors. The focused run passes 37 tests, and all 92 production-seed regressions pass.
Normal tests check mesh bounds, normals/indices, deterministic replay, aperture openness and
non-degenerate parameter coverage. Existing contact and emitted-area tests remain
in force.

The local catalog qualifies **384 meshes** across six rows, including both ordinary
bins and waste-station bins. All pass metric-envelope and mesh validation. The
room study qualifies **128 rooms** with three cameras and human density 0.35;
**32 rooms** additionally pass complete mesh validation. It reports **21 accepted
waste pairs in 19 rooms** and **75 rugs**, with accepted rug widths of about
**0.60–3.75 m**. The fixed 256-product sampler has seven decile bins of whiteboard
support span and 187 sofas whose back count differs from their seat count.

The 64-case office-chair catalog contains 16 continuous mesh backs, 29 continuous
upholstered shells, and 19 split/segmented constructions. Maximum mesh triangle
counts are 25,080 for the sofa sample and below 10,000 for the office-chair sample.
The ring audit covers 411 fixtures: the inner/outer diameter ratio's 5th/95th
percentiles are **0.306 / 0.851**; projected luminous mesh area remains consistent
with photometric area. Rings retain the existing facet count and render path.

The geometry gallery was inspected; the mobile whiteboard carrier now sits below
the panel, so its posts do not show through the writing faces. Gallery cells
normalize framing and should not be used to infer relative physical sizes; use
the exported metric dimensions.

The rejected-room neighbor sweep also exercises seed **69430991**. Its short
chair arm exposed a crest whose bend radius was smaller than the tube radius.
Arm carriers now use explicit rounded bends; an additional 54 extreme-parameter
combinations check both carrier constructions without relaxing triangle checks.
Tall/slim-bin hinges remain inside the product envelope, and waste-pair targets
are validated explicitly rather than admitted as arbitrary interaction links.

A separate matched 128-room run (eight room profiles, three views, density 0.65,
human density 0.35) has zero failures. Current CPU manifest-construction time is
57.9 ms mean, 28.1 ms median and 208.9 ms p95; the maximum is 508.8 ms.
These are shared-host diagnostic timings with active training, excluding mesh
construction, rendering and readback. The local matched timing receipt reports
no preparation-time regression. Workspace/all-target Clippy with CUDA IPC/WGPU
fusion features and the web/human-motion WASM library check also pass.

No hardware GPU, concurrent-training throughput, or training-utility claim is made
from these CPU tests. Wider input ranges alone are not an effective dataset-size
estimate; optional objects and unsuccessful placements must be counted from the
accepted room population.
