# Fixture construction, stock finishes and workstation arrangements

The object catalog is audited with `audit_objects`: metric mesh/envelope,
normal, degeneracy and coplanarity checks, plus six orthographic depth views
per construction sample. The descriptor removes translation and normalizes
the reserved envelope. It measures shape variation, not texture, real-world
scale, material appearance or downstream training utility. A low score for a
standard can, outlet or sheet of paper is not by itself grounds to deform it.

## Construction contracts

- Exit signs sample boxed and rail-mounted housings, case dimensions, bevel,
  rail proportions and readable legend proportions. They still denote the
  actual entry; arrows are not sampled independently of a destination.
- Alarm, detector and manual-call-point construction separates the mounting
  envelope from the manufactured case. Case dimensions, detector profiles,
  inlet positions and control proportions vary continuously. The wall rear
  face or ceiling mounting face remains fixed. Controls follow the sampled
  body, and existing semantic classes are retained.
- Pinboards sample backing thickness, frame depth/profile/material and inset
  face depth. Frame, divider and attached notes use the same physical section.
  Notes retain rotated containment, mutual clearance and exposed backing.
- Workstations sample monitor gaps, depth stagger, relative widths, heading,
  laptop depth and docking handedness. Dimensions and transforms are proposed
  before the existing support, peer, chair-facing and architecture checks.
  No post-admission jitter or clearance relaxation is used. Their requested
  parameters are recorded alongside the existing accepted object geometry in
  dataset metrics; a proposal does not imply that it was accepted.

Programs are available through `objects::audit::program`,
`objects::equipment::board_section::parameters`,
`objects::equipment::signals::{exit::parameters, fire::program}` and the
existing workstation metrics. Ceiling-detector studio captures look at the
visible underside rather than the concealed mounting face.

## Shared PBR stock

`materials::variants::mapping::parameters(scene_seed, surface, slot)` returns
the physical texture-period multiplier and repeat-space phase used by the
material builder. Every selected finish uses an independent cut from its
shared structure texture, without allocating additional maps or changing the
bounded finish/structure pools.

| Substrate | Texture-period multiplier |
| --- | --- |
| Wood / wood edges | 0.65–1.65 |
| Fabric / alternate fabric | 0.70–1.50 |
| Leather | 0.72–1.40 |
| Ceramic, metal, chrome, plastic | 0.70–1.60 |

Period multipliers are sampled uniformly in log space; each phase axis spans
one repeat. The same transform applies to albedo, tangent normals, roughness,
metalness and occlusion. Scaling is positive and isotropic, with no tangent
rotation, reflection or shear. Related wood and cloth faces share a mapping;
coordinated product finishes remain coordinated. Leaf anatomy, displayed
documents and printed images keep their semantic UV coordinates.

This operates on `StandardMaterial` uniforms in both native and WASM builds.
It does not alter GPU bake recipes, map encodings, mip construction, semantic
labels or mesh topology. The source/capture identity changes because the
sampled appearance and geometry differ.

## Qualification and limits

Focused tests check replay, actual mesh variation, preserved mounting planes,
physical envelopes, visible pinboard contents, workstation support/facing,
texture reuse and PBR map/mipmap contracts. Bounded population audits cover
all 107 categories and complete generated rooms, including dense eight-view
scenes. Current local receipts and inspection captures are under
`out/diversity_review_100/`; matched evidence is not published to the project
page or paper.

Keep canonical shape coverage separate from material variation and placement
coverage. Do not interpret the number of descriptor representatives as a
count of independent training samples. These checks do not establish
photographic realism, prevent model memorization, or qualify concurrent
training throughput or hardware-specific rendering.
