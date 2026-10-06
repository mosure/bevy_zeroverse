# One-time GPU material bake: design proposal, not implemented

This file is an implementation sketch. No shader or renderer path described here
has been added or qualified. Current production materials are still synthesized
on the CPU; their map bytes, mips, samplers and recipe distribution are preserved.

## What the existing exact CPU work establishes

`PreparedMaterial` is private and owned by a single map evaluation. Paint/wood
hoist invariant colors, frequencies and footprint terms. Paint spray attributes
and periodic noise share bounds of 589,824 and 262,144 bytes respectively, at
most **851,968 bytes/map**; over-budget lattices use scalar evaluation.

Mineral/concrete now cache aggregate attributes/pigments, casting voids, board
tones, periodic fields and deposits within **one shared 1MiB logical heap-table
budget per map**. Every omitted cache uses the scalar path. Dynamic feature
positions retain their original addition order, traversal and strict nearest
comparison. Variable sRGB round trips and interpolation remain unchanged.
Color-mip memoization adds **32KiB/map**, keys the full float bit pattern and
replaces colliding slots without approximation. No context is serialized,
globally retained, or keyed in a persistent per-seed cache. Native and WASM
share these material programs.

The existing independent consumed-map audit passes again: 89 triplets /
267 complete images, with its retained scope receipt recording 105,905,820
bytes. A separate strengthened original mineral/casting oracle compares
36 images / 31,457,232 complete mip bytes and 11,796,480 float channels.
Zero/tiny/full-budget, seam/wrapped-coordinate, extreme-recipe, color-transfer
collision and joint normal/roughness filtering tests pass.

The current exact CPU path also uses a map-owned woven-textile cache with one
shared **64KiB logical table budget** for yarn attributes and periodic fields.
Original-expression oracles compare all texel channels and complete color,
normal and ORM mips, including scalar fallbacks, authored extremes, wrapped
coordinates, absent recipes, carpet pile and unchanged knit dispatch.

The current CPU implementation also overlaps geometry-only transport with map
synthesis, skips provably unused mineral color endpoints and reuses a fixed
native HDR filtering stencil. Its exact render fence can shorten capture warmup
only after the requested scene, cameras and assets have rendered and passed the
existing readiness barrier. Unsupported history effects retain conservative
settling. None of these changes implements GPU texture baking.

See the [current capture qualification](generation_cpu_efficiency.md) and
[qualification receipt](evidence/cpu_efficiency/receipt.json) for matched capture
measurements, exact map/render replay, tests, source identities and rejected
experiments. Browser execution and GPU texture baking remain unqualified.

Retained counterbalanced CPU kernel diagnostics:

| Kernel | 256 reference / cached (ms) | 512 reference / cached (ms) | Timing boundary |
|---|---:|---:|---|
| Mineral | 54.965 / 43.039 | 224.145 / 176.435 | Texels only; excludes cache construction and mips |
| Actual-sized color mips | 8.052 / 5.135 | 13.033 / 7.510 | Includes allocations/cache construction; excludes source-map synthesis |

The mineral reference is the original scalar program; the accepted capture
baseline already had invariant constants prepared. The mip fixture uses actual
sampled 256/512 atlases from three scene seeds, with eight alternating timing
repetitions. These kernel speedups cannot be presented as capture or training
speedups. [Current matched capture qualification](generation_cpu_efficiency.md)
includes every preparation/upload/readback cost and reports smaller gains.

## Smallest credible prototype

Material synthesis remains the largest measured CPU preparation stage. More GI
dispatch slots cannot remove that CPU demand. A prototype should batch supported
maps for one prepared room into a bounded GPU work list, then consider a few
future rooms only under the existing residency and current-capture priority
rules. First distinguish owned task CPU time, queue delay and map tails: scoped
join wall times can include execution of other rooms and do not sum to exclusive
CPU work. Bounded row/tile CPU evaluation is a lower-risk intermediate experiment
if it preserves the original texel, normal and mip arithmetic.

Start with one bounded static paint map program, not a general material compiler
or per-fragment procedural material. Keep the current CPU engine as the default
and independent reference. The prototype should:

1. Accept the existing sampled material recipe, fixed map resolution, prepared
   constant colors and cell attributes. Sampling and distribution stay on the
   CPU. Batch a room's supported map jobs into one bounded GPU work list; avoid
   waiting once per map or requiring an API/configuration knob downstream.
2. Evaluate base color/height/roughness/AO once into bounded storage buffers. A
   subsequent dispatch derives periodic metric finite-difference normals from
   the same height grid. Leaf/clamped edge programs must retain their own edge
   rule when they are added later.
3. Produce full RGBA8 map chains in packed-u32 output buffers and copy them into
   sampled textures. This avoids relying on sRGB storage texture writes, which
   WebGPU does not permit. Use aligned buffer-to-texture rows and the exact
   existing sRGB versus linear formats, sampler, UV transforms and mip count.
4. Preserve the joint normal/roughness moment reduction, described below, rather
   than relying on automatic mips. Dispatch mip levels in order; pack unrelated
   map jobs together where possible.
5. Include completion of the GPU texture bake in scene readiness before rendering
   any RGB/annotation capture. Cancel results from superseded generation keys.
   Retain bounded buffers and retire completed scene resources using the existing
   ownership/lifecycle mechanism.

At 512 square, three RGBA8 image chains require 4,194,300 logical bytes per map
triplet, before row alignment. Temporary f32 height and normal-moment buffers
must be accounted separately and included in the scene residency budget. Do not
retain every material program ever seen.

## GI and export are part of the contract

Current diffuse GI samples a 32-square proxy from the full-size base color image.
It selects texels at integer coordinates `(x * full_width / 32,
y * full_width / 32)` and decodes their quantized sRGB bytes with a 256-entry LUT.
This is point subsampling, not an average or evaluation at pixel centers. A
future CPU proxy must evaluate those exact full-grid positions, preserve the same
RGB quantization and base-color/metalness multipliers, and keep all UV/emission
rules. The proxy contains just 1,024 RGB samples, so it is cheap relative to a
65,536/262,144-texel atlas. Make it explicit metadata consumed by GI; disguising
it as a smaller full-size `Image` would change the current sampling contract.

`reference.rs` exports actual CPU image data today. A GPU-only `Image` with no
CPU data would silently lose texture information in GI and reference export.
The prototype therefore needs lazy readback of the actual GPU-produced map chain
when a caller requests a reference/texture export. Annotation geometry and
semantic exports remain attached to the same material-independent geometry.
Until this export/readiness path is implemented, GPU-only material output is not
a correct replacement for the current engine.

## Required normal/roughness mip behavior

For the base level, decode quantized normal bytes and normalize, and initialize
the roughness moment with `(roughness_byte / 255)^4`. Reduce children in the
existing order with weight 0.25. Carry the **unnormalized** normal first moment
and the raw roughness fourth moment to the next level; do not reconstruct them
from rounded output bytes. For each emitted mip texel:

- `length = clamp(length(normal_moment), 0.0001, 1)`;
- `variance = (1 - length) / length`;
- perceptual roughness is `sqrt(sqrt(min(roughness4 + variance, 1)))`;
- normalize the moment only for emitted normal bytes;
- AO and metalness use the existing integer channel average;
- use the existing rounded normal/roughness encoding and opaque alpha.

Color mips must use the current sRGB decoding/encoding and child accumulation
order. Generic linear-RGBA averaging, renormalizing intermediate moments, or
averaging perceptual roughness would visibly reduce material quality.

## Numerical qualification and promotion gate

WGSL floating-point contraction and transcendental implementations are not a
bit-exact substitute for native Rust/WASM `powf`, `sin`, `cos`, `exp` or `sqrt`.
Merely translating the current Rust program into a shader is insufficient.
The official [WGSL numerical rules](https://www.w3.org/TR/WGSL/), sections
15.7.4–15.7.5, specify operation accuracy and permit reassociation/fusion;
CPU/GPU bit equivalence must be measured rather than inferred from shared formulas.
CPU-prepared invariant colors and hash-cell attributes reduce duplicate work,
but do not prove that variable arithmetic or quantization is identical.

For a narrow paint prototype, the sRGB-to-byte transfer could be represented by
CPU-produced f32 threshold tables rather than GPU `pow`; this preserves transfer
decisions for identical input bits. It does not resolve differences in the input
spatial arithmetic or normal normalization. Those still need independent replay.

Before promotion, compare every output channel and all mips at 256 and 512,
including extreme seeds, seam/negative UVs, PBR scalars and descriptors. Validate
the 32-square GI proxy and scene reference export against the actual rendered
textures. Repeat on native Vulkan and browser WebGPU. Then compare matched RGB
and all annotations across human/no-human and irregular envelopes, and measure
the end-to-end JIT generation-plus-training workload. Shifting material work onto
the training GPU can reduce CPU utilization while making total training slower.

The current exact CPU map path remains the fallback for unsupported recipes or
backends. Full-scene replay for the current CPU optimization round matches all
saved RGB/annotation/metadata bytes against its accepted local baseline; see
[current capture qualification](generation_cpu_efficiency.md). Map identity and
rendered-frame identity remain separate gates. Any GPU path that fails the accepted numerical replay contract must
remain an explicit unqualified experiment; it should not be hidden behind a
quality-preserving throughput claim.
