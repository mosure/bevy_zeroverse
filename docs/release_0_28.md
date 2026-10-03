# bevy_zeroverse 0.28

Patch **0.28.1** (`bevy_zeroverse_burn` **0.11.1**) fixes seed **1,013,005**
being rejected for an inverted glass-window triangle. Roof clipping previously
interpolated in float32, allowing a near-endpoint intersection to round outside
its source edge. Intersections now use double-precision arithmetic and a
consistent edge direction, preserving normals and UVs. Geometry validation and
the existing degeneracy threshold are unchanged; the seed is not filtered out.
Regression coverage includes the complete failing room, the extracted bevel at
multiple scales and windings, surface coverage and shared-edge agreement, and
1,024 envelope constructions including the surrounding seed range. The FFI
crate also advances to **0.28.1**, and both wrappers require the fixed core.

This release improves native procedural capture throughput while preserving
geometry, material detail, lighting budgets and capture-readiness checks.

Crates: `bevy_zeroverse` and `bevy_zeroverse_ffi` **0.28.1**;
`bevy_zeroverse_burn` **0.11.1**. Capture/publication and SigLIP2 versions are
unchanged. The capture wire format remains v35 and the scene generator remains 22;
the source fingerprint identifies this implementation and its refreshed captures.

## Generation

- Sequential indoor CLI generation prepares up to three future rooms on the CPU
  while the current room renders. `--indoor-prefetch-depth` accepts 1–4;
  `--indoor-prefetch=false` disables lookahead. Random access and interactive
  requests remain non-speculative.
- Independent geometry builders run on a bounded native pool. Byte-color lookup
  and indexed-vertex transforms remove redundant CPU work without rounding changes.
- Headless cameras reuse completed GPU attachments, readback storage and the
  co-visibility atlas. Each new room gets fresh capture/annotation state, and
  readiness explicitly checks the current room's identity. Active/failed copies
  are never recycled. Camera count, size and modality changes replace storage.
- GPU diffuse transport uses a deterministic surface-area BVH with bounded
  parallel construction. Probe placement retains the independent CPU index.
- A matched 128-room test improves three-view 512×512 capture from 6.14 to
  **9.97 views/s (1.62×)**. Complete single-process exports improve from 4.85 to
  **7.02 views/s**; two workers reach **8.65 views/s**. This does not meet the
  requested additional 2×. These optimized-dev workstation measurements use the
  checkout's existing WGPU patches; registry WGPU was not performance-tested.
- Presenters no longer share a fixed normalized room coordinate. Placement
  samples free space near actual presentation boards, with continuous position
  and heading variation and the existing collision/support checks. Across 512
  rooms, the repeated anchor disappears (38 → 0), the peak primary-person heatmap
  cell falls from 38 to 8, and total population remains 1,144.

The [throughput report](generation_throughput.md) records configuration, memory,
commands and scope. The matched 64-room single-process comparison is bit-exact
for RGB and all exported annotation tensors. The placement correction was applied
separately from that comparison so changed people do not inflate the speed claim.

## Rust migration

Exhaustive `AppFrameRequest` literals must set `prefetch_indoor` or use
`..Default::default()` (0). This field is a future-room count, capped at 4.
Request only consecutive rooms that will be consumed, reducing the count near
the end and setting it to 0 for the final request.

Exhaustive `GenConfig` literals must set `indoor_prefetch` and
`indoor_prefetch_depth` or use `..Default::default()` (true and 3). The generator applies lookahead only to sequential
indoor captures and suppresses it after the final sample. These public struct
additions motivate the minor version increments; existing CLI commands and
dataset tensor contracts remain compatible.
