# Primary-room scene AABB correction

Local validation on 2026-09-27. The scene AABB annotation and its cyan viewer
gizmo now use the same transformed primary-room region as O-voxel export,
including its structural shell. Neighboring rooms, shared slabs outside the
crop and exterior backdrops do not enlarge the box. Other scene types retain
their geometry-derived bounds. Bounds are updated after transform propagation
and before annotation materials, gizmos and sampling.

Voxelization preserves supplied nondegenerate corners exactly; recomputing
`min + (max - min)` previously changed some upper corners by a float rounding
unit. Position annotations use the box as an affine frame without clamping:
context outside it remains a valid geometric hit and can normalize outside
`[0, 1]`. This is capture identity **v24**, so old capture resumes cannot silently
mix the previous full-scene normalization with the new primary-room contract.

## Captured checks

- Four consecutive seeds **11–14**, two cameras each, rotated rooms, static
  people, RGB/depth/normal/semantic/position, one timestep, 64³ O-voxel grid.
- CPU and GPU captures both have **exactly equal `aabb` and `ovoxel_aabb`**.
  The audit independently reconstructs the primary-room crop from the manifest
  and checks that all captured cameras belong to that room.
- CPU/GPU sparse fields pass the existing comparison tolerances, with one bake
  per scene. See [validation.json](validation.json).
- A filesystem export of seed 11 passes the same checks. A capture with voxel
  export disabled has the same primary-room AABB and no voxel tensors.
- Across eight CPU views, **2,163 visible context pixels** lie outside the AABB.
  Decoding position with the exported bounds still agrees with depth: worst
  per-view p99 depth discrepancy **2.72 µm**, reprojection **0.00220 pixels**.
  [summary.json](summary.json) retains per-view measurements.

The 137-test library run passed before the final voxel-corner preservation fix.
After that fix, all five targeted AABB/position tests passed; ten storage
round-trip tests also passed. Strict workspace/all-target Clippy, formatting,
diff checks and a Wasm viewer check with `web,human_motion` passed. Logs are in
`checks/`. Cargo still reports the upstream `burn-cubecl 0.21` future-Rust
compatibility notice. The Wasm check is compilation, not a new browser runtime
qualification.

This is a bounded correctness check, not a distribution or photorealism claim.
Commands are recorded in [runs.json](runs.json); raw exports and the frozen
binary remain in `out/primary_aabb_review/final/`. [provenance.json](provenance.json)
records source and binary hashes. Use the current
`scripts/validate_indoor_ovoxel.py` for the new contract; older captures with
full-scene annotation bounds intentionally fail its equality check.
