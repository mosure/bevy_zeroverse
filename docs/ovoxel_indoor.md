# Indoor O-voxel export

O-voxel annotations for `procedural-indoor` describe the **primary reconstruction
room**, using the same largest program zone as camera placement. Shared floors,
ceilings, walls, trim and glazing are clipped to this room in scene-local
coordinates before applying the room's world rotation. The crop includes a
20 cm horizontal structural shell, the floor slab down to -20 cm, and the ceiling
slab up to room height +14 cm. This preserves enclosing wall/window thickness.
Furniture and static people are selected by their origins inside the unpadded
primary zone; neighboring-room instances are excluded explicitly. Their geometry
is also clipped. Neighboring rooms and outdoor context remain visible in RGB.

The annotated scene `aabb` and its viewer gizmo use these same world-space bounds,
even with O-voxel export disabled. For default indoor exports, `aabb` equals
`ovoxel_aabb`; backdrops and neighboring rooms cannot enlarge either one.
World rotation is applied before calculating both axis-aligned bounds. Position
annotations use this AABB as an affine coordinate frame, **without clamping**:
visible context outside the crop can have coordinates below zero or above one.
Decode every valid hit with `min + position * (max - min)`. This preserves
depth/position alignment while the reconstruction target remains the primary room.
The [AABB correction report](evidence/primary_aabb/README.md) checks both voxel
backends, filesystem export and capture with voxel export disabled.

O-voxel export requires **exactly one timestep and human motion disabled**.
Generated motion and explicit trajectories are both rejected before CLI engine
startup. An explicit `{"fraction":0}` policy with no trajectories is permitted.
The Rust generator, Python initialization and capture readiness enforce the same
contract, including the motion policy already applied to an active scene. Use
`--ov-mode disabled` for temporal datasets. Existing scene types retain their
tracked-geometry export scope.

```sh
cargo run -p bevy_zeroverse_burn --bin zeroverse_gen -- \
  --scene-type procedural-indoor --output out/indoor_voxels \
  --samples 16 --workers 1 --chunk-size 4 --seed 11 \
  --width 320 --height 240 --cameras 2 --playback-steps 1 \
  --indoor-camera '{"multiview":{}}' --rotation-augmentation \
  --render-modes color depth normal semantic position \
  --ov-mode cpu-async --ov-resolution 128 --ov-max-output-voxels 2000000 \
  --color-codec raw --compression none --no-ui
```

`gpu-compute` selects the GPU backend. `--output-mode fs` stores the annotation
in each sample's `meta.safetensors`; the default chunk format concatenates sparse
fields with `ovoxel_offsets` and `ovoxel_semantic_label_offsets`. Both offset
tables contain **[start, count]**, not [start, end]. CPU/GPU output-capacity
overflow fails the sample instead of writing a truncated annotation. GPU device
buffer/dispatch limits and internal sparse-work capacity also fail explicitly;
reduce the budget/resolution or use CPU for such scenes.

## Geometry contract

- `ovoxel_coords`: unique, lexicographically sorted integer XYZ cells in
  `[0, resolution)`. The grid has the same number of cells on each axis;
  world-space cell dimensions follow the exported AABB and may be anisotropic.
- `ovoxel_dual_vertices`: uint8 cell-local positions. Decode as
  `aabb_min + (coord + dual / 255) * (aabb_max - aabb_min) / resolution`.
  Use **`ovoxel_aabb`**. It equals the annotated `aabb` for default indoor exports;
  legacy scenes or explicitly overridden voxel bounds may differ.
- `ovoxel_intersected`: bits 0/1/2 identify triangle crossings of the canonical
  +X/+Y/+Z grid edges from the cell's minimum corner. Mesh preview conversion
  honors these bits.
- `ovoxel_semantic`: uint16 indices into the per-sample JSON palette, whose index
  zero is `unlabeled`. `ovoxel_base_color` is the averaged **linear semantic
  palette color**, not textured PBR albedo. It retains the existing schema.

This is a conservative triangle **surface** representation: no solid interior
fill, signed distance, clipping-plane caps or watertight-mesh guarantee. Dual
positions average nearby closest points and are quantized/clamped to each cell.
Thin objects can disappear or merge at coarse resolution; conservative cells
near rotated crop boundaries can extend by a cell diagonal. Glass retains its
actual mesh surfaces independently of optical transparency. Do not interpret
this payload as exact mesh reconstruction or photometric ground truth.

## Scheduling and audit

Capture waits for scene/asset readiness and the selected root's completed bake.
CPU voxelization runs on the async compute pool; the GPU path uses sparse tiles
and bounded pooled buffers. Geometry extraction/clipping still runs on the main
thread and is timed separately. Material-mode changes and camera/light motion
do not invalidate the semantic geometry cache. Completed image planes remain
buffered while the bake finishes, without repeating GPU image capture.
The Rust generator disables voxel work entirely when `export_ovoxel=false`.

`indoor_render_metadata.ovoxel` records scope, bounds, triangle counts,
`preparation_seconds`, voxelization `elapsed_seconds`, backend, cache version,
wait updates and cumulative completed image-capture requests. The latter count
is per live engine, not per sample. Capacity and task failures propagate to the
headless generator; missing results have a bounded wall-time timeout.

For uncompressed chunks or filesystem exports (requires NumPy):

```sh
python scripts/validate_indoor_ovoxel.py out/indoor_voxels \
  --output out/indoor_voxels/validation.json
# Optionally add --compare PATH_TO_THE_SAME_SEEDS_FROM_THE_OTHER_BACKEND.
```

The audit checks sparse fields, palettes, sorted coordinates, temporal policy,
independently reconstructed primary-room bounds, rotated crop containment,
architecture semantics and redundant bakes. Comparison runs must use identical
scene/camera configuration and seeds. The local qualification report and visual
crop review are in [evidence/ovoxel](evidence/ovoxel/README.md).
Comparison requires identical cells, crossing flags and semantic IDs/palettes.
Float32 closest-point and accumulation differences are allowed at most eight
uint8 attribute units, with at least 99.9% of dual components within one unit.
The report records actual world-space error; it does not assert bitwise equality
of CPU/GPU dual positions or averaged colors.
