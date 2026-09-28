# Primary-room O-voxel qualification

Local native Vulkan check on 2026-09-27, NVIDIA RTX PRO 6000 Blackwell
Workstation Edition / driver 610.43.02. Four consecutive seeded rooms, two
cameras each, one timestep, static people enabled, seeded world rotation,
RGB/depth/semantic/position capture. The bounded check tests annotation scope,
backend agreement and capture behavior; it is not a 10M-scene or memory-stability
qualification.

| Seed | Grid | Selected / total objects | Selected / total people | Sparse cells | CPU bake (s) | GPU bake (s) |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 64³ | 22 / 83 | 2 / 7 | 50,538 | 0.142 | 0.160 |
| 12 | 64³ | 26 / 66 | 2 / 4 | 50,630 | 0.150 | 0.538 |
| 13 | 128³ | 23 / 49 | 6 / 9 | 232,615 | 0.512 | 0.158 |
| 14 | 128³ | 11 / 70 | 1 / 7 | 186,744 | 0.261 | 0.610 |

All eight backend captures passed sparse-field validation, independently
calculated primary bounds, rotated crop containment, required architecture
semantics and the single-timestep/static-motion contract. CPU/GPU coordinates,
crossing flags, semantic IDs and palettes matched exactly. Maximum dual
component discrepancies were 2, 4, 7 and 3 uint8 units respectively; measured
world-space errors are in the JSON reports. At least 99.998% of components
agreed within one quantization unit. Maximum averaged semantic-color difference
was 4/255. Backends are numerically close, not bitwise interchangeable.

Main-thread mesh extraction/clipping took 7–27 ms. Bake times above include
async CPU work or GPU queue work/readback; they exclude scene preparation,
lighting and image rendering. Each two-room process took about 5–6 seconds
including engine startup. These short timings do not establish GPU superiority;
CPU remains the default. Every room had cache version 1, and cumulative image
capture requests were 1 then 2, confirming that waiting for voxels did not
repeat image captures.

A separate **child-process filesystem** capture of seed 11 at 128³ passed the
same independent audit: 236,070 cells, 13.6 ms preparation, 394 ms CPU bake.
Both packed chunks and `meta.safetensors` preserve aligned sparse tensors.

## Failure and regression checks

- Two timesteps, generated motion, and explicit trajectories with fraction zero
  each failed preflight in about 10 ms, before adapter/model startup or output
  creation.
- Capacity of one occupied cell failed on both backends in about 3 seconds
  including scene/engine startup. No partial safetensor was written.
- Analytic tests cover rotated triangle clipping, excluded subtrees, cache
  behavior, canonical grid-edge crossings, thin-triangle closest points, GPU/CPU
  sparse parity, capacity overflow, malformed payloads and temporal policy.
- Native library, storage round-trip, capture-alignment, strict workspace Clippy
  and WebAssembly compilation checks are recorded in `checks.txt`. WebAssembly
  compilation is not a browser runtime qualification of the O-voxel GPU backend.
- A legacy headless smoke test exposed zero normals from absolute-size weighting
  on dense primitives. The producer now uses scale-independent angle weighting,
  with analytic-normal preservation for unused UV pole vertices. Ground-truth
  validation remains strict; this fix does not change procedural indoor meshes.

## Artifacts and reproduction

- [64³ CPU/GPU report](validation.json), [128³ CPU/GPU report](validation128.json)
- [Filesystem report](validation_fs.json)
- [64³, rejection and filesystem commands](capture_runs_v3.json), [128³ commands](capture_runs_128.json)
- [RGB, full-plan selection and voxel cutaways](primary_room_review.png)
- `provenance.json`: hashes of relevant sources and retained check logs.

Raw outputs remain in `out/ovoxel_review/`. Re-run the commands with fresh output
directories, then run `scripts/validate_indoor_ovoxel.py`; `make_preview.py`
reproduces the figure from the recorded local paths with NumPy and Matplotlib.
Gray points in the plan are excluded object origins. The orange line surrounds
the selected primary room; the voxel plot includes its structural shell and
shows only the 0.18–1.7 m height band so the ceiling/floor do not hide furniture.

The payload is conservative surface occupancy with quantized dual points and
semantic colors. It does not provide solid fill, a watertight mesh, textured
albedo or exact sub-voxel geometric reconstruction. See the full
[export contract](../../ovoxel_indoor.md).
