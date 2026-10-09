# bevy_zeroverse ♾️

[![test](https://github.com/mosure/bevy_zeroverse/workflows/test/badge.svg)](https://github.com/Mosure/bevy_zeroverse/actions?query=workflow%3Atest)
[![crates.io](https://img.shields.io/crates/v/bevy_zeroverse.svg)](https://crates.io/crates/bevy_zeroverse)
[![License](https://img.shields.io/github/license/mosure/bevy_zeroverse)](LICENSE)

Bevy Zeroverse is a procedural synthetic reconstruction dataset generator.
Sample textured primitive objects or furnished interiors, capture synchronized
views, and export calibrated cameras with aligned images, geometry and labels.

[Project page](https://mosure.github.io/bevy_zeroverse/project/) ·
[WebGPU viewer](https://mosure.github.io/bevy_zeroverse/?scene_type=procedural-indoor&indoor_seed=7&num_cameras=4&camera_grid=true&regenerate_ms=0) ·
[Whitepaper](https://mosure.github.io/bevy_zeroverse/project/static/papers/bevy_zeroverse.pdf) ·
[Documentation](docs/README.md)

## Object Zeroverse

The original object scenes combine parametric primitives, mesh deformation,
randomized materials and multi-view capture. The Python/PyTorch dataloader can
generate samples online.

![Object-scene dataloader captures showing matched RGB, depth and normal views](docs/bevy_zeroverse_dataloader_grid.webp)

```sh
cargo run --bin viewer -- --scene-type object --num-cameras 4 --camera-grid
```

[Open object scenes in the WebGPU viewer](https://mosure.github.io/bevy_zeroverse/?scene_type=object&num_cameras=4&camera_grid=true&regenerate_ms=8000)

## Procedural interiors

<table>
  <tr>
    <td width="33%" align="center">
      <a href="www/project/static/media/architecture/s7-t0-c3-color.webp"><img src="www/project/static/media/architecture/s7-t0-c3-color.webp" width="320" height="200" alt="Generated office with mezzanine stairs, desks and tall windows"></a><br>
      <strong>Mezzanine &amp; stairs</strong><br><sub>Seed 7 · camera 3</sub>
    </td>
    <td width="33%" align="center">
      <a href="www/project/static/media/architecture/s8-t0-c3-color.webp"><img src="www/project/static/media/architecture/s8-t0-c3-color.webp" width="320" height="200" alt="Generated lounge with a sunken floor, steps and glazed partitions"></a><br>
      <strong>Sunken lounge</strong><br><sub>Seed 8 · camera 3</sub>
    </td>
    <td width="33%" align="center">
      <a href="www/project/static/media/architecture/s6-t0-c1-color.webp"><img src="www/project/static/media/architecture/s6-t0-c1-color.webp" width="320" height="200" alt="Generated training room with a raised platform and arched doorway"></a><br>
      <strong>Raised floor &amp; archway</strong><br><sub>Seed 6 · camera 1</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <a href="www/project/static/media/architecture/s4-t0-c0-color.webp"><img src="www/project/static/media/architecture/s4-t0-c0-color.webp" width="320" height="200" alt="Generated studio with a sloping ceiling, pillars, shelving and lounge seating"></a><br>
      <strong>Sloping ceiling &amp; pillars</strong><br><sub>Seed 4 · camera 0</sub>
    </td>
    <td width="33%" align="center">
      <a href="www/project/static/media/architecture/s2-t0-c3-color.webp"><img src="www/project/static/media/architecture/s2-t0-c3-color.webp" width="320" height="200" alt="Generated workspace with tall exterior glazing and daylight shadows"></a><br>
      <strong>Tall glazing &amp; daylight</strong><br><sub>Seed 2 · camera 3</sub>
    </td>
    <td width="33%" align="center">
      <a href="www/project/static/media/architecture/s28-t0-c0-color.webp"><img src="www/project/static/media/architecture/s28-t0-c0-color.webp" width="320" height="200" alt="Dimly lit generated interior with blue seating, a plant and a concrete pillar"></a><br>
      <strong>Low-light interior</strong><br><sub>Seed 28 · camera 0</sub>
    </td>
  </tr>
</table>

Selected native captures at the start of each camera trajectory, with no exposure
correction. Click a thumbnail for the full image. The project page includes more
rooms, synchronized views, matching annotations, floor plans and distributions.

## Capabilities

- **Object scene programs:** parametric primitives, mesh deformation, rotation
  augmentation and material sampling, with Cornell cube and simple room scenes.
- **Continuous scene programs:** polygonal footprints, cut-ins, chamfered corners,
  sloped ceilings, arches, pillars, floor levels and mezzanines. Windows and glass
  partitions connect interiors to neighboring rooms and outdoor context.
- **Procedural furnishings and appearance:** multipart furniture, potted plants,
  fixtures and clutter, with varied geometry, placement, PBR textures, glass and
  lighting. Architecture and furnishings need no downloaded mesh or texture catalog.
- **Calibrated multi-view cameras:** sampled intrinsics, primary-room placement,
  collision-checked trajectories and adjustable baselines with shared-view constraints.
- **People and motion:** AnnyBody meshes with procedural clothing and optional
  ARDY text/waypoint motion. Cameras and people share a capture timeline; static
  scenes do not initialize motion models.

Semantic-room and standalone-human scenes are also supported. The engine is
built with Bevy 0.20 and Rust, with native dataset generation, a WebGPU viewer, and
Python/PyTorch integration.

## Quick start

Run the native viewer using the repository's pinned `nightly-2026-08-31` Rust toolchain (crate MSRV
1.97.1). The pin keeps native and WebGPU builds on the qualified compiler.
Optional motion and SigLIP2 inference use Burn 0.22:

```sh
cargo run --bin viewer -- --scene-type procedural-indoor --indoor-seed 7 \
  --num-cameras 4 --camera-grid
```

Native JIT training can use `bevy_zeroverse_burn::gpu::GpuLiveDataset` (feature
`gpu_tensor`) to render into Burn-owned buffers on the same device, retaining
full native quality. See the [tensor contract and benchmark](docs/gpu_jit.md).

Scene Studio uses Bevy 0.20 Feathers controls and the built-in pan-orbit camera
on native and WebGPU; it needs no egui, inspector or external camera plugin. **Apply changes**
keeps the selected seed; **Next / R** advances it. Editing does not regenerate.
Camera baseline controls spacing between views; travel controls how far each
camera moves. The viewport selector offers **Editor camera**, **Capture grid**,
and **Room schematic**. Browser links preserve the active scene, preview and pending edits.
**View & playback → Glass in annotations** selects glass surfaces (default) or
unrefracted geometry behind glass for depth, normal, position, semantic, optical
flow and co-visibility. RGB transmission is unchanged. The same policy is available
as `--annotation-glass through` or `?annotation_glass=through` and recorded in
exports (`annotation_glass`: 0 = surface, 1 = through in tensor datasets).

Generate a dataset with four views per room:

```sh
cargo run -p bevy_zeroverse_burn --bin zeroverse_gen -- \
  --scene-type procedural-indoor --output out/indoor_dataset \
  --samples 100 --seed 0 --workers 1 --chunk-size 16 \
  --cameras 4 --width 640 --height 480 --playback-steps 1 \
  --indoor-camera '{"baseline":0.5}' \
  --render-modes color depth normal semantic position co-visibility \
  --ov-mode disabled --no-ui
```

Add `--schematic` to export top-down diagrams at each capture timestep.

The native CLI requires a GPU but no display server. Capture waits for assets,
render pipelines and requested motion. People use the body assets in
`assets/burn_human`; set `--indoor-human-density 0` to generate interiors without them.

## MatSynth

Object scenes can sample PBR materials from
[MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth). The original material
grid shows the texture variety available alongside the procedural indoor materials.

![MatSynth material grid rendered in the Bevy Zeroverse viewer](docs/bevy_zeroverse_material_grid.webp)

Download the maps using the dataset's
[download script](https://huggingface.co/datasets/gvecchio/MatSynth/blob/main/scripts/download_dataset.py),
then resize them into the local material directory:

```sh
python mat-synth/resize.py --source_dir <path-to-mat-synth> --dest_dir assets/materials
cargo run --bin viewer -- --material-grid
```

[Open the material grid in the WebGPU viewer](https://mosure.github.io/bevy_zeroverse/?material_grid=true)

## Exports

| Output | Contents |
| --- | --- |
| RGB and calibration | Color images, camera intrinsics, poses and capture times |
| Depth, normals and positions | Native float32 geometric annotations |
| Semantics and instances | Class labels, object bounds, primary-room AABB and human joints |
| Optical flow and motion vectors | Temporal correspondence, validity and visibility masks |
| Co-visibility | Per-pixel membership in up to 16 capture cameras |
| Room schematic | Metric top-down JSON/SVG/PNG with captured cameras, skeletons and optional prediction overlays |
| O-voxel | Primary-room surface geometry and semantics |

Exports support chunked or per-sample storage, resume validation and distributions
of object counts, placements, lighting and camera parameters. Glass is the first
surface in geometry annotations. O-voxel requires one timestep and disabled human
motion. Split training and evaluation by room seed.

Materials, people and indirect lighting remain approximate; photographic realism
and downstream training gains require further evaluation.

## Credits

[Zeroverse](https://github.com/desaixie/zeroverse) ·
[LGM](https://github.com/3DTopia/LGM) ·
[MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth)
