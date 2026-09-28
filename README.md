# bevy_zeroverse ♾️

[![test](https://github.com/mosure/bevy_zeroverse/workflows/test/badge.svg)](https://github.com/Mosure/bevy_zeroverse/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_zeroverse)](https://raw.githubusercontent.com/mosure/bevy_zeroverse/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_zeroverse)](https://github.com/mosure/bevy_zeroverse)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_zeroverse)](https://github.com/mosure/bevy_zeroverse/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_zeroverse.svg)](http://isitmaintained.com/project/mosure/bevy_zeroverse)
[![crates.io](https://img.shields.io/crates/v/bevy_zeroverse.svg)](https://crates.io/crates/bevy_zeroverse)

### [whitepaper (PDF)](https://mosure.github.io/bevy_zeroverse/project/static/papers/bevy_zeroverse.pdf) | [project page](https://mosure.github.io/bevy_zeroverse/project/)

bevy zeroverse synthetic reconstruction dataset generator. view the [live demo](https://mosure.github.io/bevy_zeroverse/?yaw_speed=0.7&num_cameras=4&camera_grid=true&regenerate_ms=8000&plucker_visualization=true).


## capabilities

- [X] depth/normal rendering modes
- [X] plücker camera labels
- [X] generate parameteric zeroverse primitives
- [X] primitive deformation
- [x] procedural zeroverse composite environments
- [x] online torch dataloader
- [x] safetensor chunking
- [x] hypersim semantic labels
- [x] [ovoxel](https://arxiv.org/abs/2512.14692) annotation
- [x] obb annotation
- [x] [procedural humans](https://arxiv.org/abs/2511.03589)
- [x] procedural furnished `procedural_indoor` scenes with seeded layouts and generated PBR surfaces
- [ ] primitive boolean operations
- [ ] primitive pbr wireframe
- [ ] primitive 4d augmentation


## procedural interiors

```sh
cargo run --bin viewer -- --scene-type procedural-indoor --indoor-seed 6
```

Conference rooms, open offices, lounges and training rooms use multipart furniture,
metric texture coordinates, generated PBR textures, shadowed lighting, window recesses
and a furnished neighboring room behind glass. No downloaded mesh or texture catalog
is required for the architecture and furnishings. People use the bundled AnnyBody
reference in `assets/burn_human`; `--indoor-human-density 0` runs without it. Existing
scene types remain available.

Generator v13 adds capture readiness barriers, continuous furniture and clothing
parameters, glazing and lighting variation, and validated native/WebGPU human
motion. The [current review](docs/indoor_review_v13.md) includes room and people
galleries, annotation checks and a 128-room distribution audit.

Generator v10 adds denser functional furnishing, wider collision-aware rotations,
content-aware cameras and tangent/material fixes. The [quality and diversity review](docs/procedural_domain_v10.md)
includes rendered visibility checks and a physical Cycles comparison. An optional
[SigLIP2 audit](docs/embedding_audit.md) measures spacing between captured scenes
using cached, verified model shards; it adds no model initialization to generation.

Generator v9 adds continuous workstation fields, variable recessed niches and plant
morphology, and broader interior camera placement. The [capture distribution review](docs/procedural_domain_v9.md)
includes 144 captured rooms, count distributions, camera/placement heatmaps, visual
repetition diagnostics and explicit quality gaps. The [generator-v8 review](docs/procedural_domain_v8.md)
records the room, material and low-light-to-sunlight domains. The [generator-v7 review](docs/scene_quality_v7.md)
records the controls, human surfaces, glazing and annotation fixes. The generator-v6
[program review](docs/procedural_program_review.md) retains its historical
native/browser checks and matched Cycles comparisons. The earlier
[generator-v5 review](docs/local_scene_quality_review.md) is retained as historical evidence.
See the [0.23 release notes](docs/release_0_23.md) for API and capture compatibility, the [technical paper](tex/bevy_zeroverse.tex), and the [measured generation/diversity review](docs/generation_v18.md).

See the [generation, capture and validation guide](docs/procedural_indoor.md) for
dataset examples, distribution evidence and current rendering limits. The
[generator-v4 review](docs/procedural_indoor_review_v4.md) covers architectural
families, six potted plant forms, desk clutter, render inspections and dataset
metrics. Native capture includes diffuse GI and float32 geometry annotations;
the [WebGPU viewer](docs/procedural_indoor_web.md) has explicit quality profiles.

## burn_siglip2

The independently versioned [burn_siglip2](crates/burn_siglip2) crate is maintained
and published from this workspace. It provides Burn 0.21 SigLIP2 image/text
inference, cached model loading and native/browser GPU support. See its
[release guide](crates/burn_siglip2/RELEASING.md) and the optional
[capture embedding audit](docs/embedding_audit.md).

## dataloader

![Alt text](docs/bevy_zeroverse_dataloader_grid.webp)

```python
from bevy_zeroverse_dataloader import BevyZeroverseDataset
from torch.utils.data import DataLoader

dataset = BevyZeroverseDataset(
    editor=False, headless=True, num_cameras=6,
    width=640, height=480, num_samples=1e6,
)
dataloader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=1,
)

for batch in dataloader:
    visualize(batch)
```

### chunked dataloader

requires nvjpeg: https://developer.nvidia.com/nvjpeg


## mat-synth

- download the mat-synth dataset [here](https://huggingface.co/datasets/gvecchio/MatSynth/blob/main/scripts/download_dataset.py)
- resize the mat-synth dataset (4k is heavy) using `python mat-synth/resize.py --source_dir <path-to-mat-synth> --dest_dir assets/materials`
- material basecolor grid view (`cargo run -- --material-grid` or [live demo](https://mosure.github.io/bevy_zeroverse?material_grid=true))

![Alt text](docs/bevy_zeroverse_material_grid.webp)


The current checkout uses Bevy 0.19.1, Burn 0.21.0, `burn_human` 0.5.1 and
`bevy_burn_human` 0.6.1. Capture metadata includes an engine identity; datasets
created with the older rendering contract cannot be resumed into a mixed-version
capture stream. Existing dataset files remain readable. See the
[physical accuracy and memory review](docs/procedural_indoor_review_v6.md)
for measured Cycles differences and the failed continuous-process stability gate.
Production CLI generation should retain its process lifetime limit. The
[wgpu dependency review](docs/wgpu_dependency_review.md) measures why the local
wgpu optimizations remain enabled. Crate packages exclude the vendored sources
and resolve registry wgpu; local memory and performance results do not qualify
that unpatched configuration.

## compatible bevy versions

| `bevy_zeroverse` | `bevy` |
| :--                       | :--    |
| `0.24`, `0.23`, `0.22`, `0.21`, `0.20`, `0.19` | `0.19.1` |
| published `0.17`           | `0.17` |
| `0.8`                     | `0.16` |
| `0.6`                     | `0.15` |
| `0.2`                     | `0.14` |
| `0.1`                     | `0.13` |


## credits

- [lgm](https://github.com/3DTopia/LGM)
- [mat-synth](https://huggingface.co/datasets/gvecchio/MatSynth)
- [zeroverse](https://github.com/desaixie/zeroverse)

See [numeric optical flow and motion vectors](docs/optical_flow.md) for temporal annotation conventions and lossless export.

See [co-visibility annotations](docs/co_visibility.md) for per-pixel membership in up to 16 capture cameras, additive color legends, and lossless multiview dataset exports.

See [optional indoor human motion](docs/human_motion.md) for cached ARDY text/waypoint generation, synchronized playback, and native/WebGPU validation.
See the [v19 clothing and viewer review](docs/viewer_quality_v19.md) for matched clothing renders, motion controls, pose occlusion, editor intrinsics and frame-rate-independent flow previews.
See the [v13 indoor review](docs/indoor_review_v13.md) for capture readiness barriers,
native and browser motion checks, expanded procedural parameters, glass changes,
and measured distributions with room, furniture and people galleries.
See the [v12 room review](docs/room_review_v12.md) for primary-room camera paths,
furniture programs, fixture annotations, compound motion and measured annotation
preview performance, with captured galleries and 128-room distribution metrics.

See the [generator-20 camera evaluation](docs/camera_evaluation_v20.md) for 2,048 audited layouts, 512 rendered rooms, trajectory diversity and all-camera co-visibility measurements, and the [0.24 release notes](docs/release_0_24.md) for compatibility.
