# bevy_zeroverse 🌌

[![test](https://github.com/mosure/bevy_zeroverse/workflows/test/badge.svg)](https://github.com/Mosure/bevy_zeroverse/actions?query=workflow%3Atest)
[![GitHub License](https://img.shields.io/github/license/mosure/bevy_zeroverse)](https://raw.githubusercontent.com/mosure/bevy_zeroverse/main/LICENSE)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/mosure/bevy_zeroverse)](https://github.com/mosure/bevy_zeroverse)
[![GitHub Releases](https://img.shields.io/github/v/release/mosure/bevy_zeroverse?include_prereleases&sort=semver)](https://github.com/mosure/bevy_zeroverse/releases)
[![GitHub Issues](https://img.shields.io/github/issues/mosure/bevy_zeroverse)](https://github.com/mosure/bevy_zeroverse/issues)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/mosure/bevy_zeroverse.svg)](http://isitmaintained.com/project/mosure/bevy_zeroverse)
[![crates.io](https://img.shields.io/crates/v/bevy_zeroverse.svg)](https://crates.io/crates/bevy_zeroverse)

bevy zeroverse synthetic dataset generator plugin. view the [live demo](https://mosure.github.io/bevy_zeroverse)


## capabilities

- [ ] generate parameteric zeroverse primitives
- [ ] depth/normal rendering modes
- [ ] procedural zeroverse environments

## mat-synth

- download the mat-synth dataset [here](https://huggingface.co/datasets/gvecchio/MatSynth/blob/main/scripts/download_dataset.py)
- crop the mat-synth dataset (4k is heavy) using `python tools/mat_synth_resize.py --source_dir <path-to-mat-synth> --dest_dir <path-to-cropped-mat-synth>`
- generate manifolds using `python tools/generate_manifolds.py`


## compatible bevy versions

| `bevy_zeroverse` | `bevy` |
| :--                       | :--    |
| `0.1`                     | `0.13` |


## credits

- [mat-synth](https://huggingface.co/datasets/gvecchio/MatSynth)
- [zeroverse](https://github.com/desaixie/zeroverse)
