# Procedural interiors in WebGPU

The browser viewer builds the same seeded architecture, furnishings, AnnyBody
people and procedural PBR maps as the native scene. It supports camera grids,
annotation previews and optional human motion, with explicit rendering budgets.
Use the [live viewer](https://mosure.github.io/bevy_zeroverse/?scene_type=procedural-indoor&indoor_seed=24005&num_cameras=4&camera_grid=true&regenerate_ms=0)
or build locally.

People require `assets/burn_human`; `indoor_human_density=0` omits that reference.
Architecture and furnishings need no external mesh or material catalog.
Browser dataset readback is unsupported: `image_copiers=true` reports an error.
Use native `zeroverse_gen`, `indoor_validate` or the Python dataloader for datasets.

## Build and serve

```sh
rustup target add wasm32-unknown-unknown
cargo build --locked --release --bin viewer --target wasm32-unknown-unknown \
  --no-default-features --features web,human_motion
# Match the wasm-bindgen CLI to the version recorded in Cargo.lock.
cargo install wasm-bindgen-cli --version 0.2.128 --locked
wasm-bindgen --out-dir www/out --target web \
  target/wasm32-unknown-unknown/release/viewer.wasm
mkdir -p www/assets
cp -R assets/burn_human www/assets/
python3 -m http.server 8765 --directory www
```

Open:

```text
http://localhost:8765/?scene_type=procedural-indoor&indoor_seed=6&num_cameras=4&camera_grid=true&regenerate_ms=0&indoor_quality=auto
```

The `human_motion` feature enables ARDY controls and browser model loading; motion
models are only initialized when requested. Omit the feature for a smaller static
viewer. See the [motion guide](human_motion.md) for cached weights, inference
settings, waypoint planning and model limitations.

URL parameters accept underscore or hyphen field names, typed integer seeds,
scene/layout/quality names and serialized enum names. Hosting requires HTTPS or
localhost. The startup page checks WebGPU availability and reports initialization
failures. Change generation controls, then use **Regenerate / R** to apply them.

## Rendering profiles

| Capability | Native Auto | WebGPU Auto | Portable, either target |
| --- | --- | --- | --- |
| Procedural architecture, furniture, people and PBR maps | Yes | Yes | Yes |
| Generated environment reflections and PBR fixture lighting | Yes | Yes | Yes |
| Scene-baked diffuse irradiance volume | Yes | Disabled | Disabled |
| Shadow maps | Sun/spot 2048; point 1024 per face | 1024 | Disabled |
| Direct sun | Shadowed | Shadowed | Disabled to avoid light through walls |
| Glass | Screen-space specular transmission | Screen-space specular transmission | Alpha-blended approximation |
| SSAO | Yes | Disabled | Disabled |
| Bloom | Yes | Yes | Disabled |
| FXAA | Yes | Yes | Yes |
| Dataset readback | Yes | Unsupported | Native only |
| Dedicated float32 geometric capture | Yes | Unsupported | Native only |

Set `indoor_quality=portable` to reduce the effect budget while preserving the
manifest and geometry. It does not provide equivalent lighting or remove Bevy's
minimum WebGPU/PBR binding requirements.

Browser depth/normal/position/semantic previews use the interactive HDR display
path, including float16 intermediates. Screenshots are visualizations, not native
float32 training tensors. [Co-visibility](co_visibility.md) and
[temporal-vector previews](optical_flow.md) have their own documented conventions.

## Validate a browser deployment

Use a real WebGPU-capable browser and inspect rendered pixels, not just a
successful Wasm build or network response:

```sh
python3 -m pip install playwright pillow numpy
python3 scripts/validate_project_viewer.py \
  --url http://127.0.0.1:8765/project/ --output out/project_viewer
python3 scripts/smoke_indoor_browser.py \
  --url http://127.0.0.1:8765/ --output out/indoor_browser
```

The first check follows the project page's exact viewer link and requires four
nonblank camera tiles, a loaded scene and no browser/GPU errors. The second checks
occupied Auto/Portable RGB and semantic views for missing opaque architecture.
Reports retain logs, screenshots and GPU submission/pipeline diagnostics.

For regeneration and additional displayed modes:

```sh
python3 scripts/validate_indoor_web.py --url http://127.0.0.1:8765/ \
  --browser /usr/bin/google-chrome --seeds 0 6 --profiles auto portable \
  --modes Color Depth Normal Semantic Position --wasm-path www/out/viewer_bg.wasm \
  --human-density 0.6 --min-humans 1 --regenerate --output out/indoor_web
```

Use `--generator-version` when asserting a specific archived artifact's identity.
The emitted scene log supplies the actual generator version. Keep the Wasm hash,
configuration, adapter limits and capture evidence with each report.

Hardware browser checks have exercised headed Linux Chrome on NVIDIA Blackwell
with explicit WebGPU/Vulkan test flags. This does not qualify every browser,
mobile GPU, baseline-limit adapter or default browser policy. Headless Chrome on
this machine can return blank screenshots despite successful GPU submissions;
that result is a failed capture, not evidence of correct rendering.

Browser lighting, reflections, refraction, people and cloth remain approximations.
Passing these checks establishes the bounded display behavior exercised by the
report, not photographic realism, motion quality or browser dataset capture.
