# bevy_zeroverse for web

## Project page and whitepaper

`www/project/` is a self-contained static project page. It needs no npm build,
external font service or embedded GPU runtime. The existing Pages workflow
copies it alongside the viewer. Keep media and the compiled PDF in the checkout;
deployment does not regenerate the scientific figures or run model inference.

Preview from the repository root:

```sh
python3 -m http.server 8770 --directory www
# http://127.0.0.1:8770/project/
```

The page includes a two-room/four-camera/eight-mode annotation explorer, a
pixel-aligned RGB reveal control, per-camera co-visibility membership, exact
NPZ visibility downloads, 120-frame multi-view and ARDY videos, a GIF, a
compiled technical whitepaper and its complete LaTeX source archive. Native
captures supply every render; no generated or stock images stand in for outputs.
Figures preserve the v18 audit's population and scope. Selected v19 gallery
rooms do not constitute a repeated population evaluation.

For the exact capture commands, data provenance, display mappings and browser
checks, see [the project-page review](../docs/project_page.md).

To rebuild assets after regenerating those captures, install NumPy, Pillow,
Matplotlib and safetensors in a Python environment, plus FFmpeg, latexmk/pdflatex
and Poppler on the host, then run:

```sh
python scripts/build_project_media.py
python scripts/build_project_whitepaper.py
python scripts/validate_project_page.py --url http://127.0.0.1:8770/project/
```

The media builder checks capture calibration, per-mode dimensions, camera-bit
exclusion and optical-flow unit conversions. It verifies the first eight audit
images against the recorded SHA-256 identities. The paper builder includes
source and PDF hashes. Browser checks require Playwright and Chrome; they cover
every room/camera/mode at t=0, temporal controls, peer masks, keyboard reveal,
local downloads, both videos, reduced motion and mobile layouts.

## wasm support

to build wasm run:

```bash
cargo build --locked --target wasm32-unknown-unknown --bin viewer --release --no-default-features --features "web"
```

to generate bindings:
> `wasm-bindgen --out-dir ./www/out/ --target web ./target/wasm32-unknown-unknown/release/viewer.wasm`


open a live server of `www/index.html`

The `web` feature includes the viewer's reflection registration and picking
support, which the default editor requires. Keep the editor enabled in browser
startup checks: disabling it skips inspector initialization.

With Playwright, Pillow, and a WebGPU-capable Chrome installed, validate both the
default demo URL and an indoor scene with the inspector enabled:

```bash
python scripts/validate_indoor_web.py --url-only \
  --url 'http://127.0.0.1:8765/?yaw_speed=0.7&cameras_x=2&cameras_y=2&regenerate_ms=8000&plucker_visualization=true' \
  --observe-seconds 10 --output out/web_default_editor
python scripts/validate_indoor_web.py --editor --seeds 6 --profiles auto portable \
  --generator-version 8 --regenerate --output out/web_indoor_editor
```

`--url-only` preserves the URL and its defaults exactly. The old `cameras_x` and
`cameras_y` parameters are ignored; use `num_cameras=4&camera_grid=true` for a
four-camera grid. Browser checks retain console logs and canvas screenshots and
fail on inspector-registration warnings, missing picking, and browser/GPU errors.
