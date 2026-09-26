# bevy_zeroverse for web

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
  --generator-version 4 --regenerate --output out/web_indoor_editor
```

`--url-only` preserves the URL and its defaults exactly. The old `cameras_x` and
`cameras_y` parameters are ignored; use `num_cameras=4&camera_grid=true` for a
four-camera grid. Browser checks retain console logs and canvas screenshots and
fail on inspector-registration warnings, missing picking, and browser/GPU errors.
