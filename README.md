# bevy_zeroverse for web

## Project page and whitepaper

`www/project/` is a self-contained static project page. It needs no npm build,
external font service or embedded GPU runtime. The existing Pages workflow
rebuilds and verifies it alongside the viewer. Keep media and the compiled PDF in the checkout;
deployment rebuilds HTML/PDF from the verified shipped measurements and runs no model inference.

Preview from the repository root:

```sh
python3 -m http.server 8770 --directory www
# http://127.0.0.1:8770/project/
```

The main page presents the **current architectural generator**: all 32 rendered
rooms, four synchronized views, both trajectory endpoints and six matched
channels, with per-room plans/sections and 512-room distributions. Feature
shortcuts expose mezzanines, stairs, floor levels, chamfers, cut-ins and arches.
All displayed RGB comes from actual native captures.

`project/reference.html` preserves the earlier camera-baseline sweep,
eight-mode annotation explorer, exact co-visibility masks, traversal video and
ARDY motion video. It labels their original capture scope explicitly. Those
studies are not measurements of the current architectural cohort.

The whitepaper's main figures show current architectural captures and metrics;
its reference appendix retains the older baseline and co-visibility evidence.
The PDF and complete LaTeX source archive are downloadable from both pages.

The page and paper are built by one Rust pipeline:

```sh
cargo run --locked -p bevy_zeroverse_publication -- refresh
cargo run --locked -p bevy_zeroverse_publication -- verify
cargo run --locked -p bevy_zeroverse_publication -- rebuild
```

Refresh validates and stages the current generator's complete captures, all six
annotations, figures, HTML, PDF and exact downloads. Rebuild uses the verified
self-contained shipped capture bundle without a GPU or raw `out/` files. The
Pages workflow rebuilds and verifies this bundle before deployment. Source or
version upgrades require fresh current captures; stale metrics fail the gate.

The recipe is [publication.toml](../publication.toml), and authored templates live
in [the publication crate](../crates/publication/README.md). See the
[publication protocol](../docs/project_page.md) for source identity, exact mask
formats, release requirements and separate browser/runtime qualification.

## wasm support

to build wasm run:

```bash
cargo build --locked --target wasm32-unknown-unknown --bin viewer --release --no-default-features --features "web,human_motion"
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
  --url 'http://127.0.0.1:8765/?scene_type=procedural-indoor&num_cameras=4&camera_grid=true&regenerate_ms=0' \
  --observe-seconds 10 --output out/web_default_editor
python scripts/validate_indoor_web.py --editor --seeds 6 --profiles auto portable \
  --regenerate --output out/web_indoor_editor
```

`--url-only` preserves the URL and its defaults exactly. Browser checks retain console logs and canvas screenshots and
fail on inspector-registration warnings, missing picking, and browser/GPU errors.

The [documentation index](../docs/README.md) describes current capabilities.
The [architectural evaluation](../docs/architecture_v22.md) supplies the latest
512-room structural audit and 256-view rendered cohort. The
[camera-baseline study](../docs/camera_baseline_v21.md) preserves its separate
2,048-layout / 512-rendered-room reference population. Rebuilding a report from
stored captures does not update the generator that produced them.
