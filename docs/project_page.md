# Project page and paper publication

The canonical publisher is the workspace crate [bevy_zeroverse_publication](../crates/publication/README.md). One Rust pipeline owns capture, validation, figures, the unified page, measured paper text, source/mask archives and publication attestation. There is no ordered sequence of Python publishing scripts.

## Commands and release contract

```sh
# Full update from the current compiled generator. Reuse only compatible captures.
cargo run --locked -p bevy_zeroverse_publication -- refresh
# Explicitly request a fresh GPU run:
cargo run --locked -p bevy_zeroverse_publication -- refresh --recapture
# Read-only release/deployment gate; no GPU, raw captures, Python or LaTeX:
cargo run --locked -p bevy_zeroverse_publication -- verify
# Rebuild page and PDF from the verified shipped dataset on release/deployment:
cargo run --locked -p bevy_zeroverse_publication -- rebuild --release-version 0.26.0
# Sanctioned registry entry point, which prepares/verifies the page and paper first:
cargo run --locked -p bevy_zeroverse_publication -- publish --package bevy_zeroverse --dry-run
```

[publication.toml](../publication.toml) declares the capture protocol: consecutive seeds, audit/render counts, resolution, synchronized camera/time counts, densities and GI sampling. Its current recipe audits 512 room programs and renders all 32 rooms at four cameras and two trajectory endpoints: 256 identities, each with RGB, depth, normals, semantics, positions and co-visibility. No room is removed for appearance. The explorer, static fallback, matched figures, numeric counts and paper use this same cohort.

The generator records a build-time identity through the shared [capture contract](../crates/capture/README.md): crate version, geometry grammar and a digest of explicit renderer inputs, Cargo.lock, shaders and local WGPU patches. Publication compares that identity with the actual checkout, rather than reading a version number with a source-code regex. A version/dependency/source upgrade invalidates cached measurements even when the geometry grammar number is unchanged. This source identity does not promise bit-identical rasterization across devices or audit externally downloaded model assets.

Refresh requires a native graphics device, `latexmk`/`pdflatex` and Poppler `pdftoppm`. Rust image codecs, SVG rendering and a bundled licensed font produce figures without NumPy, Matplotlib, Pillow or system-font dependencies. External LaTeX typesets the document with shell escape disabled; Rust controls its source generation, dependency closure, failure checks, downloads and provenance.

All work is staged before installation. Missing modes, incomplete runs, camera/time duplication, wrong transforms/intrinsics, geometry/manifest disagreement, invalid annotations, corrupt exact masks, mismatched legends and histogram denominators stop the build. Paper warnings and broken page links stop installation too. File replacements have rollback backups and the attestation is installed last. A partial install cannot pass the release gate.

`www/project/publication.json` binds renderer source inputs, publisher/templates, capture hashes, all managed artifacts and explicitly separate reference studies. `verify` checks the complete artifact set and exact masks from the shipped ZIPs; it never needs the original machine's absolute paths or `out/` directory. Generated files should not be manually edited: edit the recipe/templates or generate a new measured cohort and run refresh.

The Pages workflow first runs Rust `rebuild` on Linux, uploads the validated page/paper/TeX bundle, then verifies that exact bundle before deploying it alongside the WebGPU viewer. The paper workflow uses the same builder. The publication-contract workflow runs regression tests, Clippy and verification on pushes, pull requests and releases. Source upgrades require a successful native refresh before those gates can pass; CI does not silently reuse stale scenes or initialize neural models.

For crate closeout, use `publish --package <name>` (omit `--dry-run` for an actual upload). It prepares and verifies the current page/paper before invoking Cargo. Publish the new `bevy_zeroverse_capture` dependency before dependent crates; `bevy_zeroverse_publication` is independently packageable. Direct external Cargo uploads cannot be intercepted and do not run repository workflows. Actual uploads retain Cargo's clean-checkout requirement; dry runs can package local work. The gate has no Cargo build-script publishing side effects.

## Viewer and annotation contract

Serve `www/` and open `/project/`. Room, feature, time, annotation and peer controls update one four-view explorer, plan and calibration. `/project/#explore` opens its co-visibility mode. Structural shortcuts use metadata with stable seed ties; RGB appearance never filters the cohort.

All six previews now use lossless WebP, without exposure correction or cropping. Depth is the clamped 8-bit display of axial metres/15; normals encode (n+1)/2; position is annotation-AABB normalized for display. Semantic colors and additive membership codes preserve their source pixels. These previews are not float32 training labels.

Co-visibility is the production GPU's same-time first-surface test. Bit i identifies ordered capture camera i; the source bit is excluded. The same additive legend applies to all views. Selecting a peer highlights shared surfaces in other cameras while its own tile stays RGB. Fractions use valid source pixels, rather than unique 3D points.

Every room's ZIP covers both times and four cameras. The global archive includes all 256 mask/validity pairs: **16-bit grayscale PNG membership** and **8-bit 0/1 validity**. A valid unshared surface has zero membership and validity one; background has both zero. Glass is a first geometric surface, so annotations do not follow reflected/refracted light. Calibration-and-program downloads contain all 512 serialized architecture programs and every rendered calibration.

The README's object and MatSynth grids remain. The project page retains the separate recorded camera-baseline/motion viewer, exact reference masks and videos with original identities. Those inputs are never promoted to current architectural measurements.

Optional browser qualification remains independent of the Rust artifact gate:

```sh
python3 -m http.server 8770 --directory www
python scripts/validate_project_page.py --url http://127.0.0.1:8770/project/
python scripts/validate_project_page.py --reference \
  --url http://127.0.0.1:8770/project/reference.html
```

The browser regression exercises all room/time/mode and peer selections, atomic switching, filters, keyboard controls, mobile/desktop layouts and the JavaScript-disabled fallback. Signal/geometry checks do not establish photographic realism, unlimited-process stability or ten-million-sample training utility.

## Reference study provenance

The studies below retain their original generator/capture identities on
`reference.html` and in the paper's reference appendix. They are independent
experiments; the main page uses one current 32-room capture set for RGB, geometry
and co-visibility. Rebuilding a recorded reference study must not replace the
current gallery, calibration, masks or population statistics.

- **Visual baseline comparison, v21:** the same two selected rooms at five
  baseline settings (0/0.25/0.5/0.75/1), four cameras, 768×480 and t=0: 40 views.
  Noncamera manifest fields match exactly within each room. Auto quality,
  1024 GI rays/probe, static people and lighting match the gallery recipe.
  The two b=0.5 groups reuse its first timestep. Camera poses **and intrinsics**
  are resampled by the policy; camera 0 can move. Plans use common room axes,
  footprint proxies and short view-direction wedges, not visibility frusta.
  The page's local spacing is mean C0-to-peer distance at t=0; shared visibility
  pools valid pixels from four exact production GPU masks. These selected
  examples are separate from the 128-room independent-depth sweep below.
- **Gallery, v21:** two selected rooms (24005 and 24000), four cameras, 768×480,
  normalized times 0/0.1/0.2, all eight render modes, static human density 0.25,
  furnishing density 0.65, default baseline 0.5, Auto quality and 1024 GI rays/probe.
  These are illustrations, separately identified from the consecutive population.
- **Traversal video, v21:** seed 24005, four cameras at 640×400, 120 synchronized
  samples from t=0 to t=1, encoded at 20 fps. The six-second visualization is a
  chosen playback duration; normalized camera time is not measured in seconds.
- **Motion video, v21:** seed 13, two cameras at 640×400, human density 0.7,
  furnishing density 0.35, 512 GI rays/probe. The request enables at most two
  moving actors, 120 frames, ten diffusion steps, 80 history frames and guidance
  2/3. Actual accepted trajectories, prompts, waypoints, model artifacts and
  settings are retained in gallery provenance. The video uses the model's 20 Hz
  generated rate without frame interpolation. It is not a motion benchmark.
- **Population and camera baseline, v21:** 2,048 consecutive layout seeds and
  512 distinct rendered rooms / 6,144 views at default baseline 0.5. A matched
  128-room subset at five baseline settings yields 1,024 room/configurations /
  12,288 views in total. All-camera co-visibility is independently measured from
  depth/calibration. Geometry, lighting, people and materials match across levels.
  See [protocol, denominators and limitations](camera_baseline_v21.md).
- **Distributions and example cohort, v21:** these plots use the recorded 2,048-room
  population. Eight example RGB images are the first eight rendered rooms, without
  aesthetic filtering; provenance retains the original capture PNG hashes.

No new claims of photographic realism, unlimited memory stability, a ten-million
scene run, downstream pretraining gains or publication acceptance are made.

## Reproducing the recorded reference studies

Run from the repository root with generator-21 source. Each output directory
must be fresh. Human body assets must be available under `assets/burn_human`;
optional motion models use the normal upstream loader/cache.

```sh
cargo build -p bevy_zeroverse_burn --features human_motion --bin zeroverse_gen

for seed in 24005 24000; do
  target/debug/zeroverse_gen --output "out/project_page_v21/scene_$seed" \
    --asset-root . --scene-type procedural-indoor --seed "$seed" \
    --samples 1 --workers 1 --cameras 4 --width 768 --height 480 \
    --indoor-density 0.65 --indoor-human-density 0.25 --indoor-gi-rays 1024 \
    --playback-steps 3 --playback-step 0.1 \
    --render-modes color depth normal position semantic optical-flow motion-vectors co-visibility \
    --ov-mode disabled --output-mode fs --color-codec raw --timeout-secs 300 --no-ui
done

target/debug/zeroverse_gen --output out/project_page_v21/traversal_24005 \
  --asset-root . --scene-type procedural-indoor --seed 24005 \
  --samples 1 --workers 1 --cameras 4 --width 640 --height 400 \
  --indoor-density 0.65 --indoor-human-density 0.25 --indoor-gi-rays 1024 \
  --playback-steps 120 --playback-step 0.008403361 --render-modes color \
  --ov-mode disabled --output-mode fs --color-codec raw --timeout-secs 600 --no-ui

target/debug/zeroverse_gen --output out/project_page_v21/motion_13 \
  --asset-root . --scene-type procedural-indoor --seed 13 \
  --samples 1 --workers 1 --cameras 2 --width 640 --height 400 \
  --indoor-density 0.35 --indoor-human-density 0.7 --indoor-gi-rays 512 \
  --playback-steps 120 --playback-step 0.008403361 --render-modes color \
  --human-motion '{"fraction":0.7,"max_actors":2,"frames":120,"batch_size":2,"prompt_sampling":{"style_fraction":0.35}}' \
  --ov-mode disabled --output-mode fs --color-codec raw --timeout-secs 600 --no-ui
```

After the completed baseline captures and selected gallery/video captures, run:

```sh
python scripts/build_baseline_evaluation.py --analyze
python scripts/build_project_media.py --captures out/project_page_v21
cargo run --locked -p bevy_zeroverse_publication -- refresh
```

The media builder consumes population CSV/JSON and the first eight captures from
`out/baseline_v21/captures/b050_24000`. The generator is not run during normal
website builds. Checked-in assets let the page deploy without local raw captures
or neural models. Display assets are beneath `www/project/static/media/`; the
PDF, provenance and dependency-complete source ZIP are under `static/papers/`.

## Display conventions

RGB WebP is encoded directly from raw sRGB float exports, without per-image
exposure normalization. Normal RGB is the stored `(n+1)/2` encoding. Position RGB
is the stored world-position normalization; out-of-AABB values are clipped for
display. Depth uses one viridis scale per room, fixed across cameras and times,
ending at the rounded-up 99.5th percentile of foreground depth. The UI declares
the range. Semantic and additive co-visibility PNGs preserve the export palette.
Individual peer previews highlight exact bit membership over dimmed RGB.

Flow uses hue for direction and saturation for magnitude, with fixed full scale
32 pixels per captured interval. Normalized motion vectors use full scale 0.05.
Both draw only correspondence-valid pixels; black means invalid and white means
zero motion. Last-timestep forward flow is invalid. The page labels the actual
source/target normalized times. Glass remains annotation-opaque, so geometric
annotations are not reflected/refracted optical content. Exact masks and
calibration are separate downloads from the display previews.

## Matched baseline illustrations

The `reference.html#baseline` module precedes the reference annotation explorer. Its five-stop slider
selects native captures of the continuous policy, without interpolating images.
Four views, the plan and metrics update atomically after image decoding; stale
requests cannot mix room or camera identities. RGB and shared-surface overlays
are available at every setting. The expandable contact sheet shows all four
cameras at b=0/0.5/1 together and remains usable without JavaScript. Each image
opens at its native resolution. The paper includes one full-page figure per room
and a subsection specifying controlled variables, measurements and limitations.

After capturing the default gallery above, capture the other levels into fresh
directories with the same generator-21 build:

```sh
for seed in 24005 24000; do
  for entry in 000:0 025:0.25 075:0.75 100:1; do
    tag=${entry%:*}
    baseline=${entry#*:}
    target/debug/zeroverse_gen --asset-root . --scene-type procedural-indoor \
      --output "out/baseline_gallery_v21/s${seed}-b${tag}" --seed "$seed" \
      --samples 1 --workers 1 --cameras 4 --width 768 --height 480 \
      --indoor-density 0.65 --indoor-human-density 0.25 --indoor-gi-rays 1024 \
      --indoor-camera "{\"baseline\":${baseline}}" \
      --playback-steps 1 --playback-step 0.1 \
      --render-modes color depth position co-visibility \
      --ov-mode disabled --output-mode fs --color-codec raw --timeout-secs 600 --no-ui
  done
done
python scripts/build_baseline_gallery.py
cargo run --locked -p bevy_zeroverse_publication -- refresh
```

The builder verifies all noncamera manifest fields, camera policy, capture size,
GI settings, source-mask exclusion and valid-pixel denominators. It also checks
depth/position consistency and camera reprojection for all 40 views (maximum
errors: 0.000025 m and 0.0058 pixels). Original source hashes and individual errors
are retained in [the illustration report](evidence/baseline_gallery_v21/report.json).
`static/media/baseline/calibration-and-masks.zip` contains scene manifests,
expanded settings, intrinsics/extrinsics, provenance and all 40 exact uint16 masks
with source-valid masks. Website RGB and teal overlays are display previews.
No per-image exposure correction or appearance filtering is performed.

The browser regression additionally checks 80 baseline-view/display combinations,
slider keyboard navigation, rapid room/baseline/mode changes, both contact sheets,
responsive layouts and a JavaScript-disabled fallback. Validation and screenshots
are retained in `docs/evidence/baseline_gallery_v21/`.

## Validation

The media builder checks every one of the 24 scene/camera/time identities across
eight modes: finite values, image size, normalized/pixel flow equivalence,
source-bit exclusion, depth/position reconstruction and camera reprojection.
Per-view errors are retained in the calibration downloads. Original source hashes, exact settings and per-view errors are
recorded in the two calibration JSONs. Both video files contain 120 frames at
20 fps with six-second duration.

`scripts/validate_project_page.py --reference` tests all 64 room/camera/mode combinations at
t=0, temporal endpoints, per-peer visibility, keyboard sliders, local links,
PDF signatures and actual MP4 decoding/playback. It also checks 390/768/1440 px
layouts and honors reduced-motion preferences. The browser evidence and final
result are recorded in `docs/evidence/project_page/`.

[Validation results](evidence/project_page/validation.json),
[desktop preview](evidence/project_page/desktop.png),
[matched annotation explorer](evidence/project_page/matched-annotations.png), and
[phone preview](evidence/project_page/mobile-390.png) are retained for review.
The linked browser report records the checked combinations, local links, MP4
decoding/playback, and browser errors for that validation run. The histogram and heatmap totals match
their declared denominators, including 33 path samples per capture camera.

The initial link check only verified HTTP responses. It missed a runtime error
in the viewer URL: `playback_mode=sin` was rejected by the released viewer's
case-sensitive enum decoder, which expects `Sin`. The corrected project-page
button uses `playback_mode=Sin`. A separate [click-through regression](evidence/project_page/viewer_link.json)
now checks the exact button navigation against the deployed viewer, including
the requested seed, scene readiness, GPU submissions, browser/GPU errors and
nonconstant images in all four capture-camera tiles.

Run `python scripts/validate_project_page.py --reference --check-viewer --url <reference-page-url>`
for release validation. This needs Chrome with WebGPU, Playwright, NumPy and
Pillow. The lightweight default page check explicitly reports
`viewer_runtime_checked: false`. `scripts/validate_project_viewer.py` can run
the viewer regression independently and preview local HTML against the live
renderer via `--project-html www/project/index.html`.

The TeX document builds with latexmk/pdflatex; the downloadable source ZIP
includes all figures, style and table inputs. `static/papers/provenance.json`
records the PDF and source hashes. The browser validation uses the compiled
file served by the same static site.
