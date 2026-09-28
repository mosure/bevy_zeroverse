# Project page: captures, paper and evidence

The static page is at `www/project/index.html`. Serve `www/` and open
`/project/`; the live Pages URL remains
<https://mosure.github.io/bevy_zeroverse/project/>. The existing GitHub Pages
workflow deploys the page and its checked-in media when `main` is pushed.

The old academic template's placeholder images, venue, arXiv link and unsupported
claims about trained-model transfer were replaced with actual outputs. The page
links a PDF compiled from `tex/bevy_zeroverse.tex`, a self-contained
LaTeX source ZIP, exact co-visibility NPZs, calibration JSON, and machine-readable
population metrics. Videos are H.264 MP4, with a smaller-resolution GIF download.
No external JavaScript, fonts, video service or PDF viewer is required.

## Evidence boundaries

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
- **Distributions and example cohort, v21:** all plots use the current 2,048-room
  population. Eight example RGB images are the first eight rendered rooms, without
  aesthetic filtering; provenance retains the original capture PNG hashes.

No new claims of photographic realism, unlimited memory stability, a ten-million
scene run, downstream pretraining gains or publication acceptance are made.

## Capture commands

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
python scripts/build_project_whitepaper.py
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

## Validation

The media builder checks every one of the 24 scene/camera/time identities across
eight modes: finite values, image size, normalized/pixel flow equivalence,
source-bit exclusion, depth/position reconstruction and camera reprojection.
Per-view errors are retained in the calibration downloads. Original source hashes, exact settings and per-view errors are
recorded in the two calibration JSONs. Both video files contain 120 frames at
20 fps with six-second duration.

`scripts/validate_project_page.py` tests all 64 room/camera/mode combinations at
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

Run `python scripts/validate_project_page.py --check-viewer --url <project-page-url>`
for release validation. This needs Chrome with WebGPU, Playwright, NumPy and
Pillow. The lightweight default page check explicitly reports
`viewer_runtime_checked: false`. `scripts/validate_project_viewer.py` can run
the viewer regression independently and preview local HTML against the live
renderer via `--project-html www/project/index.html`.

The TeX document builds with latexmk/pdflatex; the downloadable source ZIP
includes all figures, style and table inputs. `static/papers/provenance.json`
records the PDF and source hashes. The browser validation uses the compiled
file served by the same static site.
