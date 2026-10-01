# Project page: current captures, paper and evidence

The main page is `www/project/index.html`; serve `www/` and open `/project/`.
Its main examples use the current architectural generator, including real
mezzanine stairs, raised/sunken floors, polygonal footprints and sloping ceilings.
The page contains **all 32 consecutively rendered rooms**, four cameras, two
trajectory endpoints and five aligned modes: **256 camera/time views and 1,280
channel previews**. The population plots use the same **512 consecutive room
programs**. No rooms are removed for appearance.

The four feature shortcuts select seeds 7, 8, 6 and 2 by structural coverage.
They do not change population statistics. Room/feature/time controls update the
four views, calibration and plan together; the reveal slider compares each
annotation against RGB at exactly the same camera/time. The complete first-view
contact sheet and native-resolution images also work without JavaScript.

Plans use the captured manifest's polygon, window openings, floor patches,
pillars, portals and mezzanine stairs. Furniture uses solid footprint proxies.
Camera arrows use the recorded extrinsics; they indicate direction, not visibility
frusta. The adjacent section follows the indicated dashed line and shows the
actual envelope roof/floor profile, including stair treads where intersected.
It omits interior furniture, pillars and partitions. These are explanatory
**diagrams**, not an alternate renderer or additional captured rooms.

The paper's main evaluation and figures use this current cohort. The main page
also retains the [interactive co-visibility and matched-annotation explorer](../www/project/index.html#explore):
choose a source camera, additive camera membership or an individual peer, time,
and any of eight annotations. It opens in co-visibility mode, with a static
comparison, legend and exact-mask downloads available without JavaScript.
The annotation study keeps its original generator 21 capture identity; it does
not remeasure the current architectural cohort. Camera-baseline, co-visibility
and motion studies also remain on [`reference.html`](../www/project/reference.html)
and in the paper's reference appendix. Their geometry and independent denominators
remain explicit.
The current cohort did not export co-visibility, optical flow, motion vectors
or continuous video; those outputs must not be inferred from its display PNGs.

## Current capture and rebuild recipe

```sh
cargo run --bin indoor_validate --no-default-features --features multi_threaded -- \
  --seed 0 --audit-seeds 512 --renders 32 --cameras 4 \
  --width 640 --height 400 --labels --no-raw --playback-steps 2 \
  --gi-rays 1024 --output out/architecture_v22
python scripts/report_indoor_architecture.py \
  out/architecture_v22 docs/evidence/architecture_v22
python scripts/build_architecture_media.py
python scripts/build_project_whitepaper.py
python scripts/validate_architecture_gallery.py --static-only
python scripts/validate_project_page.py --url http://127.0.0.1:8770/project/
```

The media builder consumes completed, already validated engine captures. It does
not render new scenes. It rejects a generator identity older than the source
constant, mismatched input hashes, mixed run identities, wrong dimensions,
inconsistent geometry and missing cameras/modes. It preserves original PNG
hashes, matches plans to captured manifests and checks decoded-pixel equality
for all non-RGB PNG-to-WebP conversions. Updating the cohort size/protocol requires
updating the explanatory page/paper text; it cannot silently change denominators.

Assets are under `www/project/static/media/architecture/`. `gallery.json` records
source/output hashes, camera identities and display conventions; `population.json`
contains all audit histograms. Per-room JSONs and `calibration-and-programs.zip`
retain calibration, captured manifests and the 512 architectural programs.

RGB is the original tone-mapped sRGB, encoded at WebP quality 92, without crop or
exposure correction. Depth is grayscale `clamp(depth_metres / 15, 0, 1)`.
Normals encode `(n+1)/2`; positions are normalized by the annotation AABB and
clamped for display. Semantics preserve the captured palette. These **8-bit
previews are not numeric training labels**. Float32 annotation alignment was
checked during the original capture and is retained in the metadata.

The page and figures were refreshed from the existing current-generator audit;
this refresh did not run new GPU captures. Browser closeout passed all 320
room/time/mode combinations (1,280 views), feature filtering, rapid switching,
keyboard reveal, 390/768/1440 px layouts and the JavaScript-disabled fallback.
Static/source-hash checks and the browser results are recorded separately. This
page check does not qualify the linked WebGPU renderer or fresh scene generation.
See the [page-refresh validation](evidence/architecture_page/validation.json)
for exact check counts and the initial environment limitations.

The [gallery-restoration validation](evidence/page_restoration/validation.json)
checks the main-page co-visibility explorer alongside the architectural gallery:
80 matched mode cases, 72 individual peer selections, all 320 architectural
room/time/mode combinations, independent controls, keyboard reveal, three
responsive widths and the JavaScript-disabled comparison. It also checks the
original object-capture and MatSynth grids retained in the README, and a smoke
test of the reference page's annotations, baseline controls and motion video.

No photographic-realism, unlimited-memory-stability, ten-million-scene-run or
pretraining-gain claim follows from these artifacts.

## Reference study provenance

The assets below retain their original generator/capture identities. The matched
annotation explorer is available on both the main page and `reference.html`;
baseline comparisons and videos remain on `reference.html`. Rebuilding these
studies must preserve the main-page explorer and must not overwrite the current
architectural gallery, social preview or population figures. Architectural media
updates replace only the marked architecture block in the main page.

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
python scripts/build_project_whitepaper.py
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
