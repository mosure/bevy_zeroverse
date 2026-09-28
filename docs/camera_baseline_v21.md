# Current camera-baseline evaluation (generator 21)

The continuous [camera baseline program](multiview_cameras.md) controls spacing
between views independently of camera travel length. This report measures the
current generator at five settings; the project page and whitepaper contain no
historical generator comparison.

## Protocol

- CPU audit: **2,048 consecutive seeds, 24000–26047**, baseline 0.5, four cameras.
- Render population: **512 distinct consecutive rooms, 24000–24511**, baseline
  0.5, four 320×240 cameras, normalized times 0, 0.5 and 1: **6,144 views**.
- Matched sweep: the first **128 rooms, 24000–24127**, at baseline **0, 0.25,
  0.5, 0.75 and 1**. The default subset is reused, yielding **1,024 total
  room/configuration captures and 12,288 views**, without counting reused views
  twice. Every level contains 1,536 views and 1,152 reference-pair/time observations.
- Mixed activity, density 0.65, static human density 0.25. Native Auto lighting
  and shadows; baked diffuse GI disabled for the parameter sweep. The separately
  selected project gallery uses full lighting with 1024 GI rays/probe.
- Every requested seed/time is retained. No retry with a replacement seed, removal
  of a dark scene, or overlap-based filtering. Geometry/material/person/lighting
  manifest hashes must match across all five settings for each seed.
- Four default capture processes and four additional setting processes, at most
  128 rooms per process. Run IDs and completion markers remain separate. A SHA256
  check guards the same capture executable throughout the run.

The renderer is native Vulkan on an NVIDIA RTX PRO 6000 Blackwell, with an
Intel i9-12900K host. Capture/export timing excludes process startup and includes
scene regeneration/readiness, all requested modes/times, validation and raw/image
writes. CPU validation and report work may run concurrently; these timings are
observations of this run, not isolated renderer throughput or a benchmark against
another release. Captures use the checkout-local wgpu command-cache/upload
optimizations documented in the [dependency review](wgpu_dependency_review.md).
Published crates resolve registry wgpu; these timing observations do not qualify
the unpatched package configuration.

<!-- RESULTS_START -->

| Baseline | Median room-mean reference distance, m (5th–95th percentile) | Mean reference overlap | Shared with any peer | Shared with all three | Proxy target misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.19 (0.11–0.29) | 72.84% | 92.08% | 61.02% | 177/1152 at 65.0% |
| 0.25 | 1.33 (0.87–2.05) | 55.20% | 85.18% | 39.88% | 96/1152 at 41.1% |
| 0.50 | 2.19 (1.49–3.24) | 46.51% | 80.67% | 28.40% | 52/1152 at 28.7% |
| 0.75 | 2.94 (2.23–4.77) | 38.94% | 76.89% | 21.96% | 22/1152 at 18.7% |
| 1.00 | 3.60 (2.80–5.93) | 32.86% | 72.78% | 17.26% | 11/1152 at 10.0% |

At the default baseline across **all 512 rooms / 6,144 views**, mean reference overlap is **45.69%** (minimum **15.95%**). **210/4608** pair/time observations miss the proxy target. **80.31%** of valid source pixels are shared with another camera and **27.93%** with all three. The denominator is **471,859,200** valid pixel observations.

There are **0** disconnected camera sets at a 10% edge-overlap threshold, **0** rooms with a view containing at most two semantic classes, and **0/357** occupied rooms without person pixels across their views. Maximum per-view p99 depth/position error is **0.00000381 m**.

The 2,048-room audit has **0 invalid seeds**, **508 numeric series**, and **406** with nonzero observed variance. The occupancy-signature estimate is **2050.2** for 2,048 rooms; it is an approximate HyperLogLog statistic, not an exact unique count. These measurements are correlated.

| Baseline | Median room-mean camera travel (m) | Capture/export seconds per room: median / 95th percentile |
| --- | ---: | ---: |
| 0.00 | 0.79 | 1.46 / 1.95 |
| 0.25 | 0.62 | 1.39 / 1.81 |
| 0.50 | 0.52 | 1.62 / 3.18 |
| 0.75 | 0.44 | 1.41 / 1.85 |
| 1.00 | 0.42 | 1.41 / 1.87 |

Timing uses the matched 128-room subset at each setting and the declared capture protocol. Travel bounds are separate controls, but joint collision/overlap rejection correlates the realized distributions. Wide spacing can favor shorter accepted routes; request a larger minimum travel when needed.

<!-- RESULTS_END -->

## Measurement definitions

**Camera spacing** is Euclidean distance from camera 0, sampled at 33 synchronized
times. The plot first averages these distances over the three reference edges
and times within each room, then plots the median and 5th/95th percentiles across
128 rooms. Quantiles select the nearest observed order statistic at
`round((N - 1) * p)`. Every exported group metric is independently recomputed from the
camera-path CSV. Minimum pair separation and minimum/maximum reference distances
are also retained, with spread and relative-motion scores.

**Reference overlap** is the smaller directional shared-pixel fraction of each
camera-0 pair at each captured time. Its denominator is all source image pixels.
The spacing/overlap plot aggregates overlap within each room before drawing
room-level quantiles. Table means cover 1,152 reference-pair/time observations
per setting. Views, pairs and pixel observations from one room are correlated.

**Co-visibility cardinality** counts how many of the three other cameras see the
same valid source surface: 0, 1, 2 or 3. Its denominator is valid source-pixel
observations over all views/times, not unique 3D points. Source-camera membership
is excluded. Exact membership counts, directed pair coverage, all-pair/reference
summaries, graph connectivity, proxy-target misses and worst-room tails are
included in the downloadable reports.

The independent diagnostic uses nearest-target-pixel depth agreement within
`0.01 m + 0.002*z`. The production GPU mask uses a separate symmetric tangent-plane
test; the gallery exports those exact masks independently. Neither counts
reflections or refractions: glass is annotation-opaque. The sampler's coarse
proxy test is not a rendered-pixel guarantee, so its misses remain in results.

**Diversity** includes zero-preserving object counts, intrinsics, photometry,
placement heatmaps, numeric/categorical distributions, pair correlations and an
occupancy-signature estimate. Nonzero marginal variance is not an independent
parameter count or a claim of ten million distinguishable useful samples.

## Reproduction

Human body assets must be available at `assets/burn_human`. Static scenes do not
initialize ARDY. Use a fresh output directory for every capture process.

```sh
cargo build --features human_motion --bin indoor_validate
for seed in 24000 24128 24256 24384; do
  audit=128
  if [ "$seed" = 24000 ]; then audit=2048; fi
  target/debug/indoor_validate --seed "$seed" --audit-seeds "$audit" \
    --renders 128 --cameras 4 --width 320 --height 240 --playback-steps 3 \
    --labels --no-gi --asset-root . --indoor-camera '{"baseline":0.5}' \
    --output "out/baseline_v21/captures/b050_$seed"
done
for pair in '0 b000' '0.25 b025' '0.75 b075' '1 b100'; do
  set -- $pair
  target/debug/indoor_validate --seed 24000 --audit-seeds 128 \
    --renders 128 --cameras 4 --width 320 --height 240 --playback-steps 3 \
    --labels --no-gi --asset-root . --indoor-camera "{\"baseline\":$1}" \
    --output "out/baseline_v21/captures/${2}_24000"
done
python scripts/build_baseline_evaluation.py --analyze
```

The shell loop above uses POSIX `sh` word splitting. Python analysis requires
NumPy and matplotlib. The builder verifies completed run identities, consecutive
seed sets, engine version, image sizes, camera policy and matched non-camera
manifests. It retains input hashes and per-view/pair/time CSVs under
[`docs/evidence/baseline_v21/`](evidence/baseline_v21/). The canonical
[`report.json`](evidence/baseline_v21/report.json) links each parameter level to its
full co-visibility report by SHA256; the `default/` report contains all 512 rooms.

Then follow [the gallery recipe](project_page.md) and rebuild project media and
the whitepaper. Checked-in figures and downloads require no models at site build
time. Raw float captures remain local; the reports retain original run IDs and
input hashes for provenance.

These are finite-sample geometry/coverage diagnostics. They do not establish
photographic realism, collision-free arbitrary human motion, unlimited-process
memory stability, or improved downstream pretraining accuracy.
