# Indoor lighting and reconstruction coverage

Generator 122 meters a room once for the entire camera rig. The native lighting study used 48 consecutive Mixed rooms in two independent seed ranges, two pinhole cameras per room and 224 × 224 images. It includes 18 room profiles, native PBR, shadows, diffuse GI, default exposure variation and aligned annotations. Seeds 42000–42023 were used for development. Seeds 90100–90123 were reserved for evaluation. Every requested room is in the result.

## Rendered exposure

For each view, decode the exported sRGB PNG to linear RGB, compute Rec.709 luminance and take its median. Average the two view medians in linear light and apply the sRGB transfer function to express the result as a neutral gray between 0 and 1. A room below 0.25 is counted as dark for this diagnostic. The threshold identifies rooms for visual inspection. A representative real-photo corpus would supply an independent target distribution. Individual dark views and mixtures of bright windows and dim interiors remain valid training samples.

| Measured on 24 held-out rooms, 48 views | Input sampler | Updated sampler |
| --- | ---: | ---: |
| Rooms below 0.25 gray | 15 | 6 |
| Room gray, p10 / median / p90 | 0.104 / 0.193 / 0.434 | 0.201 / 0.317 / 0.459 |
| Pixels with linear luminance below 0.01 | 15.20% | 3.87% |
| Pixels with any channel above 0.99 linear | 0.079% | 0.132% |
| Highest single-view fraction above that channel threshold | 3.10% | 3.17% |

The paired room-level median change was +0.092 gray. The 95% percentile interval from 5,000 paired, room-level bootstrap resamples was +0.061 to +0.119. Of the 24 held-out rooms, 23 became brighter. In the independent development cohort, 10 rooms were below the threshold with the input sampler and three with the updated sampler. Across both cohorts, the count was 25 versus nine. The brightest updated view across all 96 images had 3.47% of pixels above the channel threshold.

### Where the dark bias came from

- The camera previously metered nominal electric output and a fixed fraction of outdoor daylight. A north-facing, heavily shaded opening could influence exposure much like a large window facing the sun. The new estimate uses clear window dimensions, obstructed area, facade orientation, solar elevation and the actual enabled ceiling emitters. Direct and diffuse daylight enter the estimate separately. The meter records all three estimated illuminance contributions and its exposure offset in the manifest. The estimates apply to the room as a whole. Local illuminance can vary across cameras.
- A bulb's old color conversion reduced actual luminance at warm and strongly tinted white points. Lumens and lux now retain their photopic scale when the light color changes. The visible emitter, reflection map, diffuse transport and direct light use the same color. Strong direct sunlight above 35° elevation and above 4,000 lx appeared in 13 rooms. The updated solar spectrum accounts for the shorter atmospheric path at those elevations. Warm low sun and tungsten-lit rooms remain in the sampler.
- The first enumerated ceiling fixture was always on. A lighting seed now selects the guaranteed operating fixture among primary-interior mounts. The normal operating distribution favors more active circuits and still includes patchy and dim rooms. The adjoining bay light samples a mount relative to the entrance or bay, with housing, roof and furniture clearance checks.
- Decorative wall paint formerly drew its base chroma from the furnishing palette. Feature-wall lightness and saturation now have their own draw while hue can coordinate with furniture. Broad pigment variations in ordinary field paint are weaker. Mineral, fibrous and printed feature finishes retain stronger variations. In the 48 rooms, the p10 / median / p90 of feature-wall base-color linear luminance shifted from 0.032 / 0.178 / 0.422 to 0.176 / 0.322 / 0.607. Archived material recipes contain their realized values.

An explicit illumination-scale intervention still dims the sources at a fixed exposure. An exposure intervention changes the rig's EV without altering its geometry, calibration or lamp flux. The default exposure response is deliberately partial, so a dim room retains a different appearance from a sunlit room. The stop convention follows Bevy's camera exposure model. [Filament physical camera model](https://google.github.io/filament/main/filament.html#imagingpipeline/physicallybasedcamera)

## Reconstruction and runtime checks

All 96 intrinsics and extrinsics matched across the two generations. Furniture and people matched in all 48 manifests. All 576 pairs of depth, normal, position, semantic, co-visibility mask and co-visibility validity PNGs were byte-identical at these views. The moved bay lights can legitimately affect labels when visible. Across the updated rooms, 31,602,607 audited triangles had zero degeneracies. The highest per-view 99th-percentile reprojection error was 0.000253 pixels. All 48 primary-room view graphs were connected at the 10% overlap threshold. Directed primary-room overlap p10 / median / p90 was 0.366 / 0.535 / 0.775. Corresponding percentiles of the per-pair median triangulation angle were 9.25° / 18.0° / 45.05°.

A separate CPU geometry audit of seeds 31000–31127 exercised all 20 Mixed room profiles and 384 fitted cameras. All 128 layouts passed. The audit checked 10,429 object bounds, 257 human bounds and 75,774,516 triangles with zero degenerate triangles. Across 933 interior chairs, the sampler occupied 390 width × depth bins at 1 cm resolution. The most frequent bin held 15 chairs. Six furniture kinds occurred in adjoining bays and ten bays were empty. Floor laying directions share an installation frame across slab patches. Object placement, floor mapping and chair family results are detailed in the [geometry and placement study](indoor_reconstruction_space.md).

The validator also exports per-view neutral-gray percentiles, deep-shadow coverage and channel clipping over calibrated sensor pixels. A separate six-view circular-fisheye smoke test counted 39,428 valid pixels of 50,176 per view. The six pinhole views with the portable quality profile counted all 50,176 pixels. Both used the native software adapter. Browser display output requires a separate runtime check.

The initial native qualification run found a renderer failure on seed 90108. Its 1,418,804 GI triangles required 136,205,184 bytes in one storage binding, exceeding the software Vulkan device's 134,217,728-byte limit. The manual validator continued advancing frames after a fatal render error and triggered a host OOM. The GI upload now uses five 16-byte vectors per triangle. The same records take 113,504,320 bytes, with the full original positions, UVs, face normals and material indices. Oversized scenes use the existing CPU integrator at the configured probe/ray/bounce budget. Upload also checks the actual adapter binding limits. Manually driven validation, benchmarking and capacity capture stop on renderer exit. The validator streams image statistics without making full-size float copies of every frame.

A serial, capped replay captured seed 90108 and the following rooms. On the two overlapping control seeds, all 32 image files from before and after the GI packing change matched byte-for-byte. A separate cold-start replay of seed 90108 matched all 16 PNGs from the warm 24-room run. Changing the lighting seed and adding 0.5 EV changed both RGBs and preserved all 14 other PNGs. Both final 24-room captures ran below 4.84 GiB of cgroup-accounted memory under an 8 GiB limit. Neither triggered a memory-limit event. The memory measurements cover these sample counts and this software adapter.

Bounded CPU preparation was measured in alternating before/after order twice. Each run used the same 32 rooms starting at 42000, three warm-ups, two room requests and four pinned CPU cores. The input generator delivered 1.496 and 1.525 prepared rooms/s. The updated generator delivered 1.514 and 1.528. These measurements include native scene and GI input preparation. Dedicated-GPU capture and concurrent training require separate benchmarks.

The core test suite passed 1,286 tests with 68 opt-in cases skipped. The six validator tests and the archived-manifest relighting test passed, as did strict Clippy, formatting and the WASM web-feature compile check.

## Remaining limits

| Observation | Consequence for reconstruction training |
| --- | --- |
| Nine rooms still have a low median, and individual windows can reach the tone-mapper shoulder. The meter approximates diffuse transport, shade transmission and lit floor area. Desk lamps, display emission and local interreflection can dominate a particular view. | Stratify evaluation by actual pixel brightness, clipping and shared visible geometry. Stronger exposure compensation alone cannot recover detail across an extreme within-view range. |
| The adjoining bay is attached to the same local wall and has a fixed 3.2 m depth. Human faces, hands and clothing remain coarse in close views. | Architecture and human silhouettes can still supply synthetic shortcuts. A connected floor-plan holdout and separate human-boundary reconstruction scores would test these effects. |
| People were visible in 42 of the 45 rooms containing them, given two views. | Scene-level occupancy overstates the available person supervision. Exported per-view semantic pixel counts expose this difference. |
| Geometric overlap and calibration were qualified on synthetic views. These experiments use software rasterization and a small room population. | Real-photo transfer, learning gains, photographic realism and hardware throughput remain separate evaluation questions. |

## Reproduction

Use the same command for each cohort and for both checkouts, changing only the first seed and the output directory. Preserve the raw display encoding when comparing levels. One renderer at a time avoids competition for RAM. On Linux with user systemd, the following cap includes the renderer and its worker threads.

```sh
systemd-run --user --scope -p MemoryHigh=5G -p MemoryMax=8G -p MemorySwapMax=0 \
  env LP_NUM_THREADS=3 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  indoor_validate --seed 90100 --audit-seeds 24 --audit-geometry --renders 24 \
  --cameras 2 --width 224 --height 224 --labels --co-visibility --no-raw \
  --output out/lighting-holdout
```

The input sampler was the local generator 121 worktree. The baseline for 90108–90123 used the full-precision packing repair, with the input lighting and material sampler. The original binary produced the other baseline rooms. No scenes were replaced after seeing their captures.
