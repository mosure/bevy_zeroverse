# Workstations, surface mapping and multi-view coverage

This qualification uses consecutive seeds for the mixed population and a separate OpenOffice population to study desk placement. Shape checks build the same meshes used by the renderer. The image checks use the native capture API, three calibrated cameras, RGB, semantic labels, depth, world position, normals and per-peer co-visibility.

## Changes in the scene generator

- Desk banks sample a heading and a lateral offset. Where depth permits, two desks can share a rear gap with their chairs on the outside. The planner accounts for both desks and the chair clearances before adding a row. Teaching stations face forward. A desk and its chair still have to pass placement together. Serialized furnishing fields without bank settings retain their station coordinates.
- Eligible desks can use telescoping T-foot and offset C-foot frames. Glides contact the floor, columns support the upper rails, and optional timber, felt or metal modesty panels attach to the crossbeam. The same component geometry supplies the chair and body clearance query. The metrics export records installed frames, panel dimensions, realized opposed desk pairs and their gaps.
- Turned objects, including planters, vessels and lamp bases, use a continuous cylindrical texture chart along the side and a planar chart on horizontal caps. Previously, the vertical texture coordinate was height plus radius. A shoulder that lost one centimetre of radius per centimetre of rise collapsed to a line in texture space. The new coordinate follows distance along the profile, so base color, roughness and normal maps retain two usable surface dimensions. This adds no textures or render passes.
- Mixed seed `31248` reproduced a folded rear skin on an adjoining-room chair. Its 44 mm pad was too thick for a normal offset at the tightest part of its lumbar bend. The rear skin now follows the forming axis of the shell, preserving the seated side and a positive front-to-back depth. The exact object seed `4589592075449161286`, adjacent product dimensions, closed rim orientation and metric rim UVs have regression coverage.

## Study design

| Population | Seeds | Observations |
| --- | --- | --- |
| Mixed layouts | 31000–31255 | 256 layouts, three cameras per room, full candidate mesh validation |
| OpenOffice | 57030–57125 | 96 layouts, three cameras per room, full mesh validation |
| Mixed captures | 44210–44221 | 12 rooms, 36 pinhole views per generator |
| Office captures | 57040–57042 | Three rooms, nine views per generator with a mixed room/ceiling and lens configuration |
| Independent office captures | 81000–81005 | Six rooms, 18 views per generator with the same mixed camera policy, chosen before inspecting these seeds |

Density was 0.65, human density 0.25, output size 224 × 224 and SSAA automatic. Layout and capture comparisons use the same seeds and settings. Placement and camera results may change when new furniture occupies a route. Every seed stays in the denominator. A connected graph uses a 10% overlap threshold. Per-class support counts each valid source pixel once if any other camera sees it. These are different measurements.

To repeat the population check, run `indoor_validate --seed 31000 --audit-seeds 256 --audit-geometry --layout mixed --density 0.65 --human-density 0.25 --cameras 3 --width 224 --height 224 --output out/mixed-study`. For a capture, use `--labels --co-visibility` and set `--renders` to the number of seeds. The office lens configuration uses the checked-in mixed training camera fixture with `primary_room=true` and `lenses.probability=0.5`.

## Realized space

| Main-room office measure | Input generator | Updated generator |
| --- | ---: | ---: |
| Desks across the 96 rooms | 818 | 838 |
| Rooms with an opposed pair | 2 | 12 |
| Desks in opposed pairs | 4 | 36 |
| Desks sharing a heading within 0.00001 radians with a peer | 46 / 816 | 4 / 836 |
| Desks with another desk's axis within one degree | 199 / 816 | 213 / 836 |

The heading and axis denominators include only desks in rooms with at least two desks. Facing one another counts as a shared axis. The median realized gap between opposed desks was 24 cm. Including adjoining spaces, the office cohort installed 471 telescoping frames and 256 modesty panels. Parallel purchases and facing workstations still occur in the updated space.

The 256 Mixed rooms covered all 20 profiles and 107 primary-room object categories. The 10th, 50th and 90th percentiles were 33, 60 and 126.5 main-room objects. The cohort included 206 concave footprints, 170 sloping ceilings and 10 mezzanines. Seven floor installations appeared. Straight and running-bond floors accounted for 159 rooms. Sampled floor roughness varied from 0.31 to 0.95 at the 5th and 95th percentiles. These counts describe sampled installations and material parameters. Camera-visible texture detail also depends on furnishing, projection and distance.

All 352 candidate rooms passed layout and mesh checks. The audit covered 211,209,087 triangles, 29,297 object envelopes and 729 human envelopes. It found zero degenerate triangles. The maximum object envelope overrun was 2.83 mm within the existing 3 mm trim tolerance.

## What the cameras actually saw

| Updated captures | Mixed, 36 views | OpenOffice, nine views |
| --- | ---: | ---: |
| Connected primary-room camera graphs | 12 / 12 | 3 / 3 |
| Directed primary-room overlap, 10th / median / 90th percentiles | 38% / 56% / 77% | 8% / 34% / 71% |
| Directed pairs sharing under 10% of the primary room's visible source pixels | 0 / 72 | 4 / 18 |
| Median wall, floor and ceiling share of the image | 66% | 75% |
| Selected furnishing, prop and person pixels seen from any peer | 66.6% | 51.3% |
| Chair pixels seen from any peer | 36.1% | 36.2% |

The office group contained seven fisheye and two pinhole views. Its baseline, camera mounts, room population and accepted framing differ from the mixed pinhole cohort. The comparison cannot isolate a lens effect. Across both cohorts, the worst per-view 99th-percentile reprojection residual was 0.00023 pixels or less, and the corresponding depth–position discrepancy stayed under 0.000004 m.

On the six independent offices, both generators produced eleven fisheye and seven pinhole views. Pooled furnishing, prop and person support was 63.5% in the input and 65.8% in the candidate. Chair support was 53.7% and 52.8%. Median directed primary-room overlap fell from 53.8% to 47.9%. All six whole-scene graphs connected in the candidate, and five of six primary-room graphs connected (six of six in the input). In seed 81004, cameras 1 and 2 each shared more than 60% of their primary pixels with the wider fisheye camera 0. Only 7.8% and 9.5% of camera 0's primary pixels were shared in the reverse directions. This illustrates why the report retains both directed denominators.

The input Mixed captures had 69.7% shared furnishing, prop and person pixels and 61.3% shared chair pixels. Changing the furnished training room at seed 44215 accounts for 6,638 of the 6,776 fewer shared chair pixels. The updated reception at 44221 shows 7,079 chair pixels, of which only 160 have a second view. Seven of the eight rooms containing people showed a person. A room-level overlap score alone hides these cases. The furnishing group here includes chairs, desks, tables, seating, storage, counters, displays, laptops, plants, sculpture, dishware, vases and people. It excludes building surfaces.

The 256 manifests sampled 132 overcast, 76 daylight and 48 evening rooms. The 36 mixed renders had a median image-midpoint neutral gray of 94.5 on the 0–255 sRGB scale. Their mean deep-shadow coverage was 0.82% of valid pixels (linear luminance below 0.01). At least one channel approached clipping in 0.55% of valid pixels. The small image sample warrants a larger lighting-stratified comparison before changing global exposure.

### Difficult views and the baseline control

Two additional three-view captures held all objects, humans and the envelope fixed. Only the baseline control changed. These two rooms were selected after inspecting the matched results, so they are diagnostics outside the population estimates.

| Seed | Chair pixels shared at baseline 0.5 | Chair pixels shared at 0.3 | Median pair triangulation at 0.5 / 0.3 |
| --- | ---: | ---: | ---: |
| 44215, Training | 44.5% | 44.2% | 14.1° / 6.9° |
| 44221, Reception | 2.3% | 3.4% | 41.6° / 42.6° |

The baseline controls the distribution of admissible viewpoints. Even in the reception, the resulting rig at 0.3 has a slightly wider median physical baseline for this seed. These observations favor a foreground-aware camera proposal experiment. A cheap score on shared support points at chairs, work surfaces and people could rank already-clear camera candidates. Test it prospectively across unseen seeds, with labels for per-class visibility, triangulation, negative pairs, admission failures and search latency. Retain appropriate occlusions and keep difficult rooms in the evaluation.

### Material and appearance limits

The sampled floor and paint roughness ranges are broad, yet whole rooms can still have a dominant color and a recognizable repeating floor finish. Seed 57042 illustrates this in both pinhole and fisheye views. A useful next comparison would capture the same geometry, camera and exposure while varying finish, wear and lighting independently, then compare the images against matched physical references. Human skin, garments and glazing also require close-range reference comparisons. This study establishes geometric and annotation behavior. Photographic realism and transfer to real reconstruction data still need separate measurements.

## Verification and cost

The native software Vulkan runs used Mesa llvmpipe with two renderer threads. The 12-room mixed capture peaked at 3.93 GiB and the three-room office capture at 3.89 GiB of memory charged to each run. Every build and capture ran in a scope with an 8 GiB hard cap and zero swap. No scope recorded an OOM kill. This covers the measured runs and gives no long-duration memory plateau estimate.

The independent office captures peaked at 4.06 GiB. An independent read of the exported 16-bit peer masks and validity planes covered 132 views and 6,623,232 pixels across both generators and the two baseline probes. Valid counts, per-peer counts and peer-count histograms agreed with the capture metadata in every view. Invalid pixels and each camera's own membership bit were zero throughout.

On one pinned CPU core, auditing the same 48 OpenOffice layouts, camera placements and emitted metrics took 5.64 and 5.56 seconds with the input generator and 3.82 seconds in each updated run. Order was input, updated, updated, input. Including full mesh construction and validation for the first 16 increased those times to 16.33, 10.90, 10.81 and 16.07 seconds in run order. The realized rooms differ between generators, so the timing combines implementation cost with the generated workload. In a separate full-native CPU preparation benchmark, 20 Mixed rooms with one request in flight and two preparation workers took 14.94 seconds after two warm-ups. The 12-room three-view software render took 109.95 seconds with the input generator and 111.15 seconds with the updated generator.

Six matched mixed rooms had identical camera and object manifests and contained no desks. Of their 903,168 source pixels, 60 changed semantic label, 136 changed quantized depth preview and 327 changed the exact peer mask. All co-visibility validity pixels matched. The revised backs can change physical geometry, and the new texture coordinates can change RGB with identical geometry. The independent primitive oracle checks the original lathe positions, normals and indices directly.

The native work included 57 focused tests and an additional opt-in regression spanning eleven historically difficult room seeds. Strict native Clippy, formatting and a WASM web-feature library build passed. The capture runs exercise the native renderer. Browser GPU rendering, dedicated-GPU training throughput and downstream model accuracy each require their own qualification.
