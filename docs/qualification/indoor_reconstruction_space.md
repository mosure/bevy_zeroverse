# Indoor scene diversity for multi-view reconstruction

## What the audit measures

A seed count says little about which features recur in training images. This audit follows the same 128 consecutive Mixed seeds (31000–31127) from the layout manifest into metric meshes, camera calibration and rendered pixels. All 20 room profiles occur. Density is 0.65, human density is 0.25, each rig has three cameras, and the image size is 256 × 256. The comparison uses generator 120 at commit `f14ca836d` and generator 121. The first six rooms supply the matched RGB and annotation sets. The seed and settings are paired. Changes to furniture, room dimensions and visibility can change the realized cameras.

Object counts below exclude supported props when describing adjoining floor furniture. They measure what was placed. A partition or blind can still hide those objects in RGB. Interior chair counts cover the full main interior, including its partitions. Rendered overlaps use only the designated reconstruction region. A directed overlap is the fraction of valid source pixels in that region also seen by one particular peer. Both directions of a pair count separately.

## Measured regularities and changes

| Feature in 128 rooms | Generator 120 | Generator 121 |
| --- | ---: | ---: |
| Interior chairs with an exact 68 × 68 cm footprint | 602 of 916 | 0 of 933 |
| Occupied chair width × depth bins, rounded to 1 cm | 160 | 390 |
| Chairs in the most frequent 1 cm bin | 603 (65.8%) | 15 (1.6%) |
| Rooms with an empty adjoining bay | 55 | 10 |
| Adjoining floor furniture kinds | 4 | 6 |
| Adjoining floor objects at exactly the same 180° yaw (within 0.00001 rad) | 254 of 254 | 0 of 347 |
| Rooms with a horizontal span at exactly 5.6 or 21 m | 8 | 0 |
| Bright diffuse daylight (>1,000 lx) with a weak direct beam (<500 lx) | 0 possible under the old sky equation | 17 |

**Seats and occupied space.** Frame width, depth and proportions depend on chair construction. Seats purchased for a table can match. Replacement chairs introduce within-set variation. Recliners and public seating keep their own geometry programs. Placement uses the realized envelope before checking rotation, tucking and collisions. The fixed seat datum still coordinates the chair mesh with the seated skeleton.

**Space visible behind partitions.** The adjoining bay now samples workstations, waiting seats, reading/storage areas and clear circulation for office and public programs. Desk and chair placements succeed as a group. The bay admits the actual rotated footprints, respects the door approach and the lowest overhead obstruction, and reserves a continuous 0.82–1.18 m passage. Storage, planting and seating have separate product and placement streams. A room can legitimately have no furniture in the bay.

**Floor material and room proportions.** Previously, changing the floor finish retained the same laying axes and texture origin. The floor now follows a room-level installation referenced to an actual envelope edge. The prior gives 70% of installations wall-parallel/perpendicular directions, 20% diagonals and 10% other obliques. A continuous phase in both texture coordinates prevents a joint from always landing at the room origin. Cut slab patches and adjacent horizontal surfaces share the same metric frame. Tangents are generated after mapping. Existing archived finishes without the installation field replay the original mapping. The ordinary room envelope also samples an aspect feasible for its drawn area, removing the old size-limit pileup. Teaching, kitchen and other program-specific size ranges still apply.

**Lighting.** Direct sunlight, diffuse skylight and electric lighting are separate capture factors. Cloud attenuation can remove the beam while retaining diffuse daylight. Exposure meters the combined daylight and electric contributions at scene level so cameras agree on the illumination. The audit exports direct and diffuse lux per room, their joint histogram, and the floor heading and phase. These are generator photometric parameters. A desk or camera receives additional attenuation from geometry.

The follow-up exposure study measures 48 rooms, including a held-out cohort, and reports the realized lighting and reconstruction checks for generator 122. [Indoor lighting and reconstruction coverage](indoor_exposure.md)

## Bottlenecks identified in this cohort

| Priority | Evidence and training consequence | Next experiment |
| --- | --- | --- |
| Connected architecture | In this cohort every adjoining bay was 3.2 m deep on the same local wall. Several public programs share the same room-size prior. | The connected-room study exercises variable-depth plans, partial glazing and measured visibility beyond the partition. Arbitrary connection faces and held-out building layouts remain open. [Connected rooms and visible surfaces](connected_indoor_space.md) |
| Surface variation at room scale | Floor direction now varies, but each finish still uses a finite atlas. Several walls have a broad mottled color field. From a distance, a room can be recognizable from the finish alone. | Add installation-scale wear near entrances, traffic lanes and furniture contact, with object-independent clean cases. Test material swaps on fixed geometry and inspect grazing-angle crops across mip levels. |
| People, furniture and clutter | Close-up views show coarse faces and clothing silhouettes. The seat-height datum and some furniture clearances remain strong metric priors. | Couple seat-height variation to feet, arm support and table clearance. Compare reconstructed human and furniture boundaries with the room background, including held-out product families. |
| Lighting and appearance | Native PBR has approximate indirect lighting. The six-room gallery includes both dim interiors and bright exposed surfaces. A diversity count cannot establish photographic realism. | Compare fixed geometry and materials against matched path-traced renders, stratified by weather, facade area and exposure. Evaluate whole scenes and separate surface crops. |
| Reconstruction supervision | A reference-connected rig can still include a poor peer pair. Small baselines can give high overlap with little parallax. | Report directed overlap jointly with triangulation angle, lens and baseline for each split. Evaluate the same target with its true reference, a shuffled reference and a masked reference. Compare on held-out real registered imagery. |

The last experiment tests whether a model uses another view when predicting masked content. CroCo provides the cross-view completion task. ScanNet++ provides posed indoor imagery and high-fidelity reference geometry for an independent evaluation. [CroCo](https://arxiv.org/abs/2210.10716) · [ScanNet++](https://openaccess.thecvf.com/content/ICCV2023/html/Yeshwanth_ScanNet_A_High-Fidelity_Dataset_of_3D_Indoor_Scenes_ICCV_2023_paper.html)

## Geometry, image and camera checks

The mesh audit constructed every room in the cohort. It checked 75,776,028 triangles, 10,429 object envelopes and 257 human envelopes. It found zero degenerate triangles. The largest object-envelope excess was 2.77 mm under the existing 3 mm trim tolerance. The largest human-envelope excess was zero. The audit exposed a narrow drawer pedestal in room 31088 whose caster extended 3.81 mm beyond its footprint. Axle positions now account for wheel radius across the production size range.

The captures use native Bevy PBR with shadows and Mesa software Vulkan. Pairs are evaluated at the captured timestep. Overlap percentiles pool all off-diagonal directed pairs. Triangulation uses the median angle among shared primary-room pixels in each unordered pair.

| Render cohort | Views | Directed pairs | Primary overlap p10 / median / p90 | Pairs below 10% | Connected rooms |
| --- | ---: | ---: | --- | ---: | ---: |
| Three pinholes, seeds 31000–31005 | 18 | 36 | 38% / 63% / 81% | 0 | 6 of 6 |
| Four-view training fixture, seeds 31004–31006 | 12 | 36 | 11% / 39% / 63% | 4 | 3 of 3 |

The second cohort contains four fisheye views, including three ceiling mounts. All four low-overlap directions were between two peers of camera zero. The weakest was 2.3% from the ceiling camera to the refrigerator close-up in room 31004. The pinhole cohort’s median pair triangulation angle was 16.5°. Across both cohorts, the worst per-view p99 self-reprojection error was 0.00017 px or less. The worst per-view p99 discrepancy between depth and camera-space position was under 3 μm. Relighting room 31000 changed all three RGB images. The depth, normal, position, semantic, co-visibility mask and co-visibility validity PNGs for those same views were byte-identical.

The eight-view regression in room 65573956 exposed another shortcut in camera *search*. Two fisheyes could individually share geometry with the reference only on the same path in the proposed set. The planner can now propose different time sections of a known navigable linear or cubic path. It rechecks the assigned lens, actual path length, synchronized spacing, physical clearance, relative motion and visibility before admission. The common placement path does no additional search. The regression exercises repeat generation of the entire rig.

The complete native core suite passed 1,279 tests with 68 opt-in cases skipped. It includes population coverage over 1,024 seeds and density-extreme tests across the room profiles. Two additional opt-in checks passed. One covered 11 rooms neighboring historical training failures. The other used a headless renderer to retain and replay a valid sample across two intentional scene rejections. Strict Clippy for the core library and its tests, formatting and the WASM web-feature build check also passed. Browser GPU behavior, CUDA training throughput, and real-data transfer are outside these measurements.

### Preparation throughput

Three matched 48-room sweeps per build used three 512 × 512 views, full native quality, GPU material recipes, four pinned CPU cores, four preparation threads, two rooms in flight and four warm-up rooms. The measured path includes layout, cameras, meshes and material/transport preparation. The generator 120 runs prepared 1.32, 1.34 and 1.22 rooms/s. The generator 121 runs prepared 1.44, 1.51 and 1.21 rooms/s. The median rates were **1.32 and 1.44 rooms/s**. Per-room p95 preparation times were 2.62 and 2.47 s, respectively, with 447 and 456 mean mesh batches per room. The last paired rates differed by 1.4% in the other direction. Host load varied across runs. These figures characterize CPU preparation on this host and leave end-to-end GPU/JIT throughput unmeasured.

### Reproduce the audit

```sh
cargo run --locked --features human_motion --bin indoor_validate -- \
  --seed 31000 --audit-seeds 128 --audit-geometry --cameras 3 \
  --width 256 --height 256 --renders 6 --labels --co-visibility \
  --no-raw --output out/indoor-space-audit

cargo run --locked --features human_motion --bin indoor_validate -- \
  --seed 31004 --audit-seeds 3 --renders 3 --cameras 4 \
  --indoor-camera "$(cat src/scene/procedural_indoor/cameras/ceiling/tests/training_camera_settings.json)" \
  --width 256 --height 256 --labels --co-visibility --no-raw \
  --output out/indoor-space-rig-audit

cargo run --locked --features human_motion --bin indoor_prepare_bench -- \
  --seed 31000 --rooms 48 --warmup-rooms 4 --concurrent 2 \
  --preparation-threads 4 --output out/indoor-space-preparation.json
```

The validator writes `generator_identity.json`, `scenes.csv`, `objects.csv`, `metrics.json`, per-seed mesh checks and per-capture reconstruction graphs. A targeted relighting run adds `--indoor-appearance '{"lighting_seed":5413}'` with the same seed, raster dimensions and camera policy.
