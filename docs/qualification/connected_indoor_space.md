# What the reconstruction cameras see in connected indoor rooms

An adjoining corridor changes the floor plan even when a camera sees only the closed door. A repeated lampshade can be visible from several cameras in the same training example. This study measures both the built scene and the geometry that actually reaches the sensor. It uses consecutive seeds with the Mixed room sampler, including cameras that see little beyond the primary room.

## Changes to the generator

The connecting wall used to open into a full-width rectangular bay exactly 3.2 m deep in every room. It now connects to either a full-width room or a room with a deeper return on the left or right. Depth, return width and return depth vary continuously. The slab can continue the primary floor finish or transition to concrete. The walls, roof and ground share the exported floor outline. Furniture and lights fit that outline, and the motion planner reserves the missing corner so a person cannot cut through the exterior wall. The primary camera target and reconstruction bounds retain their original meaning.

The shared partition now admits broad glazing, clerestories, sidelights and fully private walls with an entry. Panel placement references the actual door jambs. The same openings define the mesh cuts, glazing, wall backing and architecture export. Adjoining ceiling lights have their own circulation-area flux budget, separate from the primary fixtures. A small area or obstructed ceiling can legitimately have no ceiling-mounted source.

Exhibit pendants used to share one size. Table pendants and exhibit pendants now sample the diameter of the physical shade, suspension length, yaw and an anchor within the host footprint. Some are centered. Each product keeps its dimensions through bounded placement retries. Clearance checks cover the host, standing space, nearby fixtures and the local roof. Ceiling paint samples muted mineral tones and darker overhead finishes alongside the common pale stock. The change acts on the PBR surface and its reflected light.

Kitchen extraction now references the actual cooking surface of a range or separate cooktop. Opening an oven door changes its floor-space reservation. Its burners stay in place. The extractor uses the fixed burner geometry for centering, width and depth. The clearance above the cooking surface depends on the appliance, and the narrow flue meets the local ceiling over its own footprint. The model tests the canopy and flue separately against obstacles.

## Qualification protocol

The geometry cohort has 128 consecutive Mixed rooms, seeds 31000–31127, three cameras, density 0.65 and human density 0.25. A second cohort of 48 kitchens, seeds 0–47, exercises appliance mounts. All meshes in each geometry cohort pass through the ordinary validator. The matched RGB cohort has 12 consecutive Mixed rooms, seeds 90100–90111, two pinholes and 224 × 224 output. The additional mixed-lens cohort uses four cameras at 256 × 256 for seeds 31004–31006. Each render uses the native Bevy pipeline, the default two-by-two SSAA at these resolutions, shadows, materials and diffuse GI on Mesa software Vulkan. The control is the local generator 122 and the candidate is generator 123, capture-v149. Seed matching preserves inputs. Furniture and sightline changes can alter the cameras the planner admits.

Each view contributes its own denominator of valid annotated surfaces. For adjoining-space visibility, the validator counts a first hit at least 25 cm beyond the shared wall, within the local floor outline and below its roof, with a 3 cm boundary tolerance. An independent calculation decodes the raw world-position and co-visibility rasters. The glass policy affects which surface receives the annotation. This metric therefore measures training targets for geometry, even when RGB shows a room through transparent glass. Directed primary overlap restricts the correspondence calculation to the designated reconstruction region. Semantic pixel coverage uses the full sensor field.

## What changed in the sampled rooms

| Built feature | Control | Candidate |
| --- | ---: | ---: |
| Maximum depth of the adjoining floor | 3.20 m in every room | 2.43–7.50 m, 116 occupied centimetre bins |
| Adjoining outlines with a deeper side return | 0/128 | 49/128, with 23 right and 26 left |
| Adjoining floors with a different substrate | 0/128 | 39/128 |
| Shared-wall area occupied by rough window openings, p10 / median / p90 | 41% / 58% / 69% | 0% / 17% / 44% |
| Walls with an entry and no additional vision panel | 0/128 | 15/128 |
| Occupied adjoining rooms | 118/128 | 117/128 |
| Distinct pendant sizes at 1 cm precision | 22 sizes across 47 installations | 57 sizes across 57 installations |
| Pendants centered within 1 mm of a table or exhibit | 47/47 | 5/57 |
| Fitted hoods in 48 kitchens with 53 cooking surfaces | 7, all 62 cm deep | 35, with 18 distinct depths at 1 cm |
| HSL lightness of the sampled ceiling paint, p5 / median / p95 | 0.74 / 0.82 / 0.94 | 0.47 / 0.78 / 0.93 |

The most common control pendant measured 21 × 46 × 21 cm and occurred 26 times. Candidate shade diameters span 12–50 cm, with installed drops of 23–87 cm. Exhibit lamps sit up to 31 cm off the plinth center. Fitted hoods in the kitchen cohort had 62–89 cm clearance over the cooktop. These are realized dimensions. No extrapolation from occupied bins to an effective dataset size is warranted.

The 128 Mixed rooms covered all 20 sampled profiles and 384 planned cameras. The audit inspected 75,571,188 triangles, 10,470 object bounds and 256 human bounds. The additional 48 kitchens contributed 19,302,879 triangles. Both cohorts had zero rejected rooms and zero degenerate triangles. The maximum object overrun across both was 2.77 mm, within the existing 3 mm mesh tolerance.

## Visible training signal

| Capture cohort | Views | Directed primary overlap, p10 / median / p90 | Directions below 10% | Connected rooms | Median triangulation angle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired control, two pinholes | 24 | 36.6% / 48.3% / 68.4% | 0/24 | 12/12 | 26.0° |
| Candidate, two pinholes | 24 | 35.0% / 48.1% / 69.1% | 0/24 | 12/12 | 25.5° |
| Candidate, mixed four-camera rigs | 12 | 17.6% / 44.0% / 58.8% | 2/36 | 3/3 | 52.3° |

Triangulation is the median of the per-pair pixel medians. The mixed cohort included four fisheye cameras, three ceiling-mounted. The two weak directions had 6.9% and 7.2% primary overlap in room 31005. Across both candidate cohorts the largest per-view 99th-percentile self-reprojection error was 0.000214 px. The largest corresponding depth/position discrepancy was 1.91 µm. The independent adjoining-pixel calculation agreed with the Rust exporter within one pixel per view.

Across the 24 candidate pinholes, 0.32% of valid annotated first hits were in the adjoining room, against 0.31% in the control. One view exceeded 1%, reaching 6.36%. First hits in the mixed four-camera cohort were 0.58% adjoining. More distinct floor plans do little by themselves to increase the portion of a training loss that uses a neighboring room. The current rigs keep their emphasis on the primary reconstruction target.

Of 1,204,224 valid pinhole pixels, 614,687 were also visible to the other camera. The table reports what actually supports a two-view correspondence. Full images include valid surfaces outside the reconstruction target.

| First-hit class | Share of valid pixels | Share of shared pixels |
| --- | ---: | ---: |
| Ceiling | 25.1% | 25.5% |
| Floor | 22.5% | 21.7% |
| Wall | 19.4% | 21.0% |
| Window | 10.0% | 12.2% |
| Desk | 2.8% | 2.2% |
| Person | 1.4% | 1.1% |
| Chair | 1.3% | 0.4% |
| Plant | 0.4% | 0.2% |

The local validator now exports per-view `semantic_co_visibility` when both labels and co-visibility are requested. Each class has a count of valid source pixels, a count supported by at least one peer, and a vector indexed by peer camera. Unsupported valid pixels and invalid fisheye pixels remain distinguishable. This calculation reuses the semantic readback and co-visibility plane already requested by the caller.

The median of view-median display grays was 0.280 in the candidate and 0.288 in the control. Two views in each cohort had a median below 0.20. The candidate had 4.4% of pixels below linear luminance 0.01 and 0.05% with any display channel above 0.99, against 5.2% and 0.06% in the control. Some rooms are still dark or strongly color-cast. Room 90105 is a warm evening showroom with a mean display-channel ratio of 2.53 in camera 1. Human figures appeared in 9 of the 10 occupied rooms in both groups.

In an independent readback check over the 12 mixed-lens views, the class export matched all 338 class/view rows and all 1,352 directed peer counts. It included 758,296 valid surfaces and excluded 28,136 sensor or background pixels. The 72 raw RGB, depth, normal, position, semantic and co-visibility planes were byte-identical to the previous capture of the same candidate scenes. Of 6,369 hood pixels in this cohort, 5,385 had support from at least one other view.

## Performance, memory and tests

The preparation test uses three 512 × 512 views, four pinned CPU cores, four preparation threads, two rooms in flight and full-quality GPU material recipes. Each pass covers the same 32 seeds after three warm-up rooms. Rates were 1.61, 1.08 and 0.76 rooms/s in the control and 1.52, 1.51 and 0.96 in the candidate. The median rates were 1.08 and 1.51 rooms/s. The host was shared and individual passes varied considerably. These measurements cannot establish a production speedup.

A separate pinned software-Vulkan replay captured the same three four-camera rooms at 256 × 256, with full PBR and every requested geometry attachment. Capture times per room were 25.1, 18.6 and 15.1 seconds in the control and 17.8, 12.3 and 8.6 in the candidate. Across this small run that was 0.051 versus 0.078 completed rooms/s. These absolute rates apply to software rendering. Hardware GPU, Burn/JIT throughput under concurrent training, browser frame rate and long-run memory stability require separate measurements.

Each native rendering process ran alone under an 8 GiB hard cgroup cap with swap disabled. The 12-room candidate peaked at 4.17 GiB versus 4.33 GiB in the control. The separate paired mixed-lens processes peaked at 2.83 and 2.78 GiB. Neither capture cohort hit its memory limit or triggered an OOM kill.

All 1,292 enabled core tests passed across the full sweep and one targeted rerun. The sweep initially found a schematic test that expected four optional fixture types in one particular seed. The test now checks the reported room and real additional rooms until all four types have been exercised, retaining the assertions for each individual object's schematic. The full run skipped 68 opt-in tests. The separate twelve-room historical camera rejection cohort passed. All seven validator tests, strict native Clippy, formatting and the WASM web-feature compile check passed. Actual rendered ground truth, intersections with the roof, and floor/wall rays were checked independently in the mesh regressions.

## Remaining bottlenecks

1. **Shared foreground geometry.** Ceiling, floor, wall and window account for about 80% of the shared pixels. Chairs and plants together supply 0.6%. Evaluate a mixture of ordinary overview rigs and rigs selected for shared, depth-separated foreground before increasing object density. Measure the per-class support and triangulation jointly. Preserve a held-out ordinary camera distribution and include hard pairs when evaluating.
2. **Connected building layout.** The adjoining room still branches off the same local wall. The next architectural change with wider reach would sample the connection on different admissible edges, including bent sightlines from within the primary room. Count realized visibility and navigability for each connection type. Requiring every camera to point through the entry would create a new shortcut.
3. **Photographic appearance.** Even the new paint and lamp distributions leave large uninterrupted surfaces. At steep angles, wall and floor relief, contact shadows and reflection roughness matter more than extra finish names. The captured dark cafe and warm showroom are useful test cases. Compare material- and exposure-controlled renders with calibrated indoor photographs before claiming a real-domain match.
4. **People and rare equipment.** A high count of installed objects does not guarantee varied examples in shared pixels. Human clothing, hair silhouettes and poses remain visible generalization risks. Report their image-space coverage and per-view scale on a larger held-out cohort. Actual transfer requires an evaluation with true versus shuffled reference views on real posed images.

## Reproducing the measurements

Build the workspace's `indoor_validate` and `indoor_prepare_bench` with the `human_motion` feature. Static people are enabled by default in these cohorts. No learned trajectories are requested. Run the following commands in separate output directories for each sampler.

```sh
indoor_validate --seed 31000 --audit-seeds 128 --audit-geometry --renders 0 \
  --cameras 3 --width 256 --height 256 --output out/connected-population

indoor_validate --layout kitchen --seed 0 --audit-seeds 48 --audit-geometry \
  --renders 0 --cameras 2 --width 224 --height 224 --output out/connected-kitchens

indoor_validate --seed 90100 --audit-seeds 12 --audit-geometry --renders 12 \
  --cameras 2 --width 224 --height 224 --labels --co-visibility --output out/connected-captures

indoor_prepare_bench --seed 42000 --rooms 32 --warmup-rooms 3 --concurrent 2 \
  --preparation-threads 4 --output out/connected-preparation.json
```

The mixed-lens cohort uses the repository training rig fixture through `--indoor-camera`, with `--seed 31004 --audit-seeds 3 --renders 3 --cameras 4 --width 256 --height 256 --labels --co-visibility`. The local qualification directory includes the raw renders, metric exports and per-run memory counters. [Training rig fixture](../../src/scene/procedural_indoor/cameras/ceiling/tests/training_camera_settings.json)
