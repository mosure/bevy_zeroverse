# Primary-room cameras, furniture, annotations and motion — local v12 review

Generator 12 addresses annotation preview cost, camera placement, furniture
geometry, fixture annotations, motion prompts and overlapping architectural trim.
The [render gallery](evidence/room_review_v12/review.html),
[measurements](evidence/room_review_v12/summary.json),
[128-room distributions](evidence/room_review_v12/metrics.json) and
[placement heatmaps](evidence/room_review_v12/placement_heatmaps.svg) contain the
bounded evaluation. Work remains local; no release or CI run was requested here.

## Rendering and geometry

Annotation materials now share equivalent handles, preserving instancing across
mode changes and regeneration. Depth, normal, position and semantic materials
disable unused shadow/prepasses; optical-flow preview retains its motion prepass.
The cache is bounded. Pose gizmos use normal depth testing, so opaque walls occlude
joints just as they occlude camera and box gizmos. Native float32 capture and its
temporal flow contract are unchanged.

On an RTX PRO 6000 Blackwell/Vulkan, the matched two-camera benchmark measured:

| Preview | Before median, ms | After median, ms |
| --- | ---: | ---: |
| RGB | 2.74 | 2.98 |
| Depth | 65.78 | 2.22 |
| Normal | 67.16 | 2.20 |
| Position | 67.11 | 2.28 |
| Semantic | 69.06 | 2.13 |
| Optical flow | 70.64 | 2.12 |

This is device-synchronized wall-frame time: 576 repeated meshes, two 640×360
cameras, 100 warmup and 80 measured updates per mode, without capture readback.
It diagnoses the lost-batching regression; it does not establish throughput for
every dense, unique-mesh room. Full timing rows, including p95 and the return to
RGB, are in `preview_before.json` and `preview_after.json` beside the report.

Tabletops use continuous bevelled superellipse outlines, aspect ratios and taper:
circles, ellipses, rounded rectangles and rounded trapezoids. Independent supports
include splayed legs, A-frames, single/double pedestals, sleds and crossed trestles.
Props must fit the actual top outline. Chair programs add rounded seats, lumbar
curvature, shoulder flare, continuously varying back heights and optional supported
headrests. The audit includes 1,440 chairs, 219 with headrests; all five table
support programs appear. Existing frame and back constructions remain composable.

Each ceiling fixture exports a separate `lamp` box covering its housing, diffuser
and suspension, with lamp semantics on its parts. Architectural fixture boxes
currently have no manifest instance ID; their `instance_id` remains null.
Door/window spans leave small construction reveals, partition branches terminate
at wall faces, and glass safety bands stop before crossing posts. A 16-seed test
checks actual projected triangle overlap on coplanar trim surfaces; it exposed
and fixed jamb, partition-end and safety-band intersections.

## Primary room and trajectories

By default, all capture cameras and their entire paths stay in the largest
generated room zone. Neighboring rooms remain visible through openings/glass.
The policy is available in the inspector under **Capture camera paths**, in
viewer/dataset CLI JSON, and in serialized manifests:

```json
{"primary_room":true,"path_length_min":0.03,"path_length_max":8.0,"long_path_fraction":0.45}
```

Lengths are metres. `long_path_fraction` chooses longer route proposals; it is
not a guaranteed accepted proportion. Routes use bounded A* and swept clearance
against architecture, furniture and people, with checked corner rounding.
Waypoint paths advance at constant arc length. Short paths retain Bezier motion.
Explicit minimum/maximum lengths are enforced: infeasible constraints reject the
scene instead of silently relaxing them. Set `primary_room:false` to permit the
other generated zones. Changes apply on **Regenerate / R**, preserving slider use.

For example, request routes of at least two metres with:

```sh
cargo run --bin viewer -- --scene-type procedural-indoor --indoor-seed 44 \
  --indoor-camera '{"primary_room":true,"path_length_min":2,"path_length_max":8,"long_path_fraction":1}'
```

The consecutive 128-room audit has no invalid layouts. All 256 cameras start in
the primary room; default paths span 0.042–6.057 m, with 86 longer than 2 m.
Tests check complete sampled trajectories and routing around an obstruction.
The four stratified render scenes produced 24 aligned RGB/depth/normal/position/
semantic views, with 7–14 semantic classes per view. Across these views, worst
p99 depth/position disagreement is 2.4 micrometres, worst p99 reprojection error
is 0.0027 pixels, and maximum normal-length error is below 2.4e-7.

## Generated motion and validation limits

ARDY text and temporal waypoints now compose walking, a pause/action, and resumed
walking. Actions include picking up, crouching, waving, jumping, kicking,
falling/recovering, stretching and looking back. The action samples matching
pelvis-height conditions and requires local clearance and headroom. Stationary
prompts also include more energetic actions. `sequence_fraction` defaults to
0.45; `energetic_fraction` defaults to 0.15 for skipping proposals. Static scenes
still perform no motion-model initialization.

The 64-room planning audit found 318 feasible requests among 395 people. Planning
coverage is not successful model admission. Four fixed render seeds admitted
13/22 people. A separate seed-44 diagnostic, selected from the planning audit,
admitted 4/4 including **walk → pick up → walk** and **walk → crouch → walk**.
Their pelvis minima were 0.414 m and 0.452 m, followed by resumed standing and
travel. Object attachment is not implemented: the pick-up is a mimed action.

Across all five inference scenes, 17/26 people were admitted; 9 remained static
because no clear plan fit or generated motion failed collision admission. The
50 captured views exported all 244 expected object/person/fixture boxes, including
63 lamp boxes. Flow validation recorded 128,460 moving-person pixel observations,
zero missing expected static correspondences and worst per-view static p99
reprojection error of 0.0032 pixels. The four-room process loaded the motion
models once and reused them. Full prompts, rejections and annotation checks are
retained in the per-seed evidence directories; raw clips remain in `out/room_review_v12`.

Validation: 110 library tests passed, 3 long qualifications remained ignored;
strict workspace/all-target Clippy, native viewer build, formatting and diff
checks passed. The Wasm `web,human_motion` viewer compile check passed; browser
runtime was not requalified. Cargo still reports an upstream `burn-cubecl` 0.21
future-compatibility notice.

These checks establish specific rendering, geometry and annotation behavior.
They do not establish every prompt's action fidelity, photographic realism,
continuous physical contact, or 10M-sample downstream learning utility. Faces,
hair, clothing and some material patterns remain visibly synthetic.

## Reproduction

```sh
cargo run --example annotation_preview_bench -- out/room_review_v12/preview_after.json
cargo run --example review_furniture -- out/room_review_v12/furniture
cargo run --bin indoor_validate -- --audit-seeds 128 --renders 4 --cameras 2 \
  --width 512 --height 384 --human-density 0.25 --density 0.65 --labels --no-raw \
  --playback-steps 3 --stratified --output out/room_review_v12/rooms
cargo run --features human_motion --bin motion_validate -- --seeds 64 \
  --render-seed-list 0,1,13,34 --flow \
  --policy '{"fraction":1,"locomotion_fraction":1,"sequence_fraction":0.8,"energetic_fraction":0.2,"max_actors":16,"frames":160,"batch_size":4}' \
  --output out/room_review_v12/motion
# Same motion policy; diagnostic selected from the preceding CPU planning audit:
# --seeds 0 --render-seed-list 44 --output out/room_review_v12/motion_sequences
```

The native viewer is opened locally on seed 44 with that policy, two cameras,
ping-pong playback and automatic regeneration disabled.
