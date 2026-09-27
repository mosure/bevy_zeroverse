# Human motion integration validation — 2026-09-27 UTC

This is a local native/WebGPU integration qualification using real ARDY and Llama
weights, with AnnyBody geometry. The reviewed upstream baseline is
`d26d7a4afa3b8657e85687e83539cbbdbbfca1f0`; the local dependency tree includes the
batch API, optional studio dependencies and cooperative text encoding work.
Resolved versions are burn_human 0.5.1, bevy_burn_human 0.6.0, burn_ardy 0.1.3,
burn_llama 0.1.1, Burn 0.21 and Bevy 0.19.1. No model weights are added to zeroverse.

[Machine-readable results](human_motion_validation.json) contain seeds, policy,
rejections, timings and artifact hashes. Full local artifacts are in
`out/motion_final`, `out/motion_static_final`, `out/motion_browser_final` and
`out/motion_cli_gi_final`. Earlier `motion_native_round*` and `motion_browser*` directories
are diagnostic iterations, not the final admission results.

## Native coverage and admission

The CPU audit covers 128 consecutive manifests: 13 observed behavior categories
and 62 distinct prompts. Requests are deterministic and routes obey swept
clearance around furniture, solid/glass walls and the real doorway.

The real-model coverage sweep uses seeds 0, 1, 2, 13, 24, 36, 44, 54, 80, 82 and
96, with fraction 0.7, actor limit 4, 120 frames and batch size 2. These seeds were
selected for behavior coverage. Their acceptance rate is not an estimate for a
10M-sample distribution.

| Seed | Requested clips | Admitted behaviors |
| --- | ---: | --- |
| 0 | 2 | none |
| 1 | 2 | walk |
| 2 | 1 | enter |
| 13 | 2 | exercise, reach |
| 24 | 1 | leave |
| 36 | 4 | stand, stretch |
| 44 | 2 | turn |
| 54 | 1 | none |
| 80 | 2 | reach |
| 82 | 3 | sit |
| 96 | 1 | none |

Ten of 21 generated clips pass admission. Eleven fail: eight intersect the actual
support-chair surface, two fail floor clearance and one intersects another
furniture item. An additional 16 selected people have no feasible planned route
or activity. All rejected/unselected people remain static, including 43 static
people across this sweep. The behavior column records requested actions that
passed geometric/kinematic checks, not a human-rated semantic accuracy score.

The sweep captures 110 views at 384×288: two cameras and normalized times
0, 0.25, 0.5, 0.75 and 1. RGB, depth, normal and semantic attachments are present;
annotation buffers have the expected dimensions and finite values. Depth has
nonempty foreground. Maximum decoded normal-length error is 2.39e-7. Pose
annotations change exactly for admitted actors and remain identical for static
actors. Inspected native/browser frames show retained clothing, hair and
accessories on moving people. Some GI-disabled views, notably seed 36, are very
dark or poorly framed. Admission and valid annotations do not establish useful
human visibility or photographic quality for every camera.

The scene sweep disables indirect-light baking to isolate motion/capture work.
It loads one model pair across all 11 scenes, records 13 motion batches and five
embedding-cache hits. First capture takes 31.4 seconds; subsequent captures take
2.0–8.4 seconds, including geometry, admission and all image readbacks. These are
diagnostic times on a shared RTX PRO 6000 Blackwell 96 GiB, not isolated throughput
benchmarks. Two separate static captures report zero model loads and identical
poses at all five times.

## Browser and dataset export

The final WebGPU viewer is exercised in Chrome 153 on the same adapter. Static
seed 1 renders without any ARDY/Llama requests. Motion seed 1 admits walking;
pressing R generates seed 2 and admits doorway entry. There is one GPU device per
page, no WebGPU validation errors, changing rendered frames, and no model-manifest
reload on regeneration. The motion case, including initial loading and
regeneration, takes 127 seconds. The loader uses a local HTTP mirror with the
unchanged pinned digests. Headless Chrome produced blank screenshots on this
driver; the successful visual qualification uses a visible browser window.

[Walking frame A](evidence/human_motion/walking_a.png) and
[walking frame B](evidence/human_motion/walking_b.png) show the real browser
render. They are integration evidence; photographic realism is not established.

The production dataset CLI completes a separate seed-2 capture in a child
process, with three times, two cameras, all four image modes, indirect lighting
enabled at 64 rays/probe, and CPU O-voxel export at resolution 24. The compressed
safetensors archive is decoded independently. It contains finite image/pose
arrays, one admitted entering actor, two unchanged static actors, 3,073 occupied
voxels and 29 person-labelled voxels. The entering actor's maximum joint
displacement across the captured times is 3.29 m; static actors have zero
displacement. Existing object/camera/human CSV metrics and placement heatmaps
also export successfully. O-voxel represents the final capture time.

This final CLI run includes the indirect-light correction: planned motion actors
are omitted from static transport so their initial poses cannot leave stale
occlusion. Static people remain in the bake, and live direct shadows still use
the moving geometry. The exported metadata records this policy. A regression
test verifies that excluding motion candidates reduces bake geometry without
changing scene bounds or lights.

## Regression checks and limits

- Full library suite: 94 passed, three existing tests ignored.
- Final focused motion/GI suite: 18 passed, including custom contact furniture,
  deterministic planning, no-initialization paths, stale-result/error recovery,
  bone-length-preserving interpolation, conservative garment/hair bounds, GPU vs
  baked skinning conventions, O-voxel geometry invalidation, and static GI caster
  exclusion.
- CLI worker-default regression passed: motion defaults to one model-owning
  process; explicit concurrency and static policies preserve their choices.
- Strict workspace/all-target Clippy, the feature-disabled native build check,
  the WASM build and final WASM policy/GI checks pass. Cargo still emits the upstream
  burn-cubecl 0.21 future-compatibility notice.

Admission uses conservative proxies and sampled surface/intermediate-pose checks.
It does not establish exact continuous triangle contact, cloth dynamics, foot
locking or universal prompt fidelity. Tight sit/stand interactions are frequently
rejected. The model and embedding caches are bounded/reused by design, but this
11-scene run is not unlimited-process memory qualification. GPU/mobile/browser
coverage is limited to the tested adapter. There is no Cycles photometric
comparison or photographic-realism claim in this qualification.

## Reproduction

```sh
cargo test --lib --features human_motion
cargo clippy --workspace --all-targets --features human_motion -- -D warnings
cargo run --features human_motion --bin motion_validate -- \
  --seeds 128 --render-seed-list 0,1,2,13,24,36,44,54,80,82,96 \
  --output out/motion_final
cargo run --features human_motion --bin motion_validate -- \
  --seeds 0 --render-seed-list 0,1 --static-only --output out/motion_static_final
cargo build --target wasm32-unknown-unknown --bin viewer \
  --no-default-features --features web,human_motion
# Generate the JS wrapper using the Cargo.lock-matching wasm-bindgen CLI,
# serve the viewer and assets, and optionally expose a pinned local model mirror.
python scripts/validate_human_motion_web.py --headed --seed 1 \
  --url http://127.0.0.1:8789/ --model-root http://127.0.0.1:8789/models \
  --output out/motion_browser_final
cargo run -p bevy_zeroverse_burn --features human_motion --bin zeroverse_gen -- \
  --output out/motion_cli_gi_final --samples 1 --workers 1 --chunk-size 1 --seed 2 \
  --scene-type procedural-indoor --indoor-density 0.35 --indoor-human-density 0.7 \
  --indoor-gi-rays 64 --width 128 --height 96 --cameras 2 \
  --playback-steps 3 --playback-step 0.5 --render-modes color depth normal semantic \
  --ov-mode cpu-async --ov-resolution 24 --timeout-secs 900 --no-ui \
  --human-motion '{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":2}'
```
