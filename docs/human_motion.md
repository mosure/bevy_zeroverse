# Optional indoor human motion

Motion uses the published `burn_ardy` 0.1.4, `burn_llama` 0.1.2,
`burn_human_inference` 0.1.4 and `burn_human_motion` 0.1.1 releases, including
ARDY and Llama's model-owned, digest-pinned loaders. No sibling checkout is
required. The companion Bevy plugin's studio UI
dependencies are optional, so zeroverse does not pull in its separate editor.
The [batch-fix upgrade review](human_motion_registry_upgrade.md) records the new
version requirements, serial/batch checks and registry transition validation.

```sh
cargo run --features human_motion --bin viewer -- \
  --scene-type procedural_indoor --indoor-seed 13 \
  --indoor-density 0.35 --indoor-human-density 0.7 --num-cameras 2 \
  --playback-mode ping-pong --playback-speed 0.167 \
  --human-motion '{"fraction":0.7,"max_actors":4,"frames":120,"batch_size":2}'
```

Omitting `human_motion` leaves people static. It creates no motion worker,
Burn device, model download, text encoding, or motion skin preparation. An
explicit policy with fraction zero and no trajectories also requests no
inference. Builds without the optional Cargo feature reject an enabled policy
instead of silently ignoring it. The `Human motion` inspector section exposes
the moving fraction, navigation fraction, actor limit and **ARDY generation
parameters** in feature-enabled builds: diffusion steps (1–10), clip frames
(40–640 at 20 Hz), history frames (0–160, multiples of four), text and trajectory
guidance (0–10), dense path conditioning, batch size and retry limit.
Defaults remain 10 steps, 80 history frames, text guidance 2 and trajectory guidance 3.
Ten steps is the published model's maximum; increasing guidance is not a general
motion-quality improvement. Clip duration and viewer trajectory traversal time
are separate: Sin/PingPong deliberately reverse clips, while Once plays forward.
**Play once at model speed (20 Hz)** restarts the active generated clip with a linear
forward timeline lasting `(frames - 1) / 20` seconds. It uses the active clip's
length, not unapplied policy edits. Default Sin playback at speed 0.2 traverses a
160-frame clip forward in 2.5 seconds (up to about 5× its generated speed), then
backward; that retiming should not be mistaken for poor model inference.
Generation settings apply only with **Regenerate / R**. Editing a motion-policy slider leaves the current
scene and inference job intact. Camera gizmos and bounding boxes have independent
checkboxes; rejected motion requests expose their reasons in the inspector.

The JSON policy also works in native/Python config and as a percent-encoded
`human_motion` browser query parameter. Build the WebGPU viewer with
`--target wasm32-unknown-unknown --no-default-features --features web,human_motion`.
Motion inference requires WebGPU; it does not run on WebGL2. Use HTTPS or
localhost. Model residency requires a capable GPU, as documented by burn_human.

## Planning and time

`human_motion::planning` is independent of rendering and model initialization.
The requested fraction is a deterministic population quota, capped by `max_actors`
(1–16). `locomotion_fraction` (default 0.85) stages that share of selected people
in walkable space after furnishing, before geometry, lighting and camera preparation.
It retains their identity, phenotype and outfit. Explicit actor trajectories keep
their original placement. Camera framing is recomputed for the staged population.
Routes reserve simultaneous occupancy instead of blocking the AABB of an entire
path. Camera trajectories are checked before inference as well as during admission.
Automatic walking requests condition standing pelvis height on the source rig;
otherwise the model can produce crouched walking despite horizontal waypoints.
It samples people deterministically from the scene seed, keeping unselected
people static. Behaviors include doorway entry/exit, room traversal, sitting,
standing up, gestures, stretching, balance, squats, knee raises, lunges, marching,
dancing, and crouching. Prompts combine actions with manner and posture cues;
paths, occupancy, dimensions, actor identity, and seeds vary independently.
Behavior availability depends on the actual free space and supporting furniture.

Prompt program 2 composes concrete actions with handedness, direction, repetition
and amplitude, plus at most one optional posture, arm or gaze modifier. The default
modifier probability is 0.35. It avoids stacked posture instructions such as
folded arms plus a stiff gait; the core action and navigation program retain
independent seeded choices. Its 40 action primitives span gestures,
exercise, dance, floor activities and idle motion. Travel additionally samples
walking, tiptoeing, shuffling, marching, backward walking, sideways steps, skipping
and jogging. Gaits are filtered by planned travel speed and headroom; backward
and sideways travel condition the corresponding body heading. Arm/gaze modifiers
are restricted to travel so they cannot contradict an action such as clapping.

`sequence_fraction` (default 0.45) proposes travel/action/travel programs, with up
to two distinct stops. Text describes the intervening travel. Action times,
repetition cycles, pelvis heights and clearances are sampled together; route
vertices survive frame rounding. Longer clips allow longer actions and more
combinations. `energetic_fraction` (default 0.15) proposes skipping/jogging where
speed and space permit. These are proposal probabilities, not admission quotas.
Pick-up actions do not attach props.

`prompt_sampling` controls relative family weights and optional modifiers:

```json
{
  "prompt_sampling": {
    "locomotion": 4.0,
    "gesture": 1.0,
    "exercise": 0.8,
    "dance": 0.5,
    "floor": 0.35,
    "idle": 0.5,
    "style_fraction": 0.35,
    "max_sequence_actions": 2
  }
}
```

Zero excludes a family, including its actions inside sequences. At least one
weight must remain positive. `locomotion` covers travel and supported chair
transitions; `locomotion_fraction` separately controls how many selected people
are staged in navigable space. A family is drawn per actor and kept through the
main geometry search, avoiding bias toward easy gestures on each retry. Failed
searches can fall back to another enabled family. Room geometry still changes
the final distribution. The inspector exposes these settings under **Motion
prompt sampling**; they apply only on **Regenerate / R**.

Each automatic `MotionPlan` exports a `prompt_recipe`: version, family, gait,
travel speed, modifiers and action phases with frame ranges, repetition counts,
amplitude, template bindings and clearance. Explicit user prompts are unchanged
and have no generated recipe. Sampled intent is distinct from measured action
fidelity in the generated clip.

Motion noise is sampled independently of appearance and prompt choice. A
dedicated ChaCha stream derives each ARDY request's full 64-bit seed from the
scene seed and actor ID. **Regenerate / R** advances the indoor scene seed, so
repeated prompts receive fresh noise even for a reused person. Replaying the
same scene seed and actor ID intentionally reproduces the request; batch size,
actor ordering and previous jobs do not advance a shared motion RNG. Explicit
trajectories use the same seed rule. Rejected-clip retries advance the request
seed deterministically, and accepted metadata exports the actual retry seed.
Only text embeddings are cached by prompt; generated clips are never cached by
prompt. Different noise can change motion without changing its requested
behavior or constrained route. See [seed validation](human_motion_randomness.md)
for real-weight comparisons and platform limits.

The [prompt grammar evaluation](motion_prompt_grammar.md) records text/token,
room-distribution and real-model checks. Use the [camera guide](multiview_cameras.md)
for primary-room trajectory and overlap controls.

Navigation uses swept clearance and A* around objects, columns, solid/glass
partitions, door frames, the open door leaf, and other people. Moving paths
reserve space against subsequent requests. The main doorway connects the
existing furnished neighboring room; walls are never treated as traversable
just because they are transparent. Explicit paths outside the generated floor
or through blocked space are errors.

Every clip is sampled from `Playback.mode.map_progress(Playback.progress)`.
Progress zero selects the first frame and one selects the last. This uses the
same easing and capture clock as camera trajectories. No accumulated animation
delta is used: pausing, jumping backward, and repeated capture times are stable.
`frames` sets source duration at 20 Hz; interactive `playback_speed` controls
how quickly normalized scene time advances. Looping can jump between endpoints;
use `once` or `ping-pong` when a discontinuity is undesirable.

Headless captures hold their render cameras inactive until scene construction,
required assets, indirect lighting and the applied motion policy are complete.
A motion report must be `Ready` for the current scene seed; completion of the
previous room cannot release a new capture. Render warmup then waits for mesh
and texture uploads, material preparation and shaders before issuing readback.
The sampler owns normalized time from the first frame (zero), so model loading
time never advances the recorded camera or human trajectories. Exported
`capture_readiness` records the scene seed and number of blocked updates.
Policies with no selected motion still require no motion models.

For a fixed-camera motion sequence, set `indoor_camera` to
`{"path_length_min":0,"path_length_max":0}`. This freezes both camera position
and orientation while people continue to follow the shared playback clock.

Explicit trajectories use stable manifest human IDs, scene-local metres, +Y
up, and frame-indexed waypoints. World rotation augmentation applies afterward.

```json
{
  "fraction": 0,
  "frames": 120,
  "strict": true,
  "trajectories": [{
    "actor_id": 42,
    "prompt": "A person walks forward calmly with a natural arm swing.",
    "waypoints": [
      {"frame": 0, "position": [0, 0, 0], "heading": 0},
      {"frame": 119, "position": [0, 0, 2], "heading": 0}
    ]
  }]
}
```

Replace IDs/positions with free space from the generated manifest. Root height
is unconstrained unless `constrain_height` is true. Custom prompts must fit
the upstream encoder's 64-token limit, including its prompt header.
For custom sit/stand prompts, the actor's assigned chair is treated as contact
furniture. `support_chair` can select another manifest chair ID; the generated
chair surface remains subject to penetration checks.

## Inference, cancellation, and geometry

One worker reuses ARDY, the text encoder, and a bounded 128-entry embedding LRU.
Native inference runs off the ECS thread. Browser inference yields between
model loading stages, text layers, and diffusion steps. Both wrap Bevy's
existing GPU device/queue. No separate graphics adapter is initialized.
Requests are conditioned in person-centred coordinates and rigidly transformed
back into the scene; model trajectory errors are retained for validation.
`batch_size` (1–8) groups complete autoregressive clips with independent prompts,
noise, and waypoints. Histories remain on the GPU and are bounded to 160 frames.
The dataset CLI defaults to one model-owning process when motion is requested;
an explicit `--workers` still controls concurrency. Batch actors within that
process before increasing process count, since processes cannot share model
weights. ARDY 0.1.4 pins its WGPU matmul, attention and reduction strategies to
address batch-dependent FSQ drift. Serial/batch equivalence is checked during
dependency upgrades; retain backend and model identities for seeded comparisons
across devices. Model downloads use upstream disk/CacheStorage caching and integrity checks;
`model_root` optionally points at a mirror with the same pinned bundle layout.

Regeneration supersedes queued work. The active job checks cancellation between
motion windows and actors; stale results cannot attach to a replacement scene.
An in-progress model download may finish populating its cache before cancellation
is observed. Models remain resident for reuse until the app exits.

Moving people retain Anny phenotype geometry, clothes, hair, eyes, and accessories.
Preparation creates an Anny rest surface, transfers its four strongest normalized
skin weights to garment seams/details, and retargets Core27 onto the full Anny rig.
Interactive rendering updates bones instead of regenerating meshes. Dataset and
O-voxel capture use the identical deformation on CPU to supply baked geometry to
the existing annotation pipelines. Changed meshes invalidate capture/voxel caches;
the 21 indoor annotation joints and person bounds update at the same time.
Motion candidates are omitted from static indirect-light bakes to avoid leaving
occlusion at their old positions. They still receive indirect lighting and cast
live direct shadows where the rendering profile supports them. A rejected
candidate stays out of that bake too; other static people remain included.

## Admission and evidence

Waypoint conditioning is approximate. Generated clips are checked before use:
finite coordinates, corridor deviation, floor/ceiling and obstacle intersections,
penetration of the actual generated support-chair mesh (12 mm contact tolerance),
camera contact, and overlap with other accepted actors. Pose interpolation receives
intermediate checks. Contact flags permit a
small vertical foot alignment adjustment; this is not a physics/contact solver.

Rejected clips receive at most `max_attempts` independent deterministic samples
(default 2, allowed 1–3). Retries reuse the models and prompt embeddings, and their
actual seeds appear in accepted request metadata. No root-path snapping is applied.
Rejected actors retain their staged static mesh. `strict:true` fails capture
instead. The capture metadata contains `human_motion` with accepted requests,
rejection reasons, batch count, embedding-cache hits, model-load count and pinned
model artifact hashes.
`HumanMotionClips` retains accepted source clips for explicit export.
These fields distinguish planned behavior from motion actually admitted.
The manifest retains the placement recipe. Read timestep pose tensors for moving
people's actual positions; manifest human positions and pose families describe
their original static placement.

```sh
# CPU manifest/navigation coverage, without model initialization
cargo run --features human_motion --bin motion_validate -- --seeds 128
# Also validate prompts with a local copy of the upstream Llama tokenizer
# (reads tokenizer metadata only; no model weights or GPU initialization):
cargo run --features human_motion --bin motion_validate -- --seeds 128 \
  --tokenizer /path/to/tokenizer.json --output out/motion_prompt_audit
# Real inference, five time samples, RGB/depth/normal/semantic and pose artifacts
cargo run --features human_motion --bin motion_validate -- \
  --seeds 0 --render-seeds 4 --output out/human_motion_render
# Verify the same native path with motion disabled
cargo run --features human_motion --bin motion_validate -- \
  --seeds 0 --render-seeds 1 --static-only --output out/human_motion_static
# Dataset generator; allow initial model loading time
cargo run -p bevy_zeroverse_burn --features human_motion --bin zeroverse_gen -- \
  --output out/motion_dataset --scene-type procedural_indoor \
  --samples 10 --workers 1 --timeout-secs 900 \
  --human-motion '{"fraction":0.4,"frames":160,"batch_size":2}'
```

Collision checks are conservative admission filters, not a guarantee of continuous
triangle-level contact correctness. Dense workstation layouts can reject many
sit/stand requests. Clothing uses skinning rather than cloth simulation; ARDY can
produce imperfect foot contacts or fail an action prompt. Photographic realism
and 10M-sample downstream learning utility are separate, unproven claims.

O-voxel export retains the existing dataset contract: one volume at the final
capture timestep; RGB/depth/normal/semantic and pose records cover every selected
time. Native capture is required for dataset readback. Browser support covers
planning, model loading, inference, dressed skinning, playback and regeneration.

See [measured integration validation](human_motion_validation.md) for artifacts,
coverage, admission rates and platform limits.

`motion_validate` writes `prompt_distribution.json` alongside its planning
report: family/action/gait frequencies, unique text and action-program counts,
sequence sizes, examples and optional token-length histograms. Action-program
counts deliberately omit wording, handedness, amplitude and timing. These counts
measure planned coverage, not rendered embedding diversity or model fidelity.

For signed per-pixel motion supervision, see [optical flow and motion vectors](optical_flow.md).
