# Procedural indoor people

Indoor people use the `burn_human` AnnyBody reference surface with sampled
phenotype, pose and appearance. `--indoor-human-density` defaults to 0.25 and
accepts 0–1; zero omits both people and reference loading. The bundled reference
lives in `assets/burn_human`.

Static poses sample continuous limb, torso, stance and chair-relative parameters.
Placement checks furniture, partitions, walls, support surfaces, doorway approaches
and other people. Cameras and their complete paths avoid occupied collision
envelopes. The manifest records sampled appearance, pose, support and stable
instance identities for reproducible captures.

Clothing, hair, footwear, eyewear and facial details are built around the body
surface with separate material roles. Garment ease, folds, finish, pigmentation
and grooming vary procedurally. This is fitted geometry rather than cloth
simulation; hair, faces and silhouettes remain visibly synthetic.

## Optional motion

Static people remain the default. The `human_motion` feature enables opt-in ARDY
text/waypoint generation, batching, cached model loaders and scene-time playback.
A configurable fraction of people can move while others retain static poses.
Models are not initialized when motion is unrequested.

Routes are planned after furnishing and clips are checked before admission.
Unsupported floor-level transitions are blocked; actors that cannot be staged
onto suitable level ground remain static with a recorded rejection reason.
Capture waits for requested trajectories and uploads to become ready. See
[human motion](human_motion.md) for prompt sampling, inference settings,
seed behavior, native/browser operation and admission limits.

## Annotations and validation

Exports associate available world-space joints, bone metadata and object bounds
with stable human IDs. Temporal capture samples poses on the same timeline as
RGB and calibrated cameras; [flow](optical_flow.md) includes supported skeletal
and garment deformation. Padded records are excluded using exported counts and
instance IDs. Semantic person pixels do not establish per-person visibility.

The [architectural evaluation](architecture_v22.md) reports occupied-room
visibility in the latest rendered cohort. For current validation commands and
annotation conventions, use the [generation guide](procedural_indoor.md),
[dataset guide](procedural_indoor_dataset.md) and [motion guide](human_motion.md).
Natural motion, hand-object contact, cloth dynamics and photographic appearance
remain separate qualification questions.
