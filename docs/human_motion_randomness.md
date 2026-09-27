# Human motion seed validation

Identical prompts can produce different ARDY motion. The published
`burn_ardy` 0.1.4 initializes an independent full-width `u64` normal-noise stream
for each request and advances it through the autoregressive windows. Reusing a
Llama text embedding does not reuse that noise or a generated clip.

Zeroverse now derives motion seeds directly from `(scene_seed, actor_id)` in a
dedicated ChaCha8 stream, independent of appearance and prompt sampling. Both
automatic plans and explicit trajectories use this rule. Previously, motion
used the person's appearance seed; ordinary generated scenes already varied
that seed, but reusing an identity could also reuse its motion noise. The new
rule makes scene variation explicit even when the person, prompt and path are
held fixed. Random access by actor ID preserves seeds through reordering,
selection changes and unrelated failed plans.

**Regenerate / R** advances the indoor sequence from its base seed to the next
scene seed. Resetting the sequence to an earlier seed intentionally replays the
same requests. Rejection retries still use distinct deterministic seeds, and
accepted request metadata records the seed actually used. Noise is not drawn
again during playback or capture. This preserves repeatable annotations at a
given time. Motion-disabled scenes retain the existing zero-model-load path.

This changes generated motion for existing scene seeds. Capture identity is now
`capture-v19`, with `ardy_motion=5`, to prevent silently mixing resumed captures
from the older seed policy. The preceding
[prompt grammar evaluation](motion_prompt_grammar.md) used `capture-v18`.

## Native real-weight check, 2026-09-27

The [regression](../tests/human_motion_randomness.rs) loads the published models
once and encodes two fixed prompts once each: walking with natural arm swing,
and standing while waving the right hand. Each request has 160 frames and fixed
waypoints, heading, pelvis height, diffusion steps and guidance. The seeds test
adjacent scenes, different actors, and a change only in bit 63.

The first eight-request batch includes six distinct prompt/seed combinations
and two duplicates. A reversed batch and a serial request test replay without
depending on batch position or prior inference. Comparisons inspect decoded
frames and forward-kinematic joint positions; seed strings and provenance are
excluded from motion-difference evidence. Articulation measurements remove
both root translation and orientation before comparing joints.

| Check | Result |
| --- | --- |
| Same prompt and conditioning, different seed | All six pairs differed |
| Root-local joint RMS difference, walking | 6.72–10.59 cm |
| Root-local joint RMS difference, waving | 9.60–16.80 cm |
| Frames with more than 1 mm articulation RMS difference | 160 / 160 for every pair |
| Identical requests in different batch lanes | Both matched exactly |
| Reversed-batch replay | All eight matched exactly |
| Serial versus batched replay | Exact match |
| Model pair loads / text encodes | 1 / 2 |

The device was NVIDIA RTX PRO 6000 Blackwell, Vulkan, driver 610.43.02.
The first eight-clip batch took 5.82 seconds; the bounded qualification took
29.78 seconds including cached model loading, encoding, replay and comparisons.
These are diagnostic timings, not a throughput benchmark.

The [machine-readable report](evidence/human_motion_randomness/summary.json)
records complete requests, model hashes, frame-content hashes and every
comparison. [Dependency resolution](evidence/human_motion_randomness/dependencies.json)
confirms the published ARDY 0.1.4, Llama 0.1.2, motion 0.1.1 and inference 0.1.4
crates. Source clips remain locally at `out/motion_seed_validation/clips.json`.

CPU tests additionally check 65,536 scene/actor pairs without seed repeats,
high-bit sensitivity, preserved conditioning when only the scene seed changes,
and the actual runtime queuing path across scene seeds 0 → 1 → 0. All 19 motion
unit tests passed, including the existing disabled-motion startup checks.
Strict Clippy and the `web,human_motion` Wasm viewer compilation passed. Browser
inference was not rerun; cross-device bitwise agreement is not claimed. The
existing upstream `burn-cubecl` future-compatibility notice remains.

Different seeds sample variation within the requested behavior and constraints;
they need not produce a different action class. This small check establishes
seed sensitivity and replay on this native backend. It does not prove unique
output for every possible seed, action fidelity, or post-admission diversity
over 10M scenes. Collision rejection can still retain a static person.

## Reproduction

```sh
cargo test -p bevy_zeroverse --lib --features human_motion --locked human_motion

# Explicit opt-in: loads cached/downloaded model weights and needs a native GPU.
ZEROVERSE_MOTION_SEED_OUTPUT=out/motion_seed_validation \
  cargo test -p bevy_zeroverse --features human_motion --locked \
  --test human_motion_randomness -- --ignored --nocapture

cargo clippy -p bevy_zeroverse --all-targets --features human_motion --locked -- -D warnings
cargo check -p bevy_zeroverse --target wasm32-unknown-unknown \
  --no-default-features --features web,human_motion --bin viewer --locked
```
