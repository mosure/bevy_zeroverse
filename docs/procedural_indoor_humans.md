# Procedural indoor people

Generator version 6 uses the real `burn_human` AnnyBody reference surface, including
its phenotype blendshapes and all skinning influences. `indoor_human_density`
(`--indoor-human-density` in the viewer CLI) defaults to 0.25 and accepts 0–1.
Zero disables people and reference loading. The reference files are in
`assets/burn_human`; furnishings and their textures remain procedural.

The reference loads asynchronously. Native scene preparation retargets surfaces on
the compute pool and retains a bounded 24-person geometry cache for render/GI reuse.
Dataset capture waits until preparation completes. On Wasm the same geometry code
runs cooperatively on the browser thread; this is not a threaded Wasm implementation.

Chair occupancy is checked against furniture, partitions, walls, door approaches
and other people. Working poses require a nearby surface facing the chair. There
are three seated and five standing activity labels. Each samples continuous end-effector,
stance, torso and swivel parameters; fixed-length two-bone IK solves each limb.
These labels are not eight stored skeletons. Neighboring room chairs participate. Cameras and paths exclude people.

Garment shells, hair and eyewear are procedural additions to the Anny surface.
Material groups preserve skin, lips, top, trousers, hair, shoes, eye and detail roles.
Continuous appearance parameters control pigmentation, garment ease, fold amplitude,
cloth roughness and combed hair geometry. Garment displacement retains shared seam
positions, and normals are recomputed from the displaced surface.
This is approximate clothed geometry, not cloth simulation or scanned appearance.

The manifest contains the stable instance ID, pose, chair support, stature,
build, shoulder width, palette/style choices, 21 local skeleton joints and
collision envelope. The capture exports world-space joint positions and
world-space bone-axis orientations, bone names/parents, stable human IDs, and
oriented boxes linked to the same IDs. Poses are static across the camera
trajectory. Chunk padding is excluded by `human_count` and uses `-1` instance
IDs. This is pose annotation, not a skinning rig or human-motion generator.

The people remain visibly procedural, particularly faces, hair and
cloth silhouettes. They are useful controlled room-scale semantic/pose content;
they are not a replacement for scanned people or learned human appearance when
human photorealism is the acceptance criterion. Facial expression, clothing
simulation, speech, hand-object interaction and interpersonal animation are not
implemented.

Current validation includes topology/normals, closed clothing material boundaries,
actual Anny surface geometry within the conservative
placement envelopes, floor contact, chair support, exclusion from obstacles and
camera trajectories, and deterministic replay. Current evidence is recorded in
the [local scene review](local_scene_quality_review.md).

Historical generator-v3 validation used the previous primitive people, not AnnyBody.
Across 128 fixed seeds at
furnishing density 0.65, human densities 0/0.25/1 produced 0/361/1354 people;
all five poses, three outfits, eight skin tones and six hairstyles appeared.
Furniture remained identical across density settings. The native dataset
qualification additionally verified every world joint against the manifest plus
world augmentation, with maximum absolute discrepancy 1.09 micrometres across
five exported scenes, and exact static poses over all three camera steps.

Evidence: `out/indoor_human_final_review` contains the final native 960×720 room render;
`out/indoor_human_visual_review` contains the initial four-scene visual audit;
`docs/procedural_indoor/dataset_qualification_v3.json` records indexed capture,
worker/resume and pose correspondence checks. The broader CPU distribution and
browser reports cover version 3 human placement as part of each room.
