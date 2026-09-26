# Procedural indoor people

Generator version 3 adds asset-free clothed adults. `indoor_human_density`
(`--indoor-human-density` in the native CLI) defaults to 0.25 and accepts 0–1.
Zero disables people. Human sampling uses a separate seeded RNG stream;
furnishings are unchanged when human density or camera count changes.

Chair occupancy is sampled independently, then checked against actual table
clearances, neighboring furniture, pillars, room boundaries, the door approach,
and existing people. Each scene has at most 16 seated adults and two standing
adults. Standing placement covers relaxed, presenting, and conversational poses;
seated poses vary torso lean and arms. Neighboring room chairs also participate.
Camera origins, swept trajectories, and viewing corridors exclude people.

Geometry uses shaped elliptical torso/head/limb profiles with articulated joints,
separate hands/fingers, facial details, ears, hair variants, shoes/soles/laces,
and clothing collars, lapels, cuffs, pockets and folds. Skin, lip, top, trousers,
hair, shoes, shirt, eye and detail materials are separate. An at-most-48-entry
material bank deduplicates palette variants across a scene and reuses generated
cloth maps. No person mesh or image files are loaded.

The manifest contains the stable instance ID, pose, chair support, stature,
build, shoulder width, palette/style choices, 21 local skeleton joints and
collision envelope. The capture exports world-space joint positions and
world-space bone-axis orientations, bone names/parents, stable human IDs, and
oriented boxes linked to the same IDs. Poses are static across the camera
trajectory. Chunk padding is excluded by `human_count` and uses `-1` instance
IDs. This is pose annotation, not a skinning rig or human-motion generator.

The asset-free people remain visibly procedural, particularly faces, hair and
cloth silhouettes. They are useful controlled room-scale semantic/pose content;
they are not a replacement for scanned people or learned human appearance when
human photorealism is the acceptance criterion. Facial expression, clothing
simulation, speech, hand-object interaction and interpersonal animation are not
implemented.

Validation includes topology/normals, actual geometry within the conservative
placement envelopes, floor contact, chair support, exclusion from obstacles and
camera trajectories, and deterministic replay. Across 128 fixed seeds at
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
