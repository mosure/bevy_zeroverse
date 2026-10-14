# Human pose capture

Indoor static people use the Anny body and fitted garments. Seeded continuous
programs sample arm reach, elevation, sweep, elbow swivel, stance, torso twist,
head yaw/pitch/roll, finger flexion and index extension. Activity names bias these
ranges; they are not a list of stored skeletons. Either hand can lead a gesture,
and the other can participate. Seated work retains palm/table contact fitting;
conversational gestures try free targets above nearby worktops when necessary.
Placement still checks room, furniture, supports, other people and camera space.
No motion model is loaded for static pose generation.

Capture automatically exports these labels alongside object boxes:

- `Sample::human_instance_ids` identifies people; indoor IDs match the manifest
  and `object_obbs[*].instance_id`, including neighboring-room people.
- `human_pose_steps[t][person]` contains realized world-space joint positions
  (metres) and XYZW rotations. `human_poses` aliases the first timestep.
- `human_bone_names` / `human_bone_parents` define the joint order and hierarchy.
  Indoor people have 21 canonical joints. These labels follow the fitted,
  grounded body and motion updates, rather than the planning skeleton.
- Indoor limb rotations align +Y to the incoming bone, without anatomical twist.
  Head rotation is the rendered Y-up, -Z-forward cranial frame and includes head
  turns. Legacy human scenes retain their original rig's world bone frames.

Use the public per-view projection API for camera/pose supervision:

```rust,ignore
let labels = sample.human_poses_for_view(view_index)?;
for person in labels.people {
    let id = person.instance_id; // shared with the person's bounding box
    for joint in person.joints {
        // world_position, world_rotation, view_position, pixel, axial_depth
        // in_frustum is geometric inclusion, NOT an occlusion label.
    }
}
```

`ViewHumanPoses` is JSON-serializable and versioned. It records camera/timestep
indices, normalized playback time, trajectory progress, and physical seconds
when configured. Projections use the captured full K and camera transform, with
image X right, Y down and top-left pixel centres at `(0.5, 0.5)`. Behind-camera
joints have no pixel; off-image projections remain available with `in_frustum =
false`. Joint centres normally lie inside the skin, so testing them directly
against surface depth is **not** a reliable joint visibility oracle. Invalid
calibration, hierarchy, IDs, joint arrays or timeline alignment return an error.

The same method works on `GpuSample::metadata`; it needs no GPU image readback,
model load or additional render pass. Native CLI safetensor archives preserve
poses as `human_pose_position [B,T,H,J,3]`, `human_pose_rotation [B,T,H,J,4]`,
`human_count`, `human_instance_ids`, names and parents. Human padding must be
masked with `human_count`. The Burn archive loader reconstructs `Sample`, so the
projection API also works after loading a dataset. Core pose components and
projection helpers compile for WebAssembly; native GPU transport is a separate
native-only API. Pose gizmos are optional and do not control label capture.

Local visual and capture qualification:

```sh
cargo run --example review_static_poses -- out/static_pose_review
```

This renders twelve dressed people, exports calibrated pose JSON, verifies
identity against bounding boxes and verifies head rotations against geometry.
The result is a local inspection artifact, not a photorealism qualification.
