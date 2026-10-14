# Articulated room motion

`BevyZeroverseConfig.indoor_motion` is an optional JSON policy. It is `None` by default. Native CLI, Rust/Burn consumers and the WebGPU viewer share this policy; no model or additional Cargo feature is required. The studio exposes it under **Scene → Room motion** and preserves it in shared URLs.

```json
{
  "chair_fraction": 0.35,
  "door_fraction": 0.65,
  "cabinet_fraction": 0.5,
  "covering_fraction": 0.5,
  "max_mechanisms": 12,
  "chair_travel_m": 0.65,
  "chair_swivel_degrees": 70.0
}
```

For a multi-frame dataset, supply this as `--indoor-motion '<json>'` alongside the usual playback/capture options. `--indoor-motion '{}'` enables the defaults. Configuration edits take effect on Apply/regeneration, not while dragging a slider. Human motion remains a separately enabled policy.

## Construction and playback

- Caster chairs can translate and swivel their upper assembly independently of their base. Occupied support chairs remain static.
- Hinged, paired, sliding, telescoping and revolving door leaves preserve their mechanical pivots, tracks, attached handles and semantic identities.
- Kitchen and office cabinet leaves retain separate hardware. Hinged, sliding and upper lift fronts can move when space permits. Drawer facades and open shelving remain static.
- Window coverings gather or deploy using stable topology and metric UVs. Rigid blind slats retain their orientation and dimensions while translating; rail/support geometry remains attached. Incompatible or blocked deformation proposals remain static.

All mechanisms sample absolute normalized timeline progress with seeded start/end intervals and smooth acceleration. Scrubbing, reversed playback and synchronized multi-camera capture use the same state, independent of FPS. Camera policy `duration_seconds` supplies physical capture time as before.

With motion disabled, the normal batched static builders are used. No temporal BVH, swept volumes, mesh splitting, model initialization or animation mesh work runs. With motion enabled, rigid parts update transforms only; deforming coverings update their bounded mesh subset. Stationary timeline samples are skipped.

## Clearance contract

After static construction, admission checks complete conservative swept volumes against the constructed triangle geometry, camera path hulls, static people and previously accepted moving mechanisms. Hinge/frame contacts belonging to the same doorway are intentional mechanical attachments. Chair travel must remain on the same floor level. Rejected candidates preserve their original static state and record a reason.

Accepted travel regions are reserved for the **entire timeline**. Human navigation uses these reservations; dressed pose admission additionally bounds the continuous skinning interpolation between motion frames. Consequently, changing playback speed, reversing time or jumping directly to a capture timestep cannot make a previously valid motion unsafe. This is intentionally conservative: the planner does not schedule a person to pass through a door only during its open interval. Requested fractions are proposals, not guaranteed acceptance rates.

This is kinematic procedural animation, not a force/cloth/contact physics simulation. It does not animate a seated person together with their chair or infer who operates a door.

## Annotations and capture

`IndoorManifest.indoor_motion` records policy, accepted/rejected mechanisms, seeded intervals, scene-local rigid parameters and swept reservations. `MechanismReport::rigid_transform_at` samples rigid scene-local deltas. Covering deformations are represented by the actual captured geometry.

Part entities, semantic labels, mesh topology and annotation order remain stable through playback. Object bounding boxes are recomputed from the current parts. RGB, depth, position, normals, semantic masks and co-visibility therefore use the same current geometry. Dataset optical flow retains surface correspondence across the captured timesteps, including deforming coverings. The existing interactive prepass flow preview is a velocity visualization; exact dataset flow remains the capture API's contract.

Preparation completes temporal clearance before publishing the scene. When human motion is also enabled, capture additionally waits for its validated clips. Animated parts are excluded from static indirect-light occluder baking; realtime shadows still follow them. Indirect illumination remains approximate. O-voxel export rejects any enabled room-motion policy, including a single-timestep request, just as it rejects human motion.

Validation artifacts for this implementation are local under `out/indoor_motion*`; they are not a published dataset or project-page study.
