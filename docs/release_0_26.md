# bevy_zeroverse 0.26

ProceduralIndoor now samples nonrectangular building envelopes: tapered walls,
chamfered corners, concave cut-ins, sloped ceilings, arched portals, interior
pillars, raised/sunken floors and furnished mezzanines. A shared architectural
program drives construction, placement, camera clearance, lighting anchors,
visibility and export. Existing scene modes remain available.

Versions: `bevy_zeroverse` **0.26.0**, `bevy_zeroverse_ffi` **0.26.0**, and
`bevy_zeroverse_burn` **0.9.0**. `burn_siglip2` remains **0.1.1**. Engine and human
model dependencies retain their published versions.

## Compatibility

- Generator **22**, capture identity **v33**, metrics schema **11**. Resume a
  dataset only with a matching capture identity; begin new shards for this
  generator. Previously captured datasets remain readable.
- `IndoorManifest` adds optional `envelope`; archived JSON without it retains
  the rectangular construction path. `Partition` adds `arch_rise`, defaulting
  to zero in older JSON. Rust struct literals need the added fields.
- Reconstruction bounds include negative floor depth. Courtyard/context meshes
  are excluded from O-Voxel even when inside its bounding box. O-Voxel still
  requires one timestep and disabled human motion.
- Motion routes treat level changes as barriers. Actors on raised, depressed or
  mezzanine floors remain static with a recorded rejection unless staged onto
  suitable base-level ground; stair traversal is not claimed.
- Larger camera groups use local roof height and verified free corridors for
  recovery proposals, retaining bounded search and all separation, overlap,
  spread, motion-variation and swept-collision requirements.
- Architectural programs and distributions are exported in `architecture.jsonl`
  and metrics; feature counts include rooms where the feature is absent.

## Evidence and paper

The [architectural review](architecture_v22.md) reports 512 consecutive valid
rooms and 256 aligned rendered views across 32 consecutive rooms. Additional
checks cover 640 activity/density-extreme configurations with eight cameras,
96 built-mesh surface seeds, and matching CPU/GPU occupancy and semantic labels
for a room containing a cut-in, depressed floor and mezzanine.

The whitepaper includes four-view RGB/co-visibility pairs, a consistent additive
camera legend, per-peer binary masks and peer-count visualizations. Its examples
and existing population study remain explicitly **generator 21**; they are not
relabeled as generator 22. The figure builder verifies exact PNG/uint16 mask
round trips, RGB capture identity, validity, calibration and denominators. The
downloadable PDF and self-contained LaTeX archive include both new figures.

Packages exclude research media, assets and the checkout's graphics-library
patches. Published dependencies resolve from crates.io. Validation of the native
registry package and the release commit's CI is required before publication;
Wasm compilation alone is not a browser runtime qualification.

This remains a bounded architectural grammar, not CAD, structural analysis or
building-code certification. Photographic realism, unlimited-process memory
stability and downstream pretraining gains remain unproven.
