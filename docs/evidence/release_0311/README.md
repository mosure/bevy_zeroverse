# Seeded furnishing patch qualification

Core/FFI 0.31.1 and Burn wrapper 0.14.1 repair seed `43218880` without
relaxing floor support, doorway, ceiling, pillar or collision constraints.
The bounded fallback completes an accepted sofa group only when ordinary
workstation repair leaves no primary activity surface. Successful ordinary
scenes and their random streams remain unchanged.

[Receipt](receipt.json) binds the final generator identity to native six-view
RGB/depth/normal/position/semantic/co-visibility captures and a 256-seed
neighborhood audit. The neighborhood checks all layouts and camera paths and
actual meshes for its first sixteen seeds. [Capture](capture.json),
[geometry](geometry.json) and [distribution](neighborhood.json) retain the
complete diagnostic measurements.

The regression suite covers three cameras, twelve density/occupancy combinations,
real geometry, deterministic replay and unchanged complete-room recovery.
The implementation passed 371 motion-enabled core tests (22 intentionally
ignored) before the metadata-only version bump; the final-version focused
regressions, strict workspace lint and motion-enabled Wasm compilation are
checked again before release. Native captures use full effects with 1024 GI
rays per probe. Wasm compilation is not a browser runtime qualification.

The canonical Rust publisher separately refreshes the 512-room audit, 32-room
gallery, factor sweep and paper from the final source. This bugfix does not
claim new throughput, photographic realism or downstream training utility.
