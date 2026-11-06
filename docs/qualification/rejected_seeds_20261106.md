# November 6 rejected-room qualification

This is a local CPU qualification of seeds **69608412, 69531153, 69430992,
69391845 and 69329109**. The actual training rejection records and camera policy
were inspected; these were placement failures, rather than polygon triangulation
failures. The capture configuration has eight square views, baseline 0.5, mixed
pinhole/fisheye lenses, moving room cameras and seed-selected moving ceiling
cameras, density 0.65 and no people.

## Causes and repairs

| Seed | Recorded rejection | Resolution |
| --- | --- | --- |
| 69608412 | ElevatorLanding missing its required Elevator | Search actual free bays and permit a lateral shift of the full-width circulation aisle. Preserve all roof members, minimum car dimensions, car headroom and the door's approach area. Check shifted aisles against structural obstructions; restore reservations on unsuccessful recovery. |
| 69531153 | Cafe missing its required DisplayCase | Turn the sampled case into a side bay when translation alone cannot place it. Keep its dimensions and all ordinary placement gates. |
| 69430992 | Cafe region in a PublicSuite missing its required DisplayCase | The same orientation-aware recovery fits the narrow functional region without reducing aisle width or shrinking the case. |
| 69391845 | Mixed mounting rig placed 3/8 views | The pending generator produces a valid eight-view rig; retain an explicit training-policy regression and deterministic replay. No seed-specific workaround was needed. |
| 69329109 | Mixed mounting rig placed 3/8 views | A small mezzanine left too few distinct compatible paths on its upper level. Rank candidate references by supported view slots, then deform already compatible peer paths to expand those partial rigs. Every proposal is requalified with its destination lens and motion role. |

The mixed-camera recovery retains its existing 16,384 derived-path and 8,192
visibility-test budgets. It does not replace requested lenses, disable motion,
lower overlap thresholds, accept coincident paths or change the room to fit the
cameras. Variable pinhole recovery samples the configured intrinsic PDF;
fixed priors and fisheye calibrations stay fixed. Compatibility uses the existing
five trajectory qualification times and swept clearance checks.

The broader regression also exposed **65697418**, an ordinary dense rig that
stalled at seven views. Dense-rig recovery now spends a bounded portion of its
existing room-space proposal slots on small deformations of known-clear paths.
It retains independent room-space exploration and the original total attempt
budget. Both mounting pipelines share this deformation helper. Ordinary
successful searches avoid this additional recovery work.

These changes use general spatial constraints, not seed allowlists. A frozen
manifest fixture retains the original rejected elevator architecture so later
sampling changes cannot silently stop exercising the narrow-bay problem.

## Results

All five exported rooms pass full CPU mesh validation and realize all eight
requested cameras. These are individual measurements of manifest construction
including camera search, excluding mesh validation and schematic export:

| Seed | Resolved layout | Objects | Cameras | CPU construction and placement |
| --- | --- | ---: | ---: | ---: |
| 69608412 | ElevatorLanding | 33 | 8 | 0.349 s |
| 69531153 | Cafe | 88 | 8 | 0.129 s |
| 69430992 | PublicSuite | 66 | 8 | 1.273 s |
| 69391845 | Workshop | 57 | 8 | 1.719 s |
| 69329109 | Kitchenette with mezzanine | 50 | 8 | 9.363 s |

The mezzanine case remains comparatively expensive, but now completes within
the unchanged search budgets. These five deliberately difficult seeds are not
a representative throughput population.

Across the validation pass, **145 distinct targeted/regression tests passed**.
The final rejected-seed rerun passed 92 tests with two optional larger cohorts
ignored. A further 37-test pass covered the five reported seeds and their ten
immediate neighbors, shared motion, mixed-rig recovery, ordinary multiview
recovery, calibration and connected-camera contracts. Four focused tests
verified the additional dense-rig fix, frozen elevator geometry, density/person
variants and failure rollback. Those counts overlap and are not additive.
The earlier broader pass also exercised 96 combinations of camera count
(2/4/8/12), baseline (0/0.5/1) and eight rooms. Its one discovered failure,
65697418, was repaired and passed both subsequent regression runs.

Strict native all-target Clippy (`-D warnings`), formatting, diff checks and the
CPU auditor build pass. Two CPU jobs/threads were used; no GPU was initialized.
The five final audit processes, including mesh checks and schematic/artwork
exports, completed in 13.6 seconds total on the shared host.

## Reproduction and evidence

The camera policy fixture is
`src/scene/procedural_indoor/cameras/ceiling/tests/mixed_training_camera_settings.json`.
It matches the recorded downstream capture policy; the source configuration
SHA-256 is `669852b338863ce7b5c6feca5817b22d7940a3894a56a6a052a4269b39b040df`.

For any of the five seeds:

```sh
cargo run --locked --example audit_placement_sampling -- \
  --profiles mixed --seed 69608412 --rooms 1 --density 0.65 \
  --human-density 0 --cameras 8 --mesh-rooms 1 --save-scenes \
  --camera-settings src/scene/procedural_indoor/cameras/ceiling/tests/mixed_training_camera_settings.json \
  --output out/seed-check
```

This exports a manifest and API schematic and validates layout, required
functional furnishings, camera geometry and full CPU construction meshes. The
test suite additionally checks directed overlap, clearance over complete curves,
slot-specific optics/mounts, motion, unchanged scene geometry during camera
search, deterministic replay, and bounded proposal counters. Immediate neighbor
seeds and furniture/human density variants are included.

Local evidence, logs, complete manifests and schematics are in
`out/seed_failures_20261106/`; `qualification.json` indexes the final results.
These original receipts used generator **114**, **capture-v139**. The combined
0.62.3 release also includes the [furnishing expansion](furnishing_diversity_20261106.md)
and uses generator **115**, **capture-v140** (`public_programs=15`,
`multiview=23`). Its final regression log is in `out/release_0623/`. It is not
bitwise equivalent to generator 112 from release 0.62.2.

## Limits

This checks CPU construction and the placement visibility oracle. It does not
qualify rendered RGB/annotation parity, GPU throughput, browser execution or
training continuation. Schematics are diagrams, not rendered camera captures.
The CPU runs share the host with active training; their latency is diagnostic,
not a production-throughput benchmark. The rendering and annotation pipelines
are unchanged. No training processes or deployments were modified.
