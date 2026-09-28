# bevy_zeroverse 0.22

This release includes the accumulated indoor appearance, tabletop, seating, architecture/window, multi-view camera and primary-room O-Voxel upgrades, with measured preparation optimizations and an updated technical paper. Companion versions are `bevy_zeroverse_ffi` **0.22.0** and `bevy_zeroverse_burn` **0.5.0**. The unchanged owned `burn_siglip2` crate remains **0.1.1**. Bevy/Burn and the published human-motion dependencies remain pinned to the existing compatible releases.

The [generation review](generation_v18.md) reports 1,024 validated layouts, 64 rendered rooms, rendered overlap/annotation checks, SigLIP variance and benchmark provenance. The matched local preparation/capture workload improved from 2.84 to 4.28 views/s; normalized registry builds measured 3.58 views/s for five channels and 4.32 views/s for RGB at 320×240. These figures include generation and readback, exclude encoding and have explicit workload/build limits.

## Behavior and compatibility

- Generator **18**, capture identity **v28**: create new shards when upgrading; resume must not mix contracts.
- Indoor dataset CLI defaults: four grouped primary-room cameras, one worker, 16-room chunks. Independent cameras remain available via `--indoor-camera '{"multiview":null}'`. Old archive fields retain independent-camera semantics; other scene types retain their defaults.
- More continuous furniture, display/content, PBR and facade parameters; multiple exterior window walls and large/floor-height openings. See [appearance](appearance_v14.md), [tabletop](tabletop_v15.md), [seating](seating_v16.md) and [facades](facade_v17.md).
- Annotated bounds and O-Voxel use the primary room. O-Voxel requests require one timestep and disabled human motion. See [O-Voxel](ovoxel_indoor.md).
- Sin playback is available in the inspector, scene regeneration remains explicit, and camera grids use their own background instead of revealing the editor view through gaps.
- Indoor preparation builds geometry once, skips unused finish textures and parallelizes tangent construction on a bounded native pool. Wasm uses cooperative serial conversion and avoids the asset-upload cap that caused missing architecture.
- Benchmark schema 3 distinguishes enqueue time from asynchronous CPU stages.
- Persistent `LiveDataset` rejects configuration changes requiring another renderer. Use a new process for new dimensions, modes, timesteps, assets or scene policy. Its tests now use actual process isolation.
- The FFI crate no longer enables unused ndarray BLAS support, avoiding an unintended native BLAS linker requirement in the workspace's SigLIP backend.

Registry packages omit assets and checkout-only graphics patches. Supply `assets/burn_human` under the application asset root for occupied rooms, or set human density to zero. Motion is opt-in; static captures do not load ARDY. Browser motion support remains compiled behind `web,human_motion`.

Photographic realism, arbitrary-motion physical correctness, unlimited-process memory stability and downstream 10M-sample learning gains remain unproven. The paper removes earlier unsupported training-result claims and separates historical Cycles results from this release's generator audit.
