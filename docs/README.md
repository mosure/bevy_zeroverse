# Documentation

These guides describe the current generator and export contracts.

| Task | Guide |
| --- | --- |
| Build and inspect procedural interiors | [Generation, controls and validation](procedural_indoor.md) |
| Generate native datasets or use Python | [Dataset configuration and export](procedural_indoor_dataset.md) |
| Tune native generation throughput | [Measured performance and correctness](generation_cpu_efficiency.md) |
| Set camera spacing, overlap and trajectories | [Multi-view cameras](multiview_cameras.md) |
| Run the browser viewer | [WebGPU](procedural_indoor_web.md) |
| Capture static and moving human poses | [Calibrated human pose labels](human_poses.md) |
| Add text/waypoint human motion | [Human motion](human_motion.md) |
| Export temporal correspondence | [Optical flow and motion vectors](optical_flow.md) |
| Export shared camera visibility | [Co-visibility](co_visibility.md) |
| Export primary-room surface voxels | [O-voxel](ovoxel_indoor.md) |
| Measure embedding-space diversity | [SigLIP2 audit](embedding_audit.md) |
| Extend object, material or camera families | [Module ownership and compatibility contracts](code_organization.md) |

The [project page](https://mosure.github.io/bevy_zeroverse/project/) publishes
the current architecture audit, gallery and paper. The [Rust publication
protocol](project_page.md) defines the recipe and its release checks.

Historical reviews link to the evidence captured at their recorded revision.
To run an older analysis locally, restore its files from that revision to the
working tree. For example, `git restore --source=6cc9ee235004b0e71b7eeb69bc5990375f77d48c --worktree -- docs/evidence/domain10`
retrieves the archived domain study. The corresponding
[evidence tree](https://github.com/mosure/bevy_zeroverse/tree/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence)
also works without a local copy of the history.
