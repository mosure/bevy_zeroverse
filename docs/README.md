# Documentation

These guides describe the current generator and export contracts.

| Task | Guide |
| --- | --- |
| Build and inspect procedural interiors | [Generation, controls and validation](procedural_indoor.md) |
| Generate native datasets or use Python | [Dataset configuration and export](procedural_indoor_dataset.md) |
| Tune native generation throughput | [Measured performance and correctness](generation_throughput.md) |
| Set camera spacing, overlap and trajectories | [Multi-view cameras](multiview_cameras.md) |
| Run the browser viewer | [WebGPU](procedural_indoor_web.md) |
| Add text/waypoint human motion | [Human motion](human_motion.md) |
| Export temporal correspondence | [Optical flow and motion vectors](optical_flow.md) |
| Export shared camera visibility | [Co-visibility](co_visibility.md) |
| Export primary-room surface voxels | [O-voxel](ovoxel_indoor.md) |
| Measure embedding-space diversity | [SigLIP2 audit](embedding_audit.md) |

The [architectural evaluation](architecture_v22.md) contains the latest
512-room structural audit and 256-view rendered evaluation, including complete
contact sheets and machine-readable distributions. The
[release compatibility notes](release_0_28.md) describe manifest and capture
compatibility.

The [Rust publication protocol](project_page.md) defines the single recipe and
release gate for the website and paper. Versioned reviews, release notes and evidence directories are
archival records: retain their original generator identities and input hashes.
Use these guides for current behavior; a recorded benchmark does not automatically
qualify a later generator or a different renderer/backend.
