# bevy_zeroverse 0.20.0

This release includes continuous indoor generation (generator 8 / capture-v10),
improved interactive scene preparation, AnnyBody-based people and clothing,
procedural material programs, corrected annotation paths and a consolidated viewer
inspector. See the [generator review](procedural_domain_v8.md) for rendered samples,
100,000-seed coverage metrics and the limits of the validation.

The companion crates are `bevy_zeroverse_ffi` 0.20.0 and `bevy_zeroverse_burn` 0.3.0.
All retain Bevy 0.19.1, Burn 0.21.0 and the human crates at 0.4.0.

## Compatibility

- Public indoor objects, cameras and manifests gained fields, and object kinds
  gained variants. Update Rust struct literals and exhaustive matches accordingly.
- The capture identity changed from capture-v6 to capture-v10. Existing datasets
  remain readable; generation must start a new shard rather than resume across
  capture identities. Generator versions identify different seeded scene spaces.
- Published `cfg_aliases` 0.2.2 replaces the checkout's earlier local warning fix.
  The release lockfile also updates spin to 0.10.1, lz4_flex to 0.12.2 and
  bitstream-io to 4.10.0, removing the three yanked dependency entries found
  during package verification.
- Registry packages exclude the asset directory. Occupied indoor scenes require
  `assets/burn_human` from the matching repository checkout; deploy that directory
  under the application's asset root, or set indoor human density to zero.
- Root Cargo patches are not inherited by registry consumers. The checkout's
  Vulkan upload and memory-retention fixes still require its documented wgpu-core
  and wgpu-hal patches. Retain bounded generation processes for production datasets.

Photographic realism, 10M-sample learning utility and unlimited-process memory
stability are not established by this release. Earlier review documents record
the local status and evidence at the time of each review.
