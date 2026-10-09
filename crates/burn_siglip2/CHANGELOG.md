# Changelog

## 0.2.0 (2026-10-08)

- Migrate to Burn 0.22 runtime devices and backend-independent tensors/models.
- Use Flex for the default CPU loader; retain `ndarray` as a feature alias.
- Initialize browser WebGPU asynchronously and use explicit devices in native loaders.
- Preserve checked CDN shards, artifact formats, F32 inference and numerical oracles.
- Avoid enabling fusion implicitly when sharing a renderer device with human motion.

## 0.1.1 (2026-09-27)

- Move canonical source and crate publication to `mosure/bevy_zeroverse`, under
  `crates/burn_siglip2`, as a workspace member with an independent version.
- Include the browser example, reference generator, CDN-bundling scripts and
  fixtures in the owned package. Preserve MIT/Apache-2.0 licensing and origin
  attribution.
- Retain Burn 0.21, the public Rust API, all three supported SigLIP2 checkpoints,
  native/browser backends, CDN URLs and cache format.
- Update fixed-width importer byte iteration for the current Clippy lints without
  changing its conversion or validation behavior.

## 0.1.0

Initial release from `mosure/burn_loom`.
