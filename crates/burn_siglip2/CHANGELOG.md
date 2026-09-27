# Changelog

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
