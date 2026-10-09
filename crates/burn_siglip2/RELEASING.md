# Releasing burn_siglip2

`mosure/bevy_zeroverse` is the canonical source and release repository for this
crate. `burn_siglip2` has an independent version; releasing it does not require a
Bevy Zeroverse release. Keep the root `burn_siglip2` dependency version in sync
with `crates/burn_siglip2/Cargo.toml`.

Run these commands from the repository root. No model downloads are needed for
the regular tests. The published package uses registry dependencies, without
requiring the sibling `burn_human` checkout or any wgpu patches.

```sh
cargo fmt -p burn_siglip2 -- --check
cargo test -p burn_siglip2 --features cli,pipeline --locked
cargo clippy -p burn_siglip2 --all-targets --features cli,pipeline --locked -- -D warnings
cargo check -p burn_siglip2 --no-default-features --features wgpu,pipeline,bootstrap --locked
cargo check -p burn_siglip2 --target wasm32-unknown-unknown --no-default-features --features wasm --locked
cargo package --list -p burn_siglip2 --locked
cargo publish --dry-run -p burn_siglip2 --locked
```

For a local, uncommitted review, add `--allow-dirty` to the two packaging commands.
Inspect the archive: source, tests, small fixtures, licenses, browser example and
reference/CDN scripts belong in it; model weights, caches and generated browser
glue do not. The dry run verifies the unpacked package using registry dependencies.

For changes to inference or preprocessing, also run the reference parity controls
described in [README.md](README.md) and [scripts/README.md](scripts/README.md).
Those tests are opt-in and use explicit model/reference paths. Preserve the
loaded-weight and tokenizer identities with their results.

When publication is requested and the validated source is ready for release:

1. Finalize the changelog/version and commit the reviewed crate changes.
2. Run `cargo publish -p burn_siglip2 --locked` using crates.io publisher credentials.
3. Verify `cargo info burn_siglip2@<version>` from outside this workspace.
4. Tag the corresponding source commit `burn_siglip2-v<version>` and push it.

The CDN model artifacts have a separate lifecycle. Moving or publishing the Rust
crate does not re-upload model weights or change their cache keys. The packaged
`scripts/bundle_siglip2_assets.sh` remains the tool for a separately authorized
CDN artifact update.
