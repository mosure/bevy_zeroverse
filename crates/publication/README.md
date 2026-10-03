# bevy_zeroverse_publication

The canonical Rust publication pipeline for the bevy_zeroverse project page and
whitepaper. It is a separate, GPU-free workspace crate. The generator shares its
wire contract and camera-color program through `bevy_zeroverse_capture`.

From a bevy_zeroverse checkout:

```sh
cargo run --locked -p bevy_zeroverse_publication -- refresh
cargo run --locked -p bevy_zeroverse_publication -- verify
cargo run --locked -p bevy_zeroverse_publication -- rebuild --release-version 0.28.2
# Registry publication through the same preparation/validation gate:
cargo run --locked -p bevy_zeroverse_publication -- publish --package bevy_zeroverse --dry-run
```

`refresh` owns the complete dependency order: compile/capture the current
generator when the cache is absent or incompatible, validate the completed
cohort, generate media/plots/HTML, compile the paper, package exact source/mask
downloads, validate the staged bundle and install it with its attestation last.
`--recapture` forces a new GPU run. Scientific validation failures leave the
published bundle unchanged. A file-install error rolls back replacements.

`verify` is a read-only release/deployment gate. It requires neither raw captures
in `out/`, a GPU, Python nor LaTeX. It checks current crate/source identity,
protocol settings, exact input/output hashes, scene/camera/time identity, every
annotation, mask/validity/preview consistency and denominators, page links and
the PDF's exact source archive. It fails on stale captures, missing channels,
untracked additions to the managed artifact set or edits to generated outputs.

`rebuild` first verifies the self-contained shipped capture bundle, then
regenerates HTML and the paper from its measured programs and reports. This is
the release/Pages command: it never relabels earlier measurements as current or
downloads/initializes motion models. Source or dependency upgrades require a
successful `refresh` before release.

The recipe is `publication.toml`; authored templates are in this crate's
`templates/`. Measured results and selected examples come from the dataset,
not manually copied numeric prose. Figure rendering uses bundled fonts, native
Rust image codecs and SVG rasterization. Preview conversion is lossless.
Membership remains an exact 16-bit mask with separate 8-bit 0/1 validity.
Historical baseline/motion inputs are explicitly declared and hashed separately.

Refresh/rebuild additionally require `latexmk`, `pdflatex` (including the usual
LaTeX packages) and Poppler's `pdftoppm`. LaTeX is the document typesetter; Rust
owns source generation, dependency closure, warnings, provenance and staging.
The publisher does not estimate photographic realism or downstream utility.

`publish --package <name>` prepares the latest page/paper before calling Cargo.
It refreshes stale inputs or rebuilds an already-current shipped bundle, verifies
the installed result, then invokes `cargo publish --locked`. `--dry-run` never
uploads and permits local dirty-checkout packaging; actual uploads retain Cargo's
clean-checkout requirement. Publish the shared capture crate before dependent
packages. Direct external `cargo publish` invocations cannot be intercepted; use
this entry point for repository releases.

`www/project/publication.json` is the commit marker and machine-readable
attestation. It binds renderer inputs, publisher/templates, capture sources,
reference identities and every managed page/paper artifact. A crash during
installation cannot yield a partially installed bundle that passes this gate.
Tests cover the failure modes that previously allowed mixed gallery cohorts.
