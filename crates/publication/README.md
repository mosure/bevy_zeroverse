# bevy_zeroverse_publication

The canonical Rust publication pipeline for the bevy_zeroverse project page and
whitepaper. It is a separate, GPU-free workspace crate. The generator shares its
wire contract and camera-color program through `bevy_zeroverse_capture`.

From a bevy_zeroverse checkout:

```sh
cargo run --locked -p bevy_zeroverse_publication -- refresh
cargo run --locked -p bevy_zeroverse_publication -- verify
cargo run --locked -p bevy_zeroverse_publication -- rebuild
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
The page and paper describe the current generator and its measured cohorts.
Every cohort must match the compiled generator identity. Historical studies,
release comparisons and proposed implementation plans stay outside this bundle.
Camera-count and baseline studies use the same identity gate as the room gallery.

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
current cohort identities and every managed page/paper artifact. A crash during
installation cannot yield a partially installed bundle that passes this gate.
Tests cover the failure modes that previously allowed mixed gallery cohorts.

## Paired room studies

`compare-indoor` qualifies two complete, explicitly identified cohorts and builds
an independent local before/after gallery without replacing the project page or paper.
Study outputs must stay below ignored `out/`, outside `out/publication/`.
Reports and receipts carry `visibility = local_only`; the release pipeline
rejects them as snapshot inputs. They are not deployment artifacts.
The default protocol is 64 consecutive rooms, three 512×512 views, one timestep,
furnishing density 0.65, human density 0.25 and 1,024 diffuse-GI rays.

```sh
cargo run --locked -p bevy_zeroverse_publication -- compare-indoor \
  --before out/study/before --after out/study/after --output out/study/report \
  --before-population out/study/before-population \
  --after-population out/study/after-population
cargo run --locked -p bevy_zeroverse_publication -- verify-study \
  --output out/study/report
```

Every seed, camera, RGB/annotation preview, exact co-visibility membership/validity
mask, schematic, actual-mesh audit and input identity is retained and checked.
Primary architecture, base lighting samples and camera controls must match.
Updated viewpoint priors, added fixture illumination and collision-safe camera
replanning are disclosed; the views are not assumed pixel-matched.
Optional population inputs must match each renderer's compiled identity and
controls. No sample is dropped because it is sparse, dark or has a close neighbor.

The first invocation produces `embedding-index.json`. Run the optional native
`indoor_embed` tool against it, then repeat `compare-indoor` with
`--embeddings out/study/embeddings` to bind the exact tensor and model checksums.
Reports include cross-room nearest pairs, equal-weight room-centroid variance,
and paired-room bootstrap intervals. Seed uniqueness and products of marginal
entropies are not estimates of perceptual procedural-space size.

Outputs are staged and installed atomically. Failed qualification preserves the
previous report, and `verify-study` detects modified or missing gallery bytes.
Study generation supports 2–256 rendered rooms and a single timestep; use
`--recipe protocol.toml` for a smaller qualification fixture or a larger cohort.
