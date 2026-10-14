# Project page and paper generation

The workspace crate [bevy_zeroverse_publication](../crates/publication/README.md)
owns capture validation, figures, the unified page, measured paper text, source
and annotation archives, and the publication receipt. The page and paper describe
the current generator. Matched comparisons between generator revisions remain
local and are excluded from this publication path.

## Commands and contract

```sh
# Current generator → verified captures → page, paper, archives and receipt.
cargo run --locked -p bevy_zeroverse_publication -- refresh
# Force new captures instead of reusing identical completed inputs.
cargo run --locked -p bevy_zeroverse_publication -- refresh --recapture
# Offline artifact and provenance check; no GPU or source capture directory.
cargo run --locked -p bevy_zeroverse_publication -- verify
# Rebuild presentation from the verified shipped dataset.
cargo run --locked -p bevy_zeroverse_publication -- rebuild
# Explicit registry workflow; this example performs no upload.
cargo run --locked -p bevy_zeroverse_publication -- publish --package bevy_zeroverse --dry-run
```

[publication.toml](../publication.toml) is the authoritative recipe. The interactive
gallery audits 512 consecutive room programs and renders the first 32 rooms with
four cameras and two endpoints: 256 camera/time identities. RGB, depth, normals,
positions, semantics and co-visibility all use these same views. Appearance never
filters this cohort. The separate canonical factor sweep has two room families,
14 settings and 120 camera/time images, including displaced principal points,
connected camera graphs, portrait aspect, overlap, material detail, illumination
and exposure.

A generator identity binds the actual renderer sources, Cargo.lock, shaders and
embedded inputs. A source/dependency change invalidates measurements even if the
crate version is unchanged. Publication checks compiled provenance rather than
inferring it from a version string. Text fingerprints normalize CRLF to LF;
binary inputs retain exact hashes. Cross-device bit-identical rasterization is
not promised.

Refresh needs native graphics, `latexmk`/`pdflatex` and Poppler. Rust image codecs,
SVG rendering and a bundled font generate the figures. LaTeX runs with shell
escape disabled. A staging transaction checks every artifact before installing
files, with rollback backups and the attestation installed last. Missing modes,
incorrect calibration, incomplete captures, invalid masks, duplicate camera/time
identities, stale evidence, broken links and paper warnings stop installation.

`www/project/publication.json` records renderer and publisher inputs, capture
hashes and every managed output. The shipped archives make `verify` independent
of absolute workstation paths and `out/`. Edit templates, recipes or source
measurements; do not hand-edit generated figures, text or counts.

## Additional measurements

Camera-count evidence is configured by `camera_study`. Its CPU audit and native
baseline/count sweep retain all attempted configurations and rejection reasons.
The page shows same-room baseline examples, placement diagrams and measured
co-visibility separately from approximate placement overlap.

An optional `room_qualification` directory holds a current-source rendered-cohort
receipt produced by Rust:

```sh
cargo run --locked -p bevy_zeroverse_publication -- cohort-study \
  --input out/qualification/consecutive out/qualification/stratified out/qualification/heldout \
  --output out/qualification/current-cohorts
```

This command verifies every selected capture and annotation before producing the
receipt for the configured evidence directory. Its source identity must match the
checkout. The current qualification uses 256 consecutive, 128 prespecified
stratified and 128 held-out rooms, each with three 512-square cameras and two
endpoints: 512 rooms and 3,072 views. Each population keeps its own denominator and
selection policy. The compact public receipt contains input hashes and summaries;
it does not bundle every training image or turn the 32-room explorer into a
512-room gallery.

Primary-room graph connectivity, directed overlap, baseline and triangulation
statistics accompany whole-image co-visibility. An edge needs 10% shared pixels
in both directions. Graphs are recomputed from their integer-count contract;
inconsistent connectivity, fractions, camera pairs or group counts are rejected.
Disconnected groups and zero-shared pairs remain in the report. These are
camera/time observations, not independent rooms or unique surface area.

Object and material atlases are regenerated from the production builders. They
show current constructions, realized conditional parameters, metric PBR swatches
and finite-sample variation. Counts of available parameters are not multiplied
into a purported procedural-space size. Paper-only timing is generated with
`timing-study`; it separates CPU preparation, completed capture and overlapping
work, with explicit machine contention. Timing charts are not put on the project
page.

The additional receipts are explicit measurement inputs. A stale input blocks
refresh or release verification; it is never silently presented as evidence for
new source. Preparing a measurement does not deploy the page or upload a crate.

## Viewer and annotation behavior

Serve `www/` and open `/project/`. One room explorer owns the room, annotation,
time and peer controls. Its schematic and calibration follow the same selection.
Co-visibility has one interactive presentation; its static figure is a
JavaScript-disabled fallback and a paper figure. All 32 rooms retain co-visibility
previews and exact downloadable masks. The README's object Zeroverse and MatSynth
examples remain separate illustrations.

Lossless display previews do not apply exposure correction or crop the capture.
Depth uses Turbo (warm near, cool far), with a fixed logarithmic 0–15 m
display scale and black background. It colorizes the existing 8-bit axial
metres/15 previews (15/255 m increments, clipped beyond 15 m); the source
captures stay untouched. Normals use `(n+1)/2`; positions use the annotation AABB. These previews are not float32 training labels. Numeric exports retain
calibration, coordinate conventions, source identity and annotation policy.

Bit `i` in a 16-bit co-visibility mask identifies capture camera `i`; the source
bit is zero and the editor has no bit. A separate 8-bit validity mask distinguishes
an unshared surface from background. The ordered additive legend is shared by all
views. Peer selection displays that peer's RGB and highlights shared surfaces in
the other views. Fractions divide by valid source pixels.

The gallery uses the surface glass policy. Through-glass annotations are an
explicit option for geometry, semantics and co-visibility: transmissive panes are
skipped, while frames and opaque frit remain occluders. Both policies use straight
geometric rays; neither follows reflected or refracted RGB content.

Browser checks exercise room/time/mode/peer switching, filters, keyboard controls,
mobile and desktop layouts, and the JavaScript-disabled fallback. WebGPU runtime
qualification uses hardware Chrome separately from these static-page checks.

Generator validity and these finite audits do not establish photographic realism,
unlimited-process memory stability, a ten-million-room dataset, or gains on a
real downstream reconstruction task.

The source-bound door/cabinet atlas is a required publication inventory alongside the object and material atlases. `door-study` binds its 64-seed category audits and aligned RGB/normal/semantic examples; `refresh` generates the gallery and paper figures together. Architectural doors are not counted as freely placeable furniture categories.
