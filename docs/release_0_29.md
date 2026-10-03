# bevy_zeroverse 0.29

Core and FFI **0.29.0**, Burn wrapper **0.12.0**, and publication tool **0.1.1**
expand continuous PBR programs for woven fabric/leather, concrete and marble,
paint/plaster, stained timber and foliage. Correlated color, metric relief,
roughness and cavity maps share bounded structure groups. Clothing reuses
textile maps, leaf atlases preserve blade alignment, and timber/stone floor
atlases resolve at 512². Native preparation uses a bounded worker pool; WASM
retains the serial fallback and optional motion support.

The [material qualification](material_quality.md) covers 1,024 recipe seeds,
1,120 production map sets and 64 furnished rooms / 192 views. Matching geometry
and semantic/co-visibility annotations remain unchanged. The formal Rust pipeline
refreshes the page and paper from current captures. Geometry remains generator
22; the appearance contract is capture-v39/finishes=7. Capture and SigLIP2 crate
versions are unchanged. Photographic realism and ten-million-sample training
utility remain unqualified.

## Rust compatibility

`MaterialRecipe` adds optional `textile`, `leather`, `mineral`, `coating`, `wood`
and `leaf` fields. Exhaustive Rust literals must initialize them; use `None`
for omitted programs or sample recipes through `program::sample` /
`program::sample_with_floor`. Serialized manifests with omitted fields continue
to load through their serde defaults. The minor release prevents these Rust
struct additions from arriving through a patch update.

The two wrappers require core 0.29.0. Resume identity checks distinguish the
new appearance contract and compiled source; existing samples are not silently
combined with new renders. Calibration and numeric annotation layouts are
unchanged.
