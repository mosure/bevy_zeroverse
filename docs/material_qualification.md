# Current material qualification

The project page and paper are generated from one checksum-bound material atlas.
`bevy_zeroverse_publication refresh` owns capture, sampling analysis, plots,
paper typesetting and installation. It does not deploy the website or publish a
crate. The separate paired room studies remain local.

## Cohorts

- **Recipe population:** 39 rows × 256 consecutive material seeds. These are the
  36 production surface roles with timber, carpet and stone Floor rows separated, plus a dedicated Rug program.
  Recipes retain optional subprograms and derived values. Eight-bin numeric
  marginals have explicit recorded-value denominators.
- **Studio:** six fixed illustration seeds, 0, 1, 4, 19, 53, 101, shared across
  rows. Seed 4 illustrates frosted interior glass. There is no visual-quality
  filtering. All 234 RGB and matched normal images are native Bevy captures using
  actual `IndoorMaterials::build` / `for_part` handles. Raw map previews and PBR
  scalars accompany every column. CPU reference synthesis is explicit; native
  hardware/engine/package provenance is retained.
- **Object assignment:** 256 consecutive mixed-layout rooms, density 0.65, no
  people, primary-room objects only. The actual builders export object and material-part denominators in the atlas.
  Category coverage is measured rather than assumed. Supported finishes
  must use the seeded selector and a valid slot; independent book covers remain
  permitted. Unsupported surfaces resolve to their base role.

All twelve finish slots occur for the major furniture substrates (wood, fabric,
leather, ceramic and chrome) and leaf roles in this cohort. Slot counts are
observations, not equal-frequency guarantees. Furniture assemblies use separate
material roles for their parts; sharing a coherent finish does not collapse
those roles into a single material.

Molded keyboard keys and printer housings use the plastic substrate. Its seeded
pigment palette includes charcoal, warm white and occasional colored finishes,
sharing the same bounded dielectric texture structures. Regression tests keep
glazed ceramic out of these device parts and check pigment coverage and replay.

## Texture and rendering contract

The swatches keep illumination, camera, exposure and the reflection environment
fixed. Metric sphere/panel UVs show repeat scale; leaves use production blade
geometry and normalized UVs. Their shapes are diagnostic probes rather than
room objects. Glass is shown against a reference pattern. The environment is a
production procedural approximation, not a path-traced studio.

The base-color preview is sRGB before the material multiplier; the tangent
normal preview is distinct from rendered view-space normals. Roughness previews
show the packed green channel before its scalar multiplier, or the scalar if
there is no map. Complete mip-chain storage is checked before map export. Normal
and GGX roughness filtering have separate production replay tests, and the
hardware GPU parity test compares all three maps and all mip levels for the
supported GPU programs. The default GPU path retains CPU synthesis for other
material families.

## What these measurements do not establish

Twelve finish slots share one to three map structures per role per room. Recipes
are correlated and some fields are conditional or unused for a particular role.
Therefore seed counts, parameter counts and numeric ranges are not independent
combination counts or effective perceptual volume. Small six-image panels do not
estimate appearance frequencies. The report makes no photographic-realism or
10M-sample training-utility claim.

The assignment cohort excludes architecture, people and neighboring objects.
The atlas covers base architectural/object substrates; installation-specific
facade privacy patterns and human complexion/groom programs require their own
geometry/actor contexts. Screen and print content use separate seeded programs;
their base material recipe alone does not describe the content. Container glass
and liquids have several fixed optical constants, retained visibly rather than
counted as continuously diverse.

## Reproduce

```sh
cargo run --locked --example qualify_materials -- --output out/materials/raw
cargo run --locked -p bevy_zeroverse_publication -- material-study \
  --input out/materials/raw --output out/materials/atlas
cargo run --locked -p bevy_zeroverse_publication -- refresh
cargo run --locked -p bevy_zeroverse_publication -- verify
```

Study binaries embed their own Rust source hashes in both audit and render
records. The publisher binds these to the checkout and verifies the exact
generator identity. All map/image files, parameter records and usage histograms
are shipped with hashes. A partial run cannot replace a verified publication.

## Large-area floors and rugs

`cargo run --example review_flooring -- out/flooring/renders` renders 6 m square
floor panels and production rug assemblies under one fixed light rig. The four
seeds are shared across timber, carpet, stone and rug rows. RGB and aligned
normal captures accompany serialized recipes and capture provenance. This local
review is separate from the small-swatch publication atlas.

Floor recipes record installation bond, panel dimensions, independent piece
tint/warmth/roughness, accent frequency and macro mottling. Timber supports
running/broken bonds and basket parquet; stone cuts have independent vein
offsets and orientation. Carpet uses a bounded 512-square room-scale dye/pile
atlas with broadloom and modular tile variation. Rugs use four lazily built
dye/pile slots with coordinated woven bindings; bindings reuse upholstery maps.
Native/WebGPU bake carpet and rug maps once, including normal/roughness mips.
CPU fallback and diffuse-transport albedo use the same recipe. No per-pixel
procedural work is added to the scene PBR fragment shader.
