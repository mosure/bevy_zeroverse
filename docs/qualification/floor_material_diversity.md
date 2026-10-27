# Floor material diversity and qualification

Floor generation separates **product geometry**, **pigment**, **finish** and
**installation**. These are continuous sampled programs, not a larger list of
fixed swatches. The target is a substantially broader floor distribution for
reconstruction training while retaining physical dimensions, texture filtering
and a bounded generation cost.

## Sampling contract

| Factor | Current behavior |
| --- | --- |
| Pigment | Independent seed domain; natural timber tones, neutral and muted pigmented mineral floors, neutral and colored carpet. Floor timber need not match furniture timber. |
| Finish | Independent domain for polish, roughness, pore filling, clearcoat and relief. Physically related properties remain coupled, such as polished stone having smaller relief. |
| Installation | Seven existing layouts retained. Rectangular stock spans short/long, narrow/wide and rectangular tile formats. Broken-bond installations can use independently offset end joints and variable cut lengths. |
| Cut stock | Bounded periodic boundary displacement; neighboring pieces share boundaries. Patterned parquet and hexagonal installations retain their exact tessellations. Uniform manufactured tiles remain common. |
| Joints | Independent dark/light grout reflectance, metric widths and roughness. Area coverage filters subpixel joints; grout is mixed in linear light. Carpet retains textile seams. |
| Carpet composition | Continuous normalized mixture of heather, striae, offset weave, clouds, stipple and yarn blocks. Six poles and their mixtures share independent dye scale, warp, shear, contrast and palette controls. |
| Carpet microstructure | Independently sampled nap, slub and fleck frequencies; no universal fixed fiber-frequency signature. Relief, roughness and dye share the same substrate field. |
| Lot variation | Independent piece reflectance/warmth/roughness and variable-scale mottling rather than one universal macro-noise frequency. |

Requested rectangular timber dimensions span approximately 0.08–0.34 m across
and 0.55–2.3 m along; tile dimensions span 0.24–1.2 m by 0.30–1.4 m. Counts are
rounded so the atlas closes exactly. These are nominal product dimensions;
partial/random-length cuts vary within their installation. They are not
guarantees that every visible board has the nominal dimensions.

`FloorFinish.stock` and `CarpetDesign.dye` record the new programs. Both are
optional: archived recipes without them preserve their original evaluation.
Resolved recipe fields remain authoritative; replay does not resample a product.
The generator/capture identities change when the default sampler changes.

## Runtime and annotation contract

The floor still uses one 512×512 atlas with three maps and complete mip chains.
There are no additional floor meshes, draw calls, resident texture sets or
per-frame procedural fragment evaluations. Carpet remains eligible for the
one-shot GPU baker, including authored variable cuts. Timber/mineral retain
their existing CPU bake paths. The CPU and WGSL implementations share packed
parameters, periodic hash conventions and sampling domains.

Albedo is sRGB; normal and AO/roughness/metalness maps are linear. Floor metalness
is zero. The existing variance-aware roughness mip pipeline remains in use.
The changes affect appearance; semantic classes, object bounds, physical room
geometry and camera calibration are unchanged. Normal maps still provide the
same material shading-normal behavior as other existing PBR floors.

## Local qualification

The ignored Rust test `floor_diversity_qualification` writes an opt-in study to
`out/floor_material_study/`. It compares 1,024 matched seeds for each of the three
floor substrates, using the retained generator-86 resolver as its reference.
The study is not part of the public project-page publication pipeline.

It checks physical descriptors and nominal dimensions, measures paired CPU map
generation, validates every mip of complete production map triplets, tests
serialized replay, and writes albedo/normal/roughness/GGX galleries. The GGX
panel is a fixed directional-light diagnostic, **not a rendered Bevy room**.
`ZEROVERSE_FLOOR_MAP_SAMPLES` controls full-map samples per substrate (2–64,
default 16); all 1,024 recipe pairs are evaluated regardless of this setting.

Regular tests cover seeded support, invalid authored inputs, exact tessellation
closure, nonempty random-cut partitions, legacy serialization, and distinct
fixed-seed dye interventions. `export_webgpu_material_fixtures` and
`scripts/validate_material_webgpu.mjs` exercise every dye pole, mixed programs,
authored jitter, old recipes, all mip levels and sampled texture footprints in
Chrome. The default browser runner uses SwiftShader; this is portability and
parity evidence, not hardware throughput evidence.

## Interpreting the 10× target

There is no unconditional finite count for a continuously parameterized material
space. More seeds, unique hashes or additional floating-point fields do not
establish a 10× increase in distinct learning examples. The report therefore
records fixed-bin descriptor occupancy, variance, dimensions and timings rather
than claiming a 10× effective dataset size. Fixed-seed interventions demonstrate
that the new controls change the actual material field, not just metadata.

Residual limits include repeated bounded atlases within large rooms and a finite
set of substrate models. Perceptual volume, cross-view robustness and reduced
overfitting require a separate rendered-room/embedding/downstream evaluation.
No material-only qualification establishes photographic realism or training
transfer, and CPU bake timing is not completed-room throughput.
