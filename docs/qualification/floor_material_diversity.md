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
| Installation | Seven layouts, each with physical scale variation. Product width/aspect, atlas extent, color and finish have separate seed domains. Small mosaics, large slabs, narrow/wide boards, parquet blocks, chevron lanes and hexagonal tiles all vary. |
| Cut stock | Uniform manufactured products and continuously variable cut stock coexist. Bounded periodic boundary displacement preserves shared seams; minimum cut widths prevent slivers. Parquet and hexagonal installations retain their exact tessellations. |
| Joints | Independent dark/light grout reflectance, metric widths and roughness. Area coverage filters subpixel joints; grout is mixed in linear light. Carpet retains textile seams. |
| Carpet composition | Continuous normalized mixture of heather, striae, offset weave, clouds, stipple and yarn blocks. Six poles and their mixtures share independent dye scale, warp, shear, contrast and palette controls. |
| Carpet microstructure | Independently sampled nap, slub and fleck frequencies; no universal fixed fiber-frequency signature. Relief, roughness and dye share the same substrate field. |
| Carpet scale/regularity | Dye motifs are sampled in metres independently of modular tile size. Broadloom accounts for 40% of carpet draws; the rest continuously vary tile definition from subtle to pronounced. Broadloom has no tile-by-tile roughness discontinuities. |
| Lot variation | Independent piece reflectance/warmth/roughness and variable-scale mottling rather than one universal macro-noise frequency. |

The sampler uses log-space physical size priors so fine products retain support
alongside large formats. Requested dimensions are:

| Product | Dimensions before periodic closure |
| --- | --- |
| Rectangular timber | Width 0.065–0.40 m; aspect 3.5–18; length bounded to 0.32–2.8 m |
| Carpet tiles | Width 0.25–1.0 m; length bounded to 0.25–1.5 m |
| Mineral tile/slab | Width 0.07–1.6 m; length bounded to 0.08–2.2 m |
| Basket parquet | Block width 0.20–0.95 m, containing 3–5 strips |
| Herringbone | Plank width 0.06–0.27 m for timber, 0.08–0.50 m for mineral; length 2–5 widths |
| Chevron | Lane width 0.30–1.2 m; course height 0.065–0.28 m; varied inclination |
| Hexagonal | Lattice width 0.08–0.80 m |
| Carpet dye | Motif spacing 0.10–3.0 m; separate dye/warp scale 0.15–3.0 m |

Square stock remains common for manufactured mineral and carpet tiles. Piece
counts are rounded for exact periodic closure, so realized dimensions differ
slightly from these requested values. Atlas extent is sampled over 1.6–6.4 m
(0.55–3.2 m for seamless mineral), adjusted where necessary to fit the stock
without collapsing narrow or large-format samples onto count limits. Variable
cuts preserve at least 4 cm width, and timber at least 12 cm length. These are
product dimensions, not guarantees about the visible fraction at room edges.

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

The ignored Rust test `floor_scale_qualification` writes a current-state study to
`out/floor_scale/`: 4,096 seeds per substrate, dimension/regularity distributions,
18 complete production-map triplets and three serialized map replays. Galleries
cover the same **4 × 4 metre** footprint at different size quantiles; they sample
the real mip chains rather than stretching every atlas to the same visible size.
Rows show albedo, normal, roughness and a fixed GGX lighting diagnostic. The GGX
panel is **not a rendered Bevy room**. Sampler and CPU map timings are recorded.
This local study is not part of the public project-page publication pipeline.

Regular tests cover seeded support, invalid authored inputs, exact tessellation
closure, minimum cut dimensions, scale support within each layout, broadloom
without hidden tile-gloss grids, legacy serialization, and distinct
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
