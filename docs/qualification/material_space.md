# Procedural surface construction and qualification

Material recipes describe construction as well as pigment. Floor installation,
wall covering and manufactured finish parameters are serialized with the scene;
seed streams are independent of furniture, lighting, cameras and human poses.
These surface changes do not move vertices or change semantic labels. Relief
remains a normal-map effect, not displaced annotation geometry.

## Construction and finish sampling

Each supported substrate samples four correlated construction regions with
continuous blend weights, physical feature scale and finish variation. The
regions guide the existing physical fields; they are not four fixed textures.
Texture phase, pigment, weave topology, installation, cut and wear parameters
retain their independent seeded variation.

Blend concentration also varies continuously, filling the intermediate finishes
between the poles. Selected substrate controls retain a 20% independent seeded
residual around the prior, rather than collapsing many physical controls onto
the mixture weights. This keeps within-process variation alongside the wider
between-process range.

| Substrate | Construction regions |
| --- | --- |
| Woven/knitted cloth | linen-like weave, twill, chenille pile, loop knit |
| Leather | aniline, pebbled, nubuck, sealed hide |
| Timber and timber floors | fine veneer/grain, open pores, limed/washed stock, dark/sealed finish |
| Paint and ceilings | rolled film, orange peel, knockdown, trowelled plaster |
| Accent walls | mineral wash, longitudinal fibers, woven covering, printed covering |
| Ceramics | porcelain, speckled stoneware, reactive glaze, crazed glaze |
| Concrete | smooth cast, board formed, trowelled, ground aggregate |
| Mineral floors | honed stone, exposed aggregate, veined stone, polished mineral |
| Carpet and rug pile | heather, striated pile, modular weave, dense felt |
| Chrome/alloy | mirror polish, satin, directional brushing, bead blasting |
| Coated metal | fine film, peened coating, directional satin, powder coat |
| Plastic/rubber/paper | category-specific gloss, grain, grip, porosity and fiber finishes |
| Foliage, soil, terracotta and bark | category-specific wax, bloom, venation, inclusions, pores and fissure scales |

Correlations keep finishes physically plausible: leather nap suppresses coating,
filled timber pores accompany sealed finishes, polishing reduces mineral relief,
and leaf wax reduces transmission. Texture scale is in metres; changing it does
not allocate a higher-resolution map. The textile sampler retains repeat lengths
2–7, bundles 1–3, arbitrary valid over/under shifts and herringbone variation
inside the construction regions. Rug composition, borders and palette remain
independent of pile finish.

`materials::construction::ConstructionSample` records four normalized weights,
a scale multiplier and a variation factor. `construction::apply` validates an
authored sample and resolves it transactionally into a `MaterialRecipe`.
`construction::recipe_labels` describes the category-specific poles, including
the separate rug-pile interpretation. Recipes serialize
both this sampling provenance and the resolved fields; the resolved fields are
authoritative for replay. Removing provenance does not change an archived
recipe's texels. Previously archived recipes without the field retain their
evaluators. Material-detail interventions, including zero relief, remain honored.

This expansion does not alter glass optics, screen/art content, emissive
surfaces or skin/hair programs. Their existing samplers remain separate. Four
conditioned construction alternatives is a concrete sampling target; it does not
establish four times the perceptual or effective training-dataset size.

## Floors

Timber installations include straight lanes, broken bond, basket parquet,
herringbone and chevron. Mineral floors include straight/broken-bond tiles,
rectangular herringbone, hexagonal tile and seamless poured finishes. Carpet
retains its broadloom/modular installation and dye/pile programs. Width, length,
aspect, joint size, pigment, cut-to-cut warmth and polish remain variable.

Herringbone is an exact periodic tessellation of equal-area rectangular planks.
Chevron courses use metric distances perpendicular to their sloped joints.
Hexagonal cells use a periodic two-sublattice Voronoi construction. Grain,
piece identity and grout share the same cut function. Seamless mineral fields
have neither tile-local coordinate resets nor fictitious grout; their smaller
physical repeat and aggregate spacing describe poured/ground surfaces. Mineral
recipes vary binder, chip size/color, aggregate exposure, veins and polish.

## Walls and other substrates

Wall programs combine quiet paint/plaster with optional limewash, longitudinal
fibers, woven wallcoverings and printed bands. Component weights, dye strength,
strand spacing, warping, relief and local roughness vary continuously. Coverings
use a larger physical repeat than paint film, and suppress heavy plaster relief
under fibrous coverings. General paint uses only washed/brushed coatings; woven
and printed coverings use the feature-wall role. Service pipes and fittings do
not share that role. Ceiling paint remains a separate, quieter role.

Primary envelopes select up to two feature-wall edges, weighted by opaque area.
Only their inward faces receive the covering. Exterior faces, cutout reveals,
baseboards and cornices retain their own coatings. This includes the irregular
envelope path, rather than leaving the feature material unused in modern rooms.
Selection is seedable, and realized edge counts are exported in dataset metrics.

Metal plating, coated metal, plastic and rubber sample independent surface
scales, directional grain, peening and finish roughness. Chrome retains metallic
response; coated metal and polymers remain dielectrics. Timber, glazes, mineral
aggregates, textiles, leather, foliage and content surfaces continue to use
their own substrate programs. There is no universal noise texture applied to
all materials.

## Cost and export contract

Existing material roles and their atlas budgets are reused; no per-piece or
per-object maps are allocated. A room that previously omitted the feature-wall
role can now use one extra 256-square triplet (1,048,572 bytes including mips)
and one additional merged wall draw. The number of selected walls does not
multiply that cost. Atlas resolutions remain 256 or 512, with full sRGB albedo,
linear tangent normal and AO/perceptual-roughness/metalness mip chains. Normal first moments and
roughness fourth moments retain their existing joint filtering. Source texels
and cached maps are evaluated once, not in each camera's fragment shader.

Wall and manufactured fields have matching CPU and WGSL implementations in the
existing bake kernels. Hard floors retain their CPU synthesis path. Unsupported
authored carpet installations explicitly use CPU synthesis rather than silently
changing the layout. GPU recipe uniforms grow by seven vec4 values (112 bytes);
texture and scratch-buffer sizes do not grow. Existing serialized recipes can
omit all new optional fields and retain their original evaluators.

Construction priors resolve once into those existing fields. They add no WGSL
branches, map resolutions, texture slots or draw calls. Lazy per-room structure
groups, twelve object finish slots and four rug slots retain their budgets.

Capture manifests contain the full recipes. Dataset metrics additionally record
floor-installation counts, piece variation, wall component strengths and
manufactured surface scales. Counts of parameters, seeds or exact hashes are
not estimates of perceptual dataset size or training utility.

## Verification

`cargo test --lib construction_map_qualification -- --ignored --nocapture`
writes `out/material_expansion/study/`. It checks four conditioned map triplets
per supported role/floor type plus rugs, every mip, exact replay and 128-seed
descriptor distributions against the unconditioned sampler. The report states
the descriptor, physical bin widths and sampling horizon. Exact hashes establish
determinism; bin occupancy is a diagnostic, not a learned visual-distance metric.

`review_materials` accepts `--construction=0..3` to capture each physical pole
with identical seed, lights, geometry and camera. Combine it with
`--surfaces=Fabric,Leather,Wood,Ceramic,Concrete,Chrome` and `--close-up` for
production PBR inspection. Use `--floor-style=0..2` for floor treatments.
Construction overrides cannot be combined with `--variant`, which resamples
the structure. GPU parity qualification covers all four process poles at the
authored physical-scale extremes, using the existing strict map/mip
tolerances.

The ignored Rust test `surface_space_qualification` writes a private report and
gallery to `out/material_diversity/`. It validates 896 recipes across 14 roles,
checks 119 production map triplets including every mip, and repeats 28 triplets
for exact replay. Tests check tessellation coverage, wrapping, metric joint
distances, legacy deserialization, scalar/prepared agreement, shader parsing,
GPU uniform mapping, and safe unsupported-program fallback.

The local construction qualification passes 156 material-related tests and two
appearance tests. The map study checks 104 conditioned triplets across 26
role/floor/rug cases, plus 26 reference triplets and 26 exact replays. A 32-room
wall-assignment check confines coverings to inward wall faces; eight of those
rooms also compare every triangle's position, normal, UV and semantic label
against the same room without feature-wall assignment. Strict library/test/review
example Clippy and the WASM library build with web/human-motion features pass.

Native Vulkan qualification on an RTX PRO 6000 Blackwell compares 188 recipes
across all base maps and mip levels against the CPU oracle: maximum error is
three encoded byte levels and maximum per-map/mip RMSE is 0.25 byte levels.
The paired CPU/GPU PBR minification check also passes. A separate 32-view PBR
gallery holds geometry, camera and key/fill parameters fixed while varying
construction, and checks finite normal exports. The production reflection
environment responds to room surface changes; it is not a fixed external HDRI.
These bounded native checks ran alongside training, without a renderer pool or
additional model loading. They are not a completed-room throughput benchmark or
a Chrome/Metal runtime qualification.

Gallery rows show actual albedo, normals and roughness, followed by a fixed CPU
GGX lighting diagnostic. That diagnostic omits scene illumination, reflections,
clearcoat and anisotropy. It cannot establish rendered scene realism. CPU map
timings exclude rendering, readback and training; they are not generation
throughput measurements. Browser runtime, full-room annotation equality and
concurrent-training throughput require separate qualification. The WebGPU
fixture exporter includes isolated wall components for that check. Texture
budgets are unchanged, but sampling different existing substrate programs can
change synthesis cost, particularly knitted versus woven cloth on CPU.
No local study artifacts are published automatically.
