# Procedural surface construction and qualification

Material recipes describe construction as well as pigment. Floor installation,
wall covering and manufactured finish parameters are serialized with the scene;
seed streams are independent of furniture, lighting, cameras and human poses.
These surface changes do not move vertices or change semantic labels. Relief
remains a normal-map effect, not displaced annotation geometry.

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

Capture manifests contain the full recipes. Dataset metrics additionally record
floor-installation counts, piece variation, wall component strengths and
manufactured surface scales. Counts of parameters, seeds or exact hashes are
not estimates of perceptual dataset size or training utility.

## Verification

The ignored Rust test `surface_space_qualification` writes a private report and
gallery to `out/material_diversity/`. It validates 896 recipes across 14 roles,
checks 119 production map triplets including every mip, and repeats 28 triplets
for exact replay. Tests check tessellation coverage, wrapping, metric joint
distances, legacy deserialization, scalar/prepared agreement, shader parsing,
GPU uniform mapping, and safe unsupported-program fallback.

The current CPU qualification passes 202 material, envelope, architecture,
appearance and metrics tests, plus the map study. A 32-room wall-assignment
check confines coverings to inward wall faces; eight of those rooms also compare
every triangle's position, normal, UV and semantic label against the same room
without feature-wall assignment. Strict workspace Clippy and the WASM library
build with web/human-motion features pass. These are CPU/compiler checks, not
hardware rendering qualification.

Gallery rows show actual albedo, normals and roughness, followed by a fixed CPU
GGX lighting diagnostic. That diagnostic omits scene illumination, reflections,
clearcoat and anisotropy. It cannot establish rendered scene realism. CPU map
timings exclude rendering, readback and training; they are not generation
throughput measurements. Hardware GPU parity, browser runtime, rendered
annotation equality and concurrent-training throughput require a separate
qualification window. The WebGPU fixture exporter includes isolated wall
components for that check. No local study artifacts are published automatically.
