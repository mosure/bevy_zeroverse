# Procedural PBR surface response

Local qualification of generator **104**, capture **v129**, material-program identity **22**. Source SHA-256: `e1bbb837e7967f890830e1cf43469f1014ac73eb379686686627c60c2c753903`. Package versions are unchanged; this work has not been published.

## Rendering contract and corrections

The renderer uses Bevy's tangent-space **normal maps** and **perceptual-roughness maps**. Procedural height fields are differentiated in metric units during baking to produce normal maps; they do not displace geometry. Base color is sRGB, while normal and ORM textures are linear. ORM channels are occlusion (R), perceptual roughness (G), and metalness (B).

The review found functioning map bindings, but weak resolved relief in several rough construction types. Many sampled wood, plaster and concrete maps were close to the RGBA8 normal encoding floor. The changes retain fine polished finishes while increasing relief for open-pore/limed timber, textured plaster, cast concrete and grain leather. Plaster coverage was being suppressed twice; construction and specialization now share its corrected physical-amplitude function.

Bevy deliberately gives clearcoat a flat normal unless its separate normal texture is bound. Thin varnish, glaze and leather-finish layers now share the resolved substrate's normal map and UV channel. The binding happens after finish-slot selection, so a variant cannot retain its parent's coat normal. Planar glass remains on its separate material path. Cycles reference export now includes coat weight, coat roughness and the coat normal texture.

The shared knit template now explicitly sets its roughness multiplier to one. Actual person assembly already overrides that multiplier, so this template correction does **not** explain existing clothed-person renders. Fabric already had substantial normal variation; its relief amplitude was retained. Chrome intentionally retains an unbound normal map when its microscopic slopes are below RGBA8 precision, avoiding quantized orange-peel highlights.

The existing joint normal/GGX mip filter remains in use: unresolved normal variance broadens base-layer roughness as a surface recedes. Disabling mipmaps or increasing normal strength globally would introduce aliasing rather than a reliable material correction.

## Map measurements

The audit baked **172 map triplets** across four scene seeds, including construction/structure groups and floor styles. Its separate recipe audit sampled 1,024 seeds. The following are actual baked-map slope diagnostics for selected classes. Degrees are `atan(RMS tangent slope)`, not geometric displacement or a perceptual realism score.

| Surface | Map sets | Median slope | Range |
|---|---:|---:|---:|
| Paint/plaster | 4 | 0.896° | 0.318–1.832° |
| Fabric | 12 | 5.664° | 2.993–8.795° |
| Leather | 12 | 2.904° | 0.845–9.034° |
| Wood | 8 | 0.760° | 0.330–1.241° |
| Concrete | 4 | 0.394° | 0.318–0.812° |

A nearly flat RGBA8 normal encodes approximately 0.318° in this diagnostic, so values near that floor are not evidence of resolved relief. Smooth finishes intentionally remain in the distribution. All 172 matched map sets retain their dimensions, base-color factors and albedo moments. This small audit is not a population estimate of photographic realism or effective dataset size.

## Validation

- **31 focused Rust tests passed**, covering selected material bindings, rough versus smooth construction poles, physical normal bounds, roughness semantics, mip filtering, CPU/GPU request consistency and serial/parallel material preparation. Three tests specifically cover the new surface-response contract.
- Software Vulkan matched **193 recipes**, three maps each, over all mip levels. Maximum CPU/GPU channel difference was **1/255**; worst per-map-level RMSE was **0.03125/255**.
- Chrome/Dawn **SwiftShader** matched **87 bake/filter cases**, including trilinear and anisotropic sampling. No validation failures occurred. Maximum channel difference was **1/255**; worst map RMSE was **0.12694/255**, and worst filtered-sampling RMSE was **0.06371/255**.
- **38 PBR swatch captures** cover one-metre and 25-centimetre samples, normal-map removal, energy-preserving uniform roughness, and forward versus normal-prepass rendering. Five matched geometric-normal annotations are byte-identical across the source change. Four further rough-finish normal annotations are byte-identical with shading normals disabled.
- Forward/prepass RGB RMSE spans **0.116–0.164 byte values** over the five fixed-light swatches. Normal and roughness ablations produce measurable RGB changes; their size depends on finish, distance and illumination. The rough leather swatch's normal ablation has RGB RMSE **3.086 byte values**, while fine paint remains subtle.
- Strict workspace Clippy with CUDA IPC/WGPU tensor-fusion features, example Clippy, formatting, and the WASM viewer check with human motion passed. **15 existing Cycles-reference contract tests** and Python syntax validation passed. Blender/Cycles rendering was not run.

The production-renderer images use software Vulkan, automatic SSAA and fixed lighting. These are actual PBR renders, not generated illustrations. The rough construction gallery uses seed 7, construction pole 1, 384×384 output and 25-centimetre swatches. It is a controlled material example, not a frequency sample of rooms.

![Current rough-construction swatches](../../out/surface_response_104/rough-finishes.jpg)

[One-metre and close-up gallery](../../out/surface_response_104/current-surfaces.jpg) · [Qualification receipt](../../out/surface_response_104/qualification.json) · [Map audit](../../out/surface_response_104/after-audit.json)

## Cost and qualification limits

The changes reuse existing texture handles, atlas sizes and bake passes. They add no mesh detail or texture allocation. Coated surfaces gain one clearcoat-normal texture lookup; this is not a claim of zero render cost. No fresh hardware-GPU throughput or concurrent-training benchmark was run. The existing upstream Burn future-incompatibility notice remains.

Chrome validation used a software adapter, not macOS/Metal hardware, and exercised bake/filter shaders rather than the full browser PBR viewer. This work does not qualify every garment UV/tangent transform or establish photographic fidelity against a path-traced reference. Clearcoat roughness remains scalar; the joint normal/roughness variance filter applies to the base layer. Room-wide atlases can still lose sub-texel floor microdetail. Coarse relief that must change silhouettes, visibility or geometric targets requires geometry and is outside this shading-only change.

## Reproduction

```sh
cargo run --example audit_materials -- out/material-audit.json --map-seeds=4
cargo run --example review_materials -- out/material-swatches 7 \
  --surfaces=Paint,Leather,Wood,Concrete --construction=1 \
  --close-up --resolution=384
```

The review example also accepts `--flat-normals`, `--uniform-roughness` and `--no-normal-prepass`. It records the settings, selected material bindings, maps, build provenance, RGB captures and geometric-normal outputs. The uniform-roughness control preserves the map's mean fourth power, consistent with the existing GGX mip filter.
