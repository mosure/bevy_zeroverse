# Stationery, cabinet joinery and exhibit supports

This local qualification reviews the current 107-category object catalog and addresses remaining narrow construction and placement regimes. The source is generator **103**, capture **v128**, SHA-256 `812d1160442b99ed5e12f6c89d00db9e07e6ee46feb2f835c096399f5fba630d`. Package versions are unchanged; these changes have not been published.

## Implemented sampling

| Area | Construction and appearance | Physical contract |
|---|---|---|
| Notepads | Glued, top-wire, side-wire, clipboard and loose-sheet construction; independent backing, page inset/offset, corner shape, thickness, binding width and ring pitch | Backing contacts the support. Binding hardware fits the pre-admitted height. The real page top receives one complete document, replacing the former metre-UV crop without adding a coplanar overlay. |
| Writing tools | Sharpened pencils, mechanical pencils and capped pens; independent barrel section, length, radius, taper, grip, eraser, tail and clip dimensions; wood, plastic and metal substrates | All geometry fits the admitted dimensions and touches the support. Pocket clips use fitted bent-sheet components rather than a self-intersecting tight tube sweep. Existing stationery semantics are preserved. |
| Printed paper | Plain, ruled, squared, dotted, agenda and sketched-note programs; variable margins, independent horizontal/vertical pitch, pale paper colors, ink, writing occupancy, slant, columns and matte roughness | One 256-pixel mipmapped color map per requested slot. Documents sample independently of furniture finish identity. The existing bounded 12-slot paper capacity is retained. |
| Cabinets | Independent rail/stile proportions, recessed/raised field depths, horizontal/vertical reed pitch and fill, and one to six requested glazing cells, fitted to the actual leaf | Existing outer front and motion envelopes remain unchanged. Pane apertures and mullions are disjoint. Pulls and drawer handles mount on real rails/stiles, including on islands; hanging towels follow the handle location. |
| Exhibit plinths | Tapered/waisted bodies, exposed legs, paired blades, cruciform supports and stepped bodies; variable base/crown/waist spans, heights, roundness, cap thickness and finishes | Ground contact and the exact flat rectangular exhibit-support plane are retained. This does not introduce round support tops. |
| Workstation props | Keyboard, mouse, phone and paper proposals use the shared metric size sampler | Dimensions are sampled before the existing support and peer-clearance gates. This removes constant mouse dimensions and paper thickness from the workstation-specific path. |

Construction programs are isolated in `clutter/tabletop/{pad,writing}.rs`, `objects/kitchen/cabinet/fronts.rs` and `objects/public/casework/plinth.rs`. Audit/export metadata and distribution metrics use those same programs. Cabinet metrics distinguish requested glazing subdivisions from the cells that fit a particular leaf. RGB changes do not rewrite semantic identities.

## Measured shape coverage

The audit constructs 64 seeds per category and rasterizes six 24×24 orthographic, envelope-normalized depth descriptors. The following are greedy representative counts at distance **0.015**, out of 64. They are a shape diagnostic, not independent training-sample counts or an effective dataset-size estimate.

| Category | Representatives | Mean triangles |
|---|---:|---:|
| Notepad | 64 | 768 |
| Pencil / writing tool | 62 | 389 |
| Exhibit plinth | 63 | 492 |
| Kitchen counter | 28 | 4,808 |
| Kitchen cupboard | 63 | 6,915 |
| Pantry | 24 | 5,722 |

The stationery and plinth construction changes substantially improve this diagnostic. Cabinet joinery is smaller than the descriptor resolution: counter/cupboard coverage is essentially unchanged and pantry coverage changes only modestly. Do not present cabinet detail parameter counts as evidence of a large geometric-space expansion.

Desktop and public-object fixtures hold their metric envelopes fixed. Kitchen fixtures retain their seeded dimensions, including reserved door travel; the same fixture policy is used in both local comparison samples. RGB, room placement, human appearance and downstream learning are outside this descriptor. The complete catalog and matched comparison are in [`out/diversity_review_103/comparison.json`](../../out/diversity_review_103/comparison.json).

## Correctness and visual inspection

- **99 focused tests passed.** Coverage includes actual triangle topology/normals, replay, metric size extremes, page UV extent, removal of the duplicate paper top, material selection/parallel preparation equivalence, cabinet hardware attachment to actual mesh faces, fixed plinth support area, room support/clearance and existing motion construction/admission.
- **6,848 object meshes** passed across all **107 categories**.
- **240 furnished rooms** passed layout and actual geometry checks: 128 mixed rooms, 16 dense eight-camera rooms, and 32 each of kitchens, kitchenettes and galleries. This includes 16,458 room-object envelope checks and the prior cafe-capacity regression seed `66369950`.
- **108 captures** were inspected: six shape-selected examples in each of six categories, in RGB, world-normal and semantic modes. These use a persistent software-Vulkan renderer, 320×320 outputs and automatic SSAA. The fixed finish seed and studio lighting isolate construction; the gallery is not a frequency sample of room appearances.
- Formatting, strict workspace Clippy with CUDA IPC/WGPU tensor-fusion features, and the WASM viewer build with human motion passed. The existing upstream `burn-cubecl-fusion 0.22.1` future-incompatibility notice remains; it is not a new Clippy warning.

The test run found and corrected the short-pen tube-clip inversion. Geometric review also corrected floating pulls over recessed cabinet fields. No validation tolerance or placement clearance was relaxed.

[RGB gallery](../../out/diversity_review_103/products-color.jpg) · [normal gallery](../../out/diversity_review_103/products-normal.jpg) · [semantic gallery](../../out/diversity_review_103/products-semantic.jpg)

## Cost and remaining limits

The 6,848-object catalog contains **19,255,655 triangles**, 0.18% fewer than the preceding local sample. The cached CPU constructor sweep took **621 ms**, versus 651 ms in the preceding sample; these short, unpaired wall-clock timings do not establish a throughput speedup. Writing-tool detail costs more per item, while thinner cabinet construction and pad geometry offset that cost in the catalog aggregate.

Reeds and wire rings have bounded counts, hardware is batched by existing material/semantic keys, and unused paper maps are not generated. Paper preparation uses the existing native worker queue and retains the cooperative WASM path. No extra render pass, per-frame synthesis, external image asset or model load was added.

Fresh hardware-GPU performance, browser visual behavior, concurrent-training throughput and long-run memory were not qualified here. The gallery is still synthetic; photographic realism and reduced downstream memorization are not established. Elevator envelopes, closed tall storage and planar displays remain low-coverage categories in the coarse shape audit, but low depth coverage alone is not evidence that standardized physical geometry should be arbitrarily distorted. Further expansion should be assessed through visible functional construction, installation relationships and RGB/material descriptors.

All evidence, exact commands, source identities and capture hashes are recorded in [`out/diversity_review_103/qualification.json`](../../out/diversity_review_103/qualification.json). The changed generator identity invalidates bitwise reuse of older generated-scene caches.
