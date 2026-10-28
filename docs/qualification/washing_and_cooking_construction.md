# Washing stations and cooking construction

Local qualification of generator **109**, capture **v134**, source SHA-256 `b2cb5485bd5a092bd9e6278d150d0226c6da6faaa8a691fc2a42879d03d6dfdb`. Package versions are unchanged. This iteration is not committed or published.

## Review and changes

The 107-category object audit was reviewed together with the construction programs. Low depth variation in artwork or a closed manufactured enclosure is not, by itself, a reason to distort that geometry. This pass targets functional repetitions: sink apertures/basin splits, tap proportions and accessory corners; dishwasher rack heights; oven cavity and shelf proportions. It also corrects plastic hot-cavity linings and appliance-frame semantics.

| Area | Realized construction and appearance | Fit contract |
|---|---|---|
| Sinks | Offset rectangular apertures, one or two unequal rounded bowls, independent bowl depths, wall tapers, corner roundness and drain locations | Continuous surrounding worktop; a real open cavity and closed supported floor. Outer rim interfaces stay rectangular while the inner bowl can be rounded. |
| Taps | Variable mount position, lateral reach, height, thickness and handle spacing; angular spouts and bounded-curvature goosenecks | The outlet projects into a real bowl; mounts and handles reserve worktop space; all geometry fits the original admitted envelope. |
| Washing accessories | Independent soap/sponge/rack sizes and headings, shuffled bounded placement attempts, fitted dish styles and positions | Rotated extents stay on actual worktop slabs and clear the cutout, tap mounting area and peers. Dishes remain inside the tapered bowl floor. Requests that cannot fit are omitted and actual counts are exported. |
| Finishes | Independent metal/ceramic/composite-like concrete basin choices and chrome/metal/black-coated taps | Finish interventions preserve triangle positions and normals. Existing PBR recipes and requested texture slots are reused. |
| Dishwashers | Two fitted baskets or two baskets plus a shallow upper tray; varied heights, proportions and offsets, with real support rails | Shelves and contents fit the cavity and retain headroom. Rack frames are labeled dishwasher; plates remain dishware. |
| Ovens and ranges | Varied cavity widths/heights, unequal double-oven chambers, one to three racks, wire pitch/orientation and baking-sheet dimensions/offsets | Door travel is derived from the same front dimensions used by the mesh. Shelves connect to cavity supports; baking sheets rest on rack wires. |
| Hot interiors | Metal linings in ovens, ranges, microwaves and dishwashers | Refrigerators retain appropriate plastic lining; decorative fronts and exterior controls retain their own materials. |

The programs live under `objects/kitchen/sink/` and `objects/kitchen/appliances/`. Geometry, bounds, audit metadata and dataset distribution metrics consume the same seeded programs. No new configuration switch, external asset, model load, material bake or render pass was introduced.

## Sampling evidence

Across **256 washing-station samples**, the admitted programs contain **82 centimetre-width bins**, **325 joint roundness/taper bins**, **nine basin/tap substrate pairs**, **100 visibly unequal two-bowl splits**, **587 admitted accessories**, **586 quantized accessory poses** and **208 quantized tap-position/reach bins**. Bins are diagnostics of this bounded cohort, not independent dataset-size estimates. Accessory counts are `{"DryingRack": 87, "SoapDispenser": 359, "Sponge": 141}`.

| Measured field | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| props_per_sink | 0.000 | 4.000 | 2.293 |
| bowl_width_m | 0.236 | 1.120 | 0.549 |
| bowl_depth_m | 0.277 | 0.434 | 0.353 |
| bowl_drop_m | 0.130 | 0.234 | 0.183 |
| bowl_roundness | 2.425 | 7.993 | 5.217 |
| bowl_taper | 0.650 | 0.910 | 0.786 |

The 128-seed appliance fitting test covers both two- and three-level dishwasher constructions and all one/two/three-rack oven counts, with **151 basket-height bins** and **159 cavity-height bins** at millimetre resolution. Complete-room cohorts contain `{"Dishwasher": 44, "KitchenSink": 81, "Oven": 26, "Range": 43}` among the four revised categories.

| Category | Shape representatives at 0.005 / 0.015 / 0.03 | Mean triangles | Median nearest-shape distance |
|---|---:|---:|---:|
| KitchenSink | 64 / 64 / 51 | 4,670 | 0.0333 |
| Range | 54 / 36 / 27 | 11,809 | 0.0098 |
| Oven | 57 / 32 / 27 | 7,520 | 0.0101 |
| Dishwasher | 53 / 37 / 24 | 8,129 | 0.0104 |

Shape coverage uses six envelope-normalized 24×24 orthographic depth descriptors per object. It excludes RGB and is threshold dependent. The catalog retains its existing fixture policy: kitchen inputs include seeded metric dimensions and reserved door travel. Separate closed-chassis tests hold appliance dimensions fixed. Captured examples are selected to expose conditional constructions; they do not estimate their frequency.

## Correctness

- **59 focused tests pass**, covering actual mesh orientation/bounds, deterministic replay, independent finish interventions, real worktop and basin ray intersections, accessory support/clearance, basket/cavity dimensions, door and opt-in motion ownership, kitchen room admission, semantic labels and catalog contracts. One pre-existing ignored kitchen population test was not run; explicit population audits follow.
- **6,848 meshes across 107 categories** pass. Additional targeted checks cover 256 washing stations and 128 seeds of dishwasher, oven and range construction.
- **240 full rooms** pass CPU layout and real-mesh validation: 128 mixed, 16 dense eight-camera rooms and 32 each of kitchens, kitchenettes and cafes.
- **72 production-renderer images** cover RGB, geometric normals and semantic labels for 24 selected fixtures. They use software Vulkan with two threads. Galleries are inspected and semantic palette identities checked.
- Strict workspace/all-target Clippy with CUDA IPC and WGPU fusion features, formatting, native audit/review builds and WASM viewer compilation with human motion pass. The upstream `burn-cubecl-fusion 0.22.1` future-compatibility notice remains.

Validation exposed and corrected folded annular triangles at mismatched corner parameterizations, overly tight faucet bends, narrow-appliance basket clearance and bevel grooves along shared worktop slab edges. The final basin rings follow shared metric rays; goosenecks retain a bounded bend radius. No mesh-validation tolerance or placement-clearance rule was relaxed.

## Bounded cost and limitations

The 6,848-object corpus has **19,929,903 triangles**, versus **19,554,167** in the saved local baseline (**+1.92%**). Bowls use a fixed 32-side mesh, rack counts and wire counts are bounded, and each optional washing accessory has at most 24 placement attempts. No additional room-level placement search was added.

Three sequential matched-seed CPU audit pairs measured median wall times of **4.424 s** for the candidate and **4.523 s** for the saved baseline, each for 128 rooms. The median paired ratio is **1.044** (range **0.885–1.073**); additional pairs alternate execution order. These runs include real-mesh validation and diagnostic JSON export, whose kitchen fields expanded in this iteration. Generated contents differ. They are bounded CPU observations, not completed-render or concurrent-training throughput measurements.

No fresh hardware CUDA replay, browser execution, full-room RGB study or long-run memory plateau was qualified here. Manufactured enclosures remain appropriately box-like; the new internal variation is most visible in open appliances and sink views. Photographic realism, effective independent dataset size and reduced downstream memorization are not established by these checks.

[RGB examples](../../out/diversity_review_109/renders-color.jpg) · [normal examples](../../out/diversity_review_109/renders-normal.jpg) · [semantic examples](../../out/diversity_review_109/renders-semantic.jpg) · [sampling distributions](../../out/diversity_review_109/washing-distributions.png) · [qualification receipt](../../out/diversity_review_109/qualification.json)

All commands, source identities, tests, distributions, comparative CPU observations and capture hashes are retained in `out/diversity_review_109/`. Prior local work remains intact. Generator identity changed; previous generated-scene caches are not bitwise equivalent.
