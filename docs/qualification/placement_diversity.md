# Supported-object placement and scale qualification

Generator revision 52 replaces fixed shelf columns and mirrored sofa pillows with seeded, bounded placement programs. Shelf contents, refrigerator supplies and display-case contents vary in count, position, scale and yaw. Counters and dining/work surfaces sample larger loose-item rotations while fitted equipment retains its practical facing. Pillows vary independently on supported seat modules. Ceiling circuits use staggered, irregular spacing and continuous orientation; fixture bodies, normals, suspension anchors, clearance reservations and reflection meshes share that orientation.

Size is sampled in metres before admission. Loose-item sizes share a category-aware sampler: round vessels stay round, paper and phones retain plausible aspect ratios, and dish envelopes follow the actual plate/bowl/cup/glass construction. Storage contents keep their sampled physical sizes; items exceeding headroom or footprint clearance are omitted instead of compressed to fit. Display contents use metric product dimensions instead of scaling to the display bay. Pillows span lumbar-to-square proportions with independently sampled width and thickness. Within a room, ceiling fixtures vary in width, length and body thickness while sharing a construction style; the same realized size drives clearance, mesh construction, suspension anchors, direct-light radius, emitter position and emissive/reflection radiance. Per-fixture metric export records realized dimensions and emitting areas.

Round vessels reserve a rotation-independent radius including handles/rims. Cartons use rotated box bounds. This prevents an artificial preference for axis-aligned vessels during admission. Every accepted layout prop still passes the existing support, outline, peer, cable and room clearance checks. Candidate budgets remain finite: at most six items per storage shelf, twenty attempts per item, and four loose pillows per sofa. Sampling adds no runtime model or simulation dependency.

## Local CPU results

| Population | Result |
| --- | --- |
| 256 seeds, two storage categories, three shelves each | 3,872 items; 0–6 per shelf; 54/64 support-position cells occupied |
| All stored items | All 16 yaw bins occupied; 4.52% within 2° of a support axis |
| 990 handled cups and cartons (rotation changes visible geometry) | All 16 yaw bins occupied; 4.85% within 2° of an axis |
| 4,058 same-shelf item pairs | 6.97% within 5 mm of the same depth row |
| Accepted storage sizes, separately within each dish/pantry type | 95th/5th-percentile width ratio 1.73–2.87×; 712 shelves contain items differing in width by more than 50% |
| 256 sofa seeds | Pillow-count histogram for 0–4 pillows: 73, 39, 53, 43, 48; 33/64 position cells; 137 asymmetric multi-pillow arrangements |
| Accepted pillow mesh diameter, 5th–95th percentiles | 0.237–0.503 m (rotation-independent bounding diameter) |
| 22 loose-item categories × 128 seeds; 4 display-content categories × 128 seeds | 3,328 meshes passed physical envelope and semantic geometry checks; display contents also checked against their own shelf-contact plane |
| 64 complete rooms, 3 cameras each | Layout, supported objects, mesh geometry, camera clearance and semantic part identities passed; 3,867 objects |
| Accepted ceiling-fixture widths, 5th–95th percentiles | 0.284–1.696 m; 55/64 rooms have more than 40% width variation within the room |
| 5,520 ceiling-fixture pairs in those rooms | 2.36% within 5 cm of the same room-axis row or column; all 16 fixture yaw bins occupied |

Accepted supported-item width distributions (5th–95th percentiles; same category, after clearance checks):

| Category | Samples | Width (m) |
| --- | ---: | ---: |
| CoffeeCup | 57 | 0.062–0.097 |
| Dishware | 59 | 0.110–0.323 |
| Mug | 43 | 0.088–0.149 |
| Notepad | 47 | 0.088–0.207 |
| Phone | 132 | 0.060–0.085 |

Room seeds are 0–15 for each of Mixed, Kitchen, Kitchenette and Lounge, at density 0.75 and human density 0.25. Shelf fixtures are 0.8 × 0.28 m; all shape vertices are checked against shelf bounds and headroom. Shelf yaw statistics distinguish asymmetric geometry from invisible rotations of round plates. Position cells are 8 × 8 bins over the complete support rectangle; boundaries are unavailable to finite-size items, so full occupancy is neither expected nor required.

Native library/test Clippy passed with warnings denied; the WebGPU/viewer library compiled for `wasm32-unknown-unknown`. Formatting and diff checks passed.

The 38 focused tests passed in 12.77 seconds (the complete-room sub-audit took 8.26 seconds on the shared host). These times include validation and are not capture-throughput measurements. RNG replay, cupboard/appliance semantics, cabinet opening envelopes, kitchen support, short/deep sofa bounds, ceiling reservations and luminaire reflection oracles are covered. Actual fixture mesh projected emitting areas are independently checked against the area used for photometry. Miniature display sculptures use proportionally bounded connecting rods to prevent fixed-thickness details from piercing shelves.

Release regression coverage additionally includes 512 narrow/tall book-stack meshes, the geometry matrices for seeds 57541088 and 57491588, and the eight-camera Library configuration for seed 63343058. Spiral bindings account for the complete wire cross-section inside the rotated cover envelope. Large-rig recovery reserves room for new proposals when replay would exhaust its budget and samples independent horizontal travel within the configured path-length bounds. The eight-camera test checks deterministic replay, unchanged scene geometry, all requested overlap/spread/separation constraints, swept clearance and 129 positions along every path. Ordinary successful camera searches retain their existing results and cost; recovery keeps the same finite proposal budget.

The final native CPU unit suite passed 681 tests plus 14 isolated voxel tests. Its 36 explicitly ignored GPU/diagnostic tests were not launched. Workspace all-target Clippy also passed with warnings denied, including the CUDA IPC and Burn fusion features.

```sh
cargo test --lib --no-default-features --features multi_threaded -- \
  placement::tests items::sizing objects::seating objects::kitchen objects::public \
  architecture::fixtures architecture::ceiling layout::surfaces layout::kitchen \
  gi::reflection_emitters --nocapture --test-threads=2
```

Machine-readable local results and logs are in `out/scale_diversity/`. No new GPU renders were performed while training was active. These results establish seeded placement variation and CPU geometry correctness; photographic realism, rendered annotation parity and reduced downstream memorization are not claimed by this audit. Existing publication captures were not replaced.
