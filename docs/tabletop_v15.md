# Tabletop and material programs, v15

The indoor generator now samples hollow mugs, tapered takeaway cups, bottles, soda cans, notepads, pencils, conference microphones and phones. Dimensions, yaw, vessel taper, fill level, wall thickness, closures and finishes vary independently where applicable. Tabletop footprints still pass the support-boundary and overlap rejection used by the room planner. Existing scene modes remain available.

Mugs have swept handles and optional saucers. Takeaway cups have kraft/printed sleeves or double-wall bodies and optional stepped lids. Bottles have ribbed PET, long/wide necks, metal bodies and loop caps. Cans have rolled rims, hollow pull tabs and sampled label bands. Pads have covers, pages and optional spiral wire; their ruling and writing are mipmapped printing. Phones can face up or down, with mobile-specific screen layouts or rear cameras. Microphones sample a boundary puck or a curved gooseneck.

Laptops vary hinge angle, aspect ratio, chassis/lid/bezel thickness, keyboard width, key count, trackpad size, corner radius, ports and speaker grilles. The lid sweep stays inside the support envelope. Displays vary panel proportions and thickness, bezels, stand construction and finish. Screens sample document/editor/chart/calendar/meeting/lock/asleep content, theme, accent, time and luminance. Clocks sample analog/digital, round/square casings, finish, dial contrast, bezel width and time; analog hands use the same fractional second value.

Hard surfaces now have correlated color, roughness and metric normal maps, with restrained machining/molding relief. Metallic-roughness maps retain the metallic channel. Wood and ceramics have finish-dependent clearcoat. Object finish variants share the surface maps, with twelve deterministic slots per scene; generated screen and print images are 256² with full mip chains. Upholstery follows the scene palette while small accessories can vary in hue. Indirect-light proxies resolve the same finish variants as the visible meshes. No external prop meshes, images or model loads are required.

[Beverage close-ups](evidence/tabletop_v15/beverages.png), [pads/phones/microphones](evidence/tabletop_v15/tabletop.png), [computers and clocks](evidence/tabletop_v15/devices.png), and [all sixteen room views](evidence/tabletop_v15/rooms.png) show the result. The studio gallery contains illustrative seeds; room views are **consecutive seeds 0–7 without filtering**. During review, excessive metal relief and paper staining were reduced, raised notepad lines were replaced with filtered printing, and microphone support bounds were corrected.

## Bounded distribution check

The audit covers seeds 0–127 at the validator defaults, with two cameras and rotation augmentation. All 128 layouts passed; actual mesh checks covered the first sixteen rooms. Counts below include only primary-room instances, with zero-count rooms retained in the distributions.

| Object | Instances | Rooms containing it / 128 |
| --- | ---: | ---: |
| Mug | 297 | 97 |
| Takeaway cup | 319 | 92 |
| Water bottle | 520 | 111 |
| Soda can | 344 | 100 |
| Notepad | 145 | 75 |
| Pencil | 537 | 107 |
| Table microphone | 23 | 16 |
| Phone | 297 | 100 |
| Laptop | 409 | 109 |
| Monitor | 119 | 37 |
| Clock | 119 | 78 |

The object-parameter audit includes neighboring-room objects: 119 clocks covered 83 analog and 36 digital displays with times from 00:04:56 through 23:56:08. All eight screen-content programs appeared among 1,043 devices. Laptop hinges spanned 1.480–2.280 radians; display aspect ratios spanned 1.282–2.650; cup/mug mouth-to-base radius ratios spanned 0.720–1.278. These are sampled ranges, not guarantees of uniform image-space or learning-task coverage. Microphones are deliberately confined to tables and occur less often than personal clutter.

[Distribution plots](evidence/tabletop_v15/distributions.png), [machine-readable measurements and artifact hashes](evidence/tabletop_v15/summary.json), and [layout audit](evidence/tabletop_v15/layout_audit.json) are retained. Full count histograms, object CSVs, placement heatmaps and camera metrics are in `out/tabletop_v15/rooms_final/`.

## Verification and limits

- 142 Rust library tests passed, three ignored. The final small sleeve/microphone/label adjustments also passed the three tabletop regression tests.
- Support/mesh/semantic tests exercise 48 seeds for each of eight tabletop types; placement tests cover 64 rooms. Screen replay/content diversity and fractional clock hands have analytic checks.
- Workspace/all-target strict Clippy, formatting and `wasm32-unknown-unknown` compilation with `web,human_motion` passed. Burn CubeCL retains its upstream future-Rust compatibility notice. This turn did not repeat a browser runtime or ARDY inference test.
- The 28-object studio render exported all 28 expected object boxes and semantic buffers. Eight rooms produced sixteen aligned RGB/depth/normal/position/semantic views at 640×480. The largest per-view depth/position p99 error was 2.87 µm; reprojection p99 stayed below 0.00235 pixels. Each view contained at least seven semantic classes.
- New props use the existing semantic vocabulary (`paper` or `other_prop`). Manifests record distinct `ObjectKind` values; bounding boxes retain instance IDs.

These images still look computer-generated in close-up. Small text is schematic, bottle optics use the renderer's screen-space approximation, and indirect light remains approximate. This check does **not** establish photographic realism, unlimited-process memory stability, or 10M-sample training utility. Timings in the report were collected alongside CPU checks and are not isolated throughput benchmarks.

The grammar is version 15 and the capture identity is `capture-v25`; older resumptions must not silently mix these materials and meshes.

```sh
cargo build --example review_furniture --bin indoor_validate
target/debug/examples/review_furniture out/tabletop_review/gallery tabletop
target/debug/indoor_validate --seed 0 --audit-seeds 128 --renders 8 \
  --cameras 2 --width 640 --height 480 --labels --rotation-augmentation \
  --output out/tabletop_review/rooms
python scripts/report_indoor_tabletop.py \
  --captures out/tabletop_review/rooms --gallery out/tabletop_review/gallery \
  --output out/tabletop_review/report
```
