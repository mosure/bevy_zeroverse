# Exterior architecture and windows, v17

Exterior windows previously occupied only the negative-X wall, with the same sill/head heights in every bay. New manifests store an `exterior` program shared by architecture construction, wall-attachment rejection, validation and metrics. The primary room can expose its left, right and rear sides independently. All seven nonempty combinations occur, including adjacent corner exposure, opposite walls and three exposed sides. The fourth wall retains the internal glazed partition and furnished neighboring room.

Each exposed wall independently samples a continuous aperture program: bay pitch and unequal widths, pier spacing, sill/head heights, optional separated vertical bands, mullion columns, transom height, frame width/depth, recess and sill projection. The height distribution favors near-floor-to-ceiling glazing while retaining ordinary punched windows, high ribbons and clerestories. Frames use metal, wood or plastic; panes, inset reveals, sills and handles are separate geometry. Optional roller or tilted Venetian shades have continuous deployment. Openings are rectangular; curved walls, arched windows, projecting bay windows and an arbitrary building envelope are not implemented here.

Wall meshes are constructed from the complement of their apertures. Feature panels, timber slats, wainscot and skirting are clipped against the same openings. Rear-wall niches are built only on opaque rear walls. Displays, boards, art, clocks, outlets and switches sample all three walls and require solid backing. Trim joints use separated faces; the coplanar-face test caught and removed a duplicate sill/outer-wall plane. Exterior ground and neighboring buildings extend to exposed sides. Geometry remains batched by material and semantic label; this adds no model downloads, textures or per-window lights.

The full program is serialized in the manifest. Scene-local primary-room camera and O-voxel/AABB bounds are retained, with exterior context outside the reconstruction region. Missing `exterior` fields deserialize to the legacy shell path. Generator **17**, capture identity **capture-v27** and metrics schema **8** distinguish the new outputs; older captures must not be resumed into the same dataset.

## Measured coverage

The consecutive seeds **0–127**, density 0.65, human density 0.25 and two 640×480 cameras per room passed the layout audit. The population contains **894 exterior rough openings**:

| Exposure | Rooms / 128 |
| --- | ---: |
| One exterior window wall | 47 |
| Two exterior window walls | 59 |
| Three exterior window walls | 22 |
| At least one near-full-height opening | 44 |

The last row overlaps the first three. “Near-full-height” means rough sill ≤0.16 m and head clearance ≤0.18 m; the frame reduces the clear glass extent. Widths span **0.70–8.14 m**, heights **0.52–4.56 m**, and total rough aperture area spans **21.6–90.7%** of an exposed wall. These are sampled extrema, not guarantees for every finite run. Window programs use independent deterministic RNG streams and can be inspected in stored manifests.

[Distributions](evidence/facade_v17/distributions.png), [opening CSV](evidence/facade_v17/windows.csv), [audit/annotation metrics and artifact hashes](evidence/facade_v17/summary.json), [layout audit](evidence/facade_v17/layout_audit.json), and [schematic elevations](evidence/facade_v17/elevations.png) are retained. Numeric metrics distinguish per-opening, per-facade and per-room populations. `exterior_opening_area_fraction` measures the rough wall aperture, including its frame; it is not a net glass-area ratio.

## Render and regression checks

[All 16 native RGB views](evidence/facade_v17/rooms.png) use consecutive seeds **0–7**, two sampled cameras each, with whole-scene rotation augmentation. They are unfiltered. Depth, normal, position and semantic passes were captured alongside color; the largest per-view depth/position p99 error was **1.91 µm**, and reprojection p99 was **0.00259 pixels**. Windows retain the existing window semantic class and annotation-opaque depth convention. The first sixteen rooms also passed actual mesh validation.

- **152 library tests passed; three ignored.** New tests cover 512 seeds for exposure combinations, near-full-height prevalence and replay; 128 seeds for exact aperture-complement area; 32 scenes for glass visibility through actual wall/finish meshes; malformed programs and missing-field compatibility. Existing trim tests check competing coplanar faces, and layout checks reject window-backed wall fixtures.
- Workspace/all-target Clippy with `-D warnings`, formatting and the `web,human_motion` Wasm build passed. Burn CubeCL still emits its upstream future-Rust compatibility notice.
- Headed Chrome/WebGPU passed occupied **Auto seed 44** and **Portable seed 4** with paired RGB/semantic views, pipeline settling and no runtime/uncaptured GPU errors. Both semantic views covered 100% of the scene test region. The RGB wall/floor/ceiling mask had zero near-background pixels. [Screenshots and logs](evidence/facade_v17/browser/report.json) are retained. Egui reports its optional texture-binding-array fallback, and human skinning reports its existing 8-to-4 influence clamp.

The browser check exposed a separate upload regression: the viewer's 32 MiB/frame asset-upload cap left complete material groups missing indefinitely on this Bevy/WebGPU build. Wasm now uses Bevy's default unlimited upload policy, retaining GPU preprocessing and the cooperative CPU preparation path. Native viewer/capture policies are unchanged. This favors complete geometry over limiting a browser upload burst; startup frame-time improvements are not claimed. The stronger smoke script requires annotation-opaque surface coverage and checks RGB wall/floor/ceiling pixels against a matched semantic render. It rejects both retained pre-fix images; [before/after measurements](evidence/facade_v17/browser/upload_regression.json) document that negative control. The near-background threshold applies to these two illuminated fixtures, not arbitrary low-light scenes. A nonblank canvas alone no longer passes.

Some camera views still face sparse regions or fail to show all exposed sides at once. Several high-contrast window views have bright exteriors. The room envelope and outside buildings remain simplified. This is a bounded geometry/distribution and annotation qualification, **not proof of photographic realism or 10M-sample learning utility**. Capture wall times include initialization and concurrent checks; they are not throughput measurements. No new ARDY motion clips were generated in this review.

```sh
cargo test --lib --features human_motion
cargo clippy --workspace --all-targets --features human_motion -- -D warnings
cargo build --features human_motion --bin indoor_validate
target/debug/indoor_validate --seed 0 --audit-seeds 128 --renders 8 \
  --cameras 2 --width 640 --height 480 --labels --rotation-augmentation \
  --output out/facade_v17/rooms
python scripts/report_indoor_facades.py --captures out/facade_v17/rooms \
  --output out/facade_v17/report
cargo build --no-default-features --target wasm32-unknown-unknown \
  --features web,human_motion --bin viewer
# After serving www with those bindings and assets:
python scripts/smoke_indoor_browser.py --url http://127.0.0.1:8766 \
  --output out/facade_v17/browser
```
