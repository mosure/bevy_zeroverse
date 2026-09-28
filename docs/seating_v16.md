# Seating, activity and viewer review, v16

Indoor generation now includes modular sofas, loveseats, left/right chaise returns and lounge armchairs. Seat width, cushion thickness, arm width, back tilt, rounding, leg height and exposed frame construction vary continuously. Upholstery samples fabric, alternate fabric or leather; cushions, arms, backs, legs and pillows have separate geometry/material roles. A chaise keeps a conservative rectangular placement envelope, so its open corner is not reused for another object or a camera path.

Chair generation covers task/conference chairs, cantilever visitors, wood and plastic shells, backless and low-back stools, lounge armchairs and executive chairs. Shell curvature, taper, lumbar support, shoulder flare, recline, seat contour, armrests, base proportions and headrests vary within applicable families. Wood spindles stay with wood chairs; executive upholstery has a continuous back shell. Dining and lounge chairs use shorter height ranges than high-back task/executive chairs. Stools retain the existing 0.47 m seated-human datum; counter-height stools are not claimed here.

Bookcases vary shelf spacing, bay count, frame material, backing, fill, upright/stacked books, book dimensions and individual cover finishes. Whiteboards have filtered marker textures with process graphs, curves, bars, agendas, planning grids, mind maps, numerical sketches and erased remnants. TVs have presentations, sports diagrams, landscapes, video-call tiles, signage and off states. These are asset-free schematic graphics, not photographs or readable prose. Electrical fixtures add one-to-three-gang plates, socket conventions, rocker/toggle/rotary switches, mounting screws and separate finishes. A scene shares its socket convention. Their placement rejects openings, niches, columns and existing fixtures; roots retain instance boxes and the existing `other_prop` semantic class.

Activity choices now include **Conference, OpenOffice, Lounge, Training, Coworking, Breakroom, Reception, Library, Workshop and Studio**, plus **Mixed**. These are biases for a stored per-zone probability mixture of workstation, meeting, social and learning furniture groups. Group scale, storage bias, spatial field, room dimensions and partitions remain independently sampled. Groups resample the mixture within a zone; selecting a profile does not select a fixed floor plan. The mixture is serialized in the manifest and checked on replay.

The inspector exposes all playback modes, including the viewer default **Sin**. Sin advances its own phase from the shared playback resource, so pausing or changing speed does not jump to application elapsed time. Camera trajectories and generated human actors consume the same mapped progress. Sin reverses motion on its return half-cycle; use Once or Loop when backward human action is unwanted. This work did not generate new ARDY clips.

Camera-grid mode disables the editor's 3D camera and gives the grid an opaque background. One explicitly managed UI/egui camera keeps the inspector usable; transparent final-output blending preserves the HDR editor view outside grid mode. Turning the grid off restores the editor camera and its orbit state. Changing generation properties still requires **Regenerate**; sliders do not automatically rebuild the scene.

## Evidence

[Sofas](evidence/seating_v16/sofas.png), [chairs](evidence/seating_v16/chairs.png), [bookshelves](evidence/seating_v16/bookshelves.png), [boards/TVs](evidence/seating_v16/boards.png), [electrical hardware](evidence/seating_v16/hardware.png), and [all room views](evidence/seating_v16/rooms.png) are retained. The 31-object studio gallery deliberately exercises programs. The eight room captures use **consecutive seeds 0–7 without filtering**, two views each at 640×480, with whole-scene rotation augmentation.

The CPU audit covers seeds 0–127. All 128 layouts passed; all ten activity biases and chair families appeared. The 370 generated zones had continuous social-group probabilities from 0.035 to 0.782, meeting probabilities from 0.039 to 0.679, group scales from 0.701 to 1.343, and storage priors from 0.082 to 0.944. All eight whiteboard and TV content programs appeared. Both chaise handednesses appeared, but only six of the 119 sofas had a return: those need deliberate weighting or stratification when balanced sectional coverage is required.

| Primary-room object | Instances | Rooms containing it / 128 |
| --- | ---: | ---: |
| Sofa | 119 | 81 |
| Chair | 1,198 | 128 |
| Bookcase | 287 | 113 |
| Whiteboard | 175 | 98 |
| TV/display | 67 | 48 |
| Wall outlet | 503 | 128 |
| Light switch | 232 | 126 |

Counts retain zero-count rooms. Parameter distributions include neighboring-room objects; upholstery-program statistics also include lounge armchairs. [Plots](evidence/seating_v16/distributions.png), [measurements and artifact hashes](evidence/seating_v16/summary.json), and [layout audit](evidence/seating_v16/layout_audit.json) are retained. Full object/camera CSVs, count histograms and placement heatmaps are in `out/seating_v16/rooms/`.

## Verification and limits

- 148 library tests passed, three ignored; workspace/all-target Clippy with `-D warnings`, formatting and the `web,human_motion` Wasm build passed. Burn CubeCL still emits its upstream future-Rust compatibility notice.
- Geometry/semantic/envelope tests cover 32 seeds across 15 seating/storage/hardware cases. Activity tests validate all ten profiles, density extremes, stored continuous mixtures and manifest round trips. Board and television tests check deterministic nonrepeating content.
- Actual mesh validation covered the first sixteen rooms. The summary records mesh/triangle counts, including the small number of degenerate triangles accepted by the existing mesh gate; it does not claim every emitted triangle is nondegenerate.
- The studio export checked all 31 expected instance boxes and nonempty semantic buffers. The sixteen room views include RGB/depth/normal/position/semantic passes with float32 geometry annotations. Largest per-view depth/position p99 error was 1.91 µm and reprojection p99 was below 0.0024 pixels.
- Playback tests check a full Sin cycle, pause/resume and speed changes. A viewer regression checks editor-camera deactivation/restoration and UI preservation. The focused viewer regression also passed after the final compositing fix. Headed Chrome/WebGPU passed both initial editor/grid modes and both transitions, with no runtime or uncaptured GPU errors and no scene regeneration from those controls. Sin is visible and selectable in the retained dropdown screenshots. Browser checks and screenshots are in [the controls report](evidence/seating_v16/browser/report.json).

The renders still show simplified upholstery, people, books and screen content. Some consecutive views have awkward framing or face sparsely furnished regions; the contact sheet keeps those failures visible. These bounded checks establish replay, layout and annotation behavior, **not photographic realism, unlimited-process memory stability or 10M-sample training utility**. Capture timings include startup and concurrent checks and are not throughput benchmarks.

Generator version is **16** and capture identity is **capture-v26**; older resumptions must not silently mix these programs.

```sh
cargo build --features human_motion --example review_furniture --bin indoor_validate
target/debug/examples/review_furniture out/seating_v16/gallery seating
target/debug/indoor_validate --seed 0 --audit-seeds 128 --renders 8 \
  --cameras 2 --width 640 --height 480 --labels --rotation-augmentation \
  --output out/seating_v16/rooms
python scripts/report_indoor_seating.py --captures out/seating_v16/rooms \
  --gallery out/seating_v16/gallery --output out/seating_v16/report
```
