# Kitchen construction and placement regularity

This review targets active constraints that survived the broader furniture,
material, floor and ceiling expansions. Fitted cabinetry should retain level
worktops and useful service relationships. Exact ordering, identical appliance
ratios and preferred first placement coordinates need not be universal.

| Bottleneck | Sampling and construction |
|---|---|
| Sink-first service runs, exact 25 mm gaps and center-first shifts | Required units are permuted before transactional admission. Each run samples a 12–45 mm installation gap and shuffled continuous strata across its available span. Optional units try both ends in seeded order. |
| Repeated island coordinates and dining crosses | Islands sample position, a small orientation offset and bounded radial alternatives. Dining tables sample full yaw and metric height; chairs have bounded offset, heading and occupancy variation. Existing floor, aisle, door and object-clearance gates remain authoritative. |
| Narrow appliance dimensions | Category-specific metric ranges are sampled before calculating door travel and admitting the complete envelope. Counter-height appliances still share the recorded worktop plane. |
| Identical fridge leaves and shelf intervals | Paired doors sample unequal widths; their real maximum width drives the opening reserve. Adjustable shelf heights use unequal positive intervals, with contents fitted to each actual bay's headroom. |
| Four identical-position burner circles | A bounded physical layout samples 2–5 burners, independent radii, placement and an actual control strip. Gas grates and electric rings use the same admitted footprint. Knob count follows burner count. |
| Fixed microwave control/door proportions and control pointers | Continuous control-strip proportions determine both the door geometry and its opening reserve. Control spans, radii and pointer angles vary within their fascia. |
| Repeated diagonal fridge-note rows | Paper/photo formats, position, yaw, content and mounts vary on each operated door. Rotated footprints clear the raised field or viewport, hardware and other notes. Notes use the actual skin plane; glass uses adhesive tabs. |
| Every appliance inherits cabinet appearance | Cabinetry retains a coherent collection finish. Appliances sample collection coherence separately, with independent shell/front substrate choices. Hot appliances use metal finishes; wood panel skins are restricted to refrigerators and dishwashers. |

The kitchen placement, appliance structure, burner field and note composition
have separate seed domains. Finish changes do not change dimensions or geometry.
The existing material batches and bounded finish/content atlases are reused.
All parts retain their semantic labels and operated-door ownership. Archived
kitchen assembly diagnostics without the optional sampling record remain valid.

`kitchens=7` identifies the revised construction contract. Exact replay requires
the generator and complete capture identity, recorded with the study.

## Qualification

Routine tests check finite/bounded meshes, clear openings, independent finish
sampling, burner/control separation, shelf headroom, rotated note containment,
actual accepted unit order and continuous placement distributions. Complete-room
checks include supported objects, multi-view camera paths and temporal appliance
construction.

The ignored test `kitchen_regularity_population_qualification` writes the local
`out/kitchen_regularity/report.json` and an actual-mesh gallery. It examines six
appliance categories at fixed chassis dimensions, closed doors and finish
identity, using six
orthographic 24 × 24 first-surface depth maps. Ordered greedy coverage at depth
differences 0.005, 0.015 and 0.03 measures shape differences independently of
material, color, mesh indexing and triangle count. It also records accepted
kitchen/kitchenette groups and CPU manifest/mesh construction times.

### Measured local population

The local population receipt records generator 90 / capture-v115: seeds 0–63
for each of Kitchen and Kitchenette, density 0.65, human density 0.15 and no
cameras in the timing sample. All 128 manifests pass layout validation; the
first 16 seeds per profile also pass complete-room geometry validation. A
separate routine test covers 48 rooms across Kitchen, Kitchenette, Cafe and
Mixed with three cameras and people, including 16 complete-room mesh checks.

| Accepted placement measurement | Result |
|---|---|
| First unit in the run | 68 sinks, 45 counters, 15 ranges |
| Accepted configurations | 81 straight, 15 island, 19 perpendicular, 13 parallel |
| Installation gap, median / 5th–95th percentile | 27.8 mm / 13.4–43.9 mm |
| Run offset, median / 5th–95th percentile | 0.337 m / −2.346–2.769 m |
| Sampled appliance collection coherence, median / 5th–95th percentile | 0.585 / 0.312–0.808 |
| CPU room manifest construction, median / 95th percentile | 1.83 ms / 5.25 ms |

The object cohort checks 64 meshes in each of six appliance categories (384
total), with chassis dimensions and finish held fixed and doors closed. Cooktop
burner counts are 17 two-burner, 15 three-burner, 28 four-burner and four
five-burner layouts. Open-door travel, semantic ownership and motion planning
are checked separately against the actual reserved object envelopes.

| Category | Shape representatives at 0.005 / 0.015 / 0.03 | CPU construction median / 95th percentile |
|---|---:|---:|
| Cooktop | 64 / 64 / 64 | 0.094 / 0.299 ms |
| Range | 8 / 2 / 1 | 0.175 / 0.421 ms |
| Microwave | 17 / 10 / 3 | 0.067 / 0.076 ms |
| Refrigerator | 7 / 2 / 1 | 0.460 / 0.642 ms |
| Oven | 10 / 2 / 1 | 0.145 / 0.163 ms |
| Dishwasher | 9 / 3 / 2 | 0.165 / 0.190 ms |

Shape distance is union-pixel mean absolute depth difference, normalized by the
fixed envelope, with missing silhouette support counted as unit difference.
These are threshold-dependent coverage counts, not independent-sample counts.
Manufactured closed enclosures still share a rectangular silhouette at coarse
resolution. Metric size, appearance, opened interiors and placement contribute
variation outside this controlled measurement; the low coarse shape coverage
remains a useful limitation to track.

The local `out/kitchen_regularity/index.html` gallery and `report.json` retain
the exact sampled construction parameters. `qualification.json` records source
hashes and the completed native, motion, camera and portability checks. CPU
timings are from a shared machine and exclude GPU uploads, rendering, readback,
and concurrent training; object timings include review fixture setup.

The gallery is CPU flat shading. These bounded diagnostics do not establish
photographic realism, GPU throughput, embedding-space volume or reduced model
overfitting. The local study is not included automatically in published pages.
