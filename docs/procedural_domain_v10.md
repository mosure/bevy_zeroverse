# Generator 10: furnishing, visibility and embedding evidence

> **Archived evaluation.** This report describes its recorded build. Use the
> [documentation index](README.md) for current capabilities and defaults.

Generator 10 increases usable furniture density and rotation variance, conditions
cameras on visible content, and fixes procedural tangent/material defects. An
optional Burn/WGPU SigLIP2 audit measures redundancy in captured image space.
The changes are local. Existing scene modes, static AnnyBody people and opt-in
motion generation remain available.

The final qualification contains **12,048 CPU-audited rooms and 144 rendered
rooms / 576 views**. Open the [capture dashboard](evidence/domain10/consecutive/capture_dashboard.svg),
[placement/trajectory heatmaps](evidence/domain10/consecutive/capture_placement.svg),
[embedding spacing plot](evidence/domain10/embeddings/spacing.svg), or
[evidence index](evidence/domain10/README.md) for contact sheets, CSVs and provenance.

## Final captured cohorts

The consecutive cohort uses the same seeds **100000–100127** and capture settings
as generator 9: density **0.65**, human density **0.25**, two cameras at two
trajectory endpoints, **480×360**, native Auto quality and GI at **64 rays**.
The CPU audit covers **100000–109999**. A separate stress cohort selects **16
strata** from **200000–202047**, with densities **0.95 / 0.55** and **640×480**.
No selected seeds or views were discarded or replaced. Stress results are not
population estimates. People are static; this cohort does not evaluate ARDY motion.

| Consecutive cohort quantity | Generator 9 | Generator 10 |
| --- | ---: | ---: |
| Rooms / views | 128 / 512 | 128 / 512 |
| Rooms with a view containing at most two classes | 6 | **0** |
| Rooms with a view dominated by one class (>90%) | 4 | **0** |
| Rooms with people but no person pixels in any view | 23 / 79 | **0 / 111** |
| Main-room chairs, min / median / max | 0 / 2 / 9 | **2 / 8 / 26** |
| Median chair deviation from a cardinal axis | 10.21° | **20.43°** |
| Median table deviation from a cardinal axis | 2.66° | **10.12°** |
| Median desk deviation from a cardinal axis | 8.80° | **15.08°** |
| Median coffee-table deviation from a cardinal axis | 5.26° | **15.14°** |

These compare complete generator snapshots, not an isolated camera ablation.
Furnishing and people counts also changed. Visibility means at least one person
pixel in any view of a room with main-room people; it does not measure recall of
every individual. The new stress cohort has **0 / 16** hidden-people rooms and
zero low-class/dominance flags; chair counts span **6–30**, median **17**.

The consecutive cohort has **4–18** classes per view, median **12**. Vertical FOV
spans **28.26–106.87°**, camera height **0.719–3.310 m**, and trajectory endpoint
baseline **0.038–2.887 m**. The exposure tail remains intentionally broad: the
darkest view has mean linear luminance **0.00127** and **83.4%** dark pixels.
No view reaches the existing 95%-dark or 10%-clipped review thresholds; those
thresholds do not establish good exposure or photographic quality.

All 12,048 final CPU manifests/placements/swept paths pass. Main-room chair counts
in the 10,000-room audit span **1–30**, compared with **0–23** previously. All 576
final captures pass signal, palette, pose and geometric annotation checks. Worst
per-view p99 depth/position disagreement is **4.77e-6 m**, reprojection disagreement
**0.00276 px**, and maximum normal-length error **2.39e-7**. Geometry annotations
are native float32. These are not lighting-accuracy measurements.

Zero failures in 128 scenes still permits a **2.31%** one-sided 95% upper rate,
assuming independent seed draws. Zero hidden-people outcomes among 111 eligible
rooms permits **2.66%**. Correlated views are not independent trials. This sample
cannot certify rare failures in 10M rooms.

## Placement and visibility

Oriented rectangle tests now drive furniture placement and peer-overlap validation.
This permits valid rotated furniture arrangements that inflated axis-aligned
bounds previously rejected. Room boundaries, portals and swept camera clearance
remain conservative. Workstation placement is transactional: a desk that cannot
accommodate its chair is removed. Functional zones receive additional compact
workstations/seating before service objects, with a bounded area/density target
and 128 attempts per zone. A target is not a guarantee that every zone fills.

Zone orientation has a wider continuous range and chair rotation has a wider
central range and tail. Placement constraints still condition the accepted
distribution. Captured object angle CSVs, circular histograms and distance from
the nearest cardinal orientation quantify the accepted result, rather than just
the requested parameter range.

Camera proposals use a cheap 7×5 first-surface proxy at trajectory start, middle
and end. Four classes must cover at least two rays each, with no class covering
more than 80% of the rays. The first camera in a room containing people also
requires person coverage and an unobstructed anatomical point. Upper-head points
handle seated people whose head centers lie inside conservative chair envelopes.
After 256 unsuccessful proposals in a preferred height band, human-focused
cameras search the full safe height range. Swept-path checks remain in force.

Glazing is opaque for geometry/semantic annotations, so a person visible only
through glass does not satisfy this policy. The proxy is not an exact render or
a guarantee over every pixel/timestep; rendered semantic masks remain the final
measurement. First-camera conditioning intentionally biases views toward people.
It does not establish an unbiased camera population or visibility of every person.

## Materials and finite rendering

The procedural mesh builder now repairs degenerate tangent frames after MikkTSpace
generation. A seeded hair regression previously produced zero-length tip tangents.
Hair's anisotropic lobe is also disabled: Bevy 0.19.1 initializes its anisotropic
basis outside the normal-prepass path, which can leave that basis invalid here.
Directional fibre geometry and texture remain; the aggregate lobe is isotropic.
The regression scene now passes exposed-linear-RGB finiteness checks. The old and
new generated scenes differ, so this is not a material-only paired ablation.

Finish layers no longer put mineral veins on timber or textile stripes on stone.
Floor repetitions use metric plank/tile dimensions, and filtered joints preserve
their coverage instead of turning millimeter gaps into full-contrast texel-wide
lines. Procedural people, hair, garments and material detail remain visibly
synthetic in some views; these changes do not establish photographic realism.

## Physical lighting reference

Seed 201110 was exported with its actual meshes, material maps, lights and cameras
and rendered in Blender 5.2.2 LTS / Cycles / OptiX. Two independent runs used
512 samples per pixel, 12 bounces and no denoising. The existing passing radiometry
controls for that Blender version/backend were reused explicitly; they were not
rerun. Native linear RGB uses the same physical exposure without fitted scaling.

Both geometry/camera alignment checks passed. In the two views, native-versus-Cycles
RGB relative MAE after averaging **32×32-pixel blocks** is **31.15% / 53.82%**.
Independent Cycles repeats differ by **0.88% / 1.23%** at that scale; both meet the
existing coarse convergence gate. Raw per-pixel references are still noisy and
are not claimed converged. Native mean luminance is **80.28% / 69.76%** of the
reference. This is a negative physical-accuracy result, not a photorealism pass.

The existing export also has approximation limits: it omits glass absorption,
uses Blender's Principled BSDF, and represents emissive fixtures/sky differently
from native raster light proxies/probes. The reported discrepancy combines those
mapping differences with native transport differences; it is not an isolated
measurement of native GI error or a spectrally accurate ground truth.

The diagnostic uses the same new geometry/material implementation but precedes
the final anatomical camera-visibility refinement. Its exported camera poses are
pinned in the reference hashes. It is one dense scene, not a population estimate.
See [comparison measurements](evidence/domain10/cycles/comparison.json) and
[shared-exposure previews](evidence/domain10/cycles/view_00_native_cycles.png).

## SigLIP2 protocol

See the [embedding audit guide](embedding_audit.md) for model caching, verified
shards, bounded batching and reproduction. The runtime control uses the existing
`burn_loom` PyTorch F16-weight reference image, included twice among eight inputs.
Base-model inference matched its normalized reference within **4.36e-7** maximum
absolute error; batch-8 and batch-1 results were identical. Duplicate cosine
distance was **5.96e-8**. These are image-encoder controls, not downstream task
validation. Only the Base checkpoint was run in this qualification.

Spacing comparisons exclude all views of the same room from nearest-neighbor
search. Camera trajectories and same-room cameras are reported separately. Scene
centroids have equal scene weight. A matched sample-size comparison is required;
global embedding distances can miss local material and geometry defects. No
minimum spacing threshold certifies useful 10M-sample pretraining.

The matched audit embeds **512 images from 128 rooms per generator**, using the
same Base checkpoint and preprocessing. Results are mixed:

| SigLIP2 statistic | Generator 9 | Generator 10 |
| --- | ---: | ---: |
| Median nearest image distance to another scene | 0.06564 | **0.05945** |
| Minimum nearest image distance to another scene | 0.02856 | **0.03675** |
| Fraction of images with nearest distance <0.05 | 9.96% | **16.60%** |
| Median nearest scene-centroid distance | 0.04369 | **0.03793** |
| Scene-centroid effective rank | 46.22 | **47.33** |
| Dimensions explaining 90% of scene-centroid variance | 52 | **53** |
| Median same-camera trajectory distance | 0.02309 | **0.02176** |

Typical cross-scene spacing decreased despite better furnishing/visibility.
The closest pair became less similar, while typical nearest neighbors became
more similar. This does **not** demonstrate a broad semantic-diversity gain.
Viewpoint conditioning and denser office contents are plausible contributors,
but this audit does not isolate causes. The reviewed nearest-pair sheets show
distinct rooms with recurring office compositions, not exact duplicate images.
No cross-scene cosine distance was below 0.02 in either cohort. That threshold
is descriptive, not a quality gate. Wider architectural/subject variation and
real-data downstream evaluation remain open work.

The 1,024-image GPU pass took **11.12 s including preprocessing/readback/IO**,
after **4.89 s** loading/verifying cached weights. It used 64 batches and 64
embedding readbacks. The first control run downloaded/loaded the 14 Base shards
in 19.49 s; all shard checksums passed. These timings are local observations,
not sustained throughput or GPU-utilization qualification.

## Validation and provenance

**95 Rust library tests passed, 3 explicitly ignored; 49 Python tests passed.**
Native build, strict Clippy (default targets and the optional embedding CLI),
root-package formatting and the WebGPU viewer compile check passed. The optional
Burn dependency graph emits upstream Rust future-incompatibility notices for
`burn-cubecl` / `burn-cubecl-fusion`; these are not local compiler diagnostics.
No new browser runtime qualification or unlimited-process memory test is claimed.

Consecutive four-view capture time was **0.61–2.60 s**, median **1.07 s**, versus
0.86 s previously. The denser stress cohort median was **2.31 s**. This includes
generation, warmup and disk IO, with some CPU build work overlapping; it is not
an isolated performance comparison. More furniture increases rendering work.
SigLIP2 is absent from the default dependency graph and adds no generation-time
model loading or inference.

An earlier complete diagnostic cohort left one glass-occluded person; a subsequent
CPU audit exposed four high-back-chair visibility failures. Both motivated the
anatomical-point/height search refinements. Final evidence comes only from the
fully rerun `out/domain10_verified` and `out/domain10_verified_stress` cohorts,
with no skipped seeds. Their immutable measurements and source/binary hashes,
plus the separately scoped Cycles and embedding controls, are retained in
[provenance](evidence/domain10/provenance.json).

Generator/capture identities are **10 / capture-v14**; metric schema remains 6.
The same seed changes across generator versions. Photographic realism, sufficient
semantic novelty for 10M samples and downstream learning utility remain unproven.
