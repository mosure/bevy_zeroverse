# Physical reference renders

The optional Cycles bridge exports the **actual generated meshes, PBR maps,
transforms, lights and sampled cameras**. It uses no downloaded room meshes,
reference photographs or generative image postprocessing. Native annotations
remain aligned to the generated geometry, including the first glass surface.

```sh
target/debug/indoor_validate --audit-seeds 1024 --renders 12 --stratified \
  --cameras 2 --width 640 --height 480 --labels --linear-rgb \
  --export-reference --output out/physical_native

blender -b --threads 8 --factory-startup --python-exit-code 1 \
  --python scripts/validate_cycles_radiometry.py -- --device OPTIX --output out/radiometry

blender -b --factory-startup --python-exit-code 1 \
  --python scripts/render_indoor_cycles.py -- \
  --scene out/physical_native/seed_000000/reference/scene.json \
  --samples 4096 --device OPTIX --radiometry-report out/radiometry/report.json \
  --output out/physical_cycles_seed0

python scripts/compare_indoor_cycles.py \
  --native out/physical_native/seed_000000 \
  --cycles out/physical_cycles_seed0 --output out/physical_comparison_seed0
```

Use a new output directory for each run. Comparison needs NumPy and Pillow;
rendering uses Blender's bundled Python. `--python-exit-code 1` matters: Blender
otherwise can return success after a Python exception. GPU fallback is explicit;
an unavailable OptiX/CUDA device fails instead of silently switching to CPU.

## Measurement contract

* Geometry/camera alignment is independently checked with Blender BVH rays
  against decoded native float32 world-position annotations. Hit agreement and
  a 1 mm p99 position-error gate precede image comparisons.
* Native `--linear-rgb` disables tone mapping, bloom, FXAA and dithering. Native
  RGB still passes through an RGBA16F render intermediate; geometry annotations
  use the separate RGBA32F path.
* Cycles writes unexposed, scene-linear 32-bit EXR and top-left-origin RGBA32F.
  Comparison applies only the recorded Bevy exposure, `2^-EV100 / 1.2`. No fitted
  exposure, image registration or aesthetic filtering is permitted.
* Reports include full-image radiance error, luminance bias, error in stops and
  separate wall/floor/ceiling/chair/window/lamp/person/desk measurements. The
  source mesh, texture, camera and raw-image hashes are checked and retained.
* Quantitative comparisons use raw samples. `--denoise` is available for an
  explicitly recorded display/dataset render; denoising is not a convergence
  test. Increase samples and compare independent runs before treating narrow
  sunlight caustics or glossy reflections as converged ground truth.

Use `--seed-offset 104729` for an independent reference render, then pass its
directory to `compare_indoor_cycles.py --reference-repeat`. The comparison
checks scene hashes, optical calibration, bounce depth, Blender build and
independent sampling seeds. It reports 8-, 16- and 32-pixel block averages in
addition to individual pixels. A 32-pixel block comparison below 5% RGB relative
MAE and whole-image mean luminance within 2% is labelled **coarse convergence**;
it does not certify individual pixels. Raw pixel error remains noise-sensitive.
Partial right/bottom blocks retain every pixel and are weighted by their actual
pixel area. Image dimensions need not be multiples of 32. This also preserves
whole-image radiance means in block comparisons without dropping edge content.
For native transport ablations, `indoor_validate --gi-bounces 6 --gi-rays 512`
records the additional transport budget explicitly in its selection manifest.

An existing Cycles render can be reused across native engine revisions with
`--reference-source <original-native-directory>/reference`. Original file hashes
must still match the Cycles report. The comparator additionally resolves mesh,
material and texture indices and requires identical optical content, including
vertex/normal/UV/index bytes, material maps, transforms, lights, exposure and
cameras. The manifest supplying the external sky must also be hashed. Only table
ordering and the native engine label may differ. This avoids
rerendering an unchanged physical scene when testing raster-renderer changes.

References containing people currently require one trajectory time per export:
the bridge stores one geometry snapshot. It rejects animated multi-time exports
instead of rendering all camera times against the final human pose. Native
multi-time RGB/annotation capture remains supported and tested.

The calibration checks emission, a Lambertian receiver under a measured sun,
the upper-hemisphere sky, glass transmission to the camera, and sunlight through
an 8 mm solid glass pane onto a diffuse receiver. The last control needs many
samples and includes the analytic interreflection between the pane and floor.
It reports sampling uncertainty separately and rejects relative bias
over 2.5% or a relative mean standard error over 2%.

Approximate MNEE shadow caustics are disabled in the reference bridge after
failing the planar-glass illumination control. The calibration's
`--approximate-caustics --glass-samples 512` option preserves that diagnostic.
Cycles itself documents that MNEE is approximate and can produce incorrect
brightness; it is not automatically a physical oracle merely because it is
enabled. See the [Cycles caustics limitations](https://docs.blender.org/manual/en/4.5/render/cycles/object_settings/object_data.html#caustics).

## Model differences

Cycles integrates actual emissive luminaire surfaces, full diffuse/specular and
refraction paths, and the external hemispherical sky. Native rendering uses
spot/point proxies, finite-bounce diffuse probes, a preconvolved indoor reflection
cubemap, shadow maps and optional screen-space effects. The solar disk is finite
in Cycles. Principled and Bevy PBR lobes also differ. Glass volume absorption is
currently recorded by the exporter but not mapped into Cycles.

These differences belong in the error report. A geometric match, a visually
appealing preview, a low-noise render, or an irradiance-probe comparison alone
does not establish photographic realism or physical agreement of final images.

## Large-population metrics

The metric exporter uses exact external sorting after 32,768 values per numeric
metric. Quantiles and histogram counts remain exact while RAM is bounded by
the number of metrics, not the number of scenes. Scratch runs are removed on
completion or error. CSV records and spatial heatmaps are written incrementally.
The 100,000-scene audit in `out/indoor_v5_metrics100k` used 52,052 KiB peak RSS and
found no invalid layouts; this audits procedural constraints, not image realism.
