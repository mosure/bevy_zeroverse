# Procedural diffuse global illumination

Native `Auto` now computes diffuse indirect illumination from the generated
scene, including its furniture and procedural people. No lightmap, room scan,
HDR photograph, or imported mesh is required. `Portable` and Wasm keep the
documented environment-light fallback.

The baker builds a triangle BVH from the same architecture, object, and human
assemblies used for rendering. Surface colors come from the generated PBR
materials, with a small linear-color texture representation for diffuse
reflectance. Transparent glazing and luminous fixtures follow the renderer's
`NotShadowCaster` policy. Luminaire flux, angular falloff, inverse-square/range
attenuation, sun direction, sun illuminance, and light colors are shared with
the renderer. A hemispherical sky radiance is visible only through unoccluded
paths; it contains no sun disk, which would double-count the analytic sun.

Cosine-weighted diffuse paths and shadow rays estimate reflected radiance.
The result is convolved into six cardinal irradiance lobes and stored in a
filterable RGBA16F volume as irradiance divided by π, in cd/m². Raster PBR then
uses Bevy's existing ambient-cube lookup. Its diffuse-source precedence means
the volume replaces environment diffuse illumination, and camera ambient fill
is zero while GI is active. The reflection environment remains a proxy.

Probe positions remain in room coordinates. Lobe directions use world
coordinates because Bevy's shader evaluates them with world normals, including
under rotation augmentation. Texture packing follows the actual Bevy WGSL
positive/negative-axis ordering rather than the contradictory prose ordering
in its module documentation.

Native Auto dispatches this transport once per scene on the GPU. One 64-thread
workgroup traces each probe's rays and reduces its six irradiance lobes into
the final 3D texture before the camera render graph runs. Capture waits for
the compiled pipeline and dispatch readiness, with GPU queue ordering ensuring
the light field is available. Production rendering never reads the volume back
to the CPU and never waits on a blocking buffer map. Transport buffers and the
probe image retain only the current scene's data.

An explicit CPU fallback (`IndoorGiSettings.gpu=false`) prepares the first scene
before `SceneLoadedEvent`. A bounded background task prepares the next sequential
seed while the current scene renders. Only
an exact match of seed, layout, furnishing density, human density, camera count,
rotation policy, and bake settings can consume that result. At most one
background preparation exists; random indexed requests cannot accumulate
tasks or cached GPU images. A task for the requested scene is joined before
capture proceeds. Statistics distinguish prefetch use and remaining wait.

`IndoorGiSettings` is an application resource for matched ablations and offline
bake budgets. Setting `enabled=false` retains native direct lights and shadows.
`BevyZeroverseConfig.indoor_gi_rays` (viewer/CLI `--indoor-gi-rays`) initializes
the ray budget to 256 by default; accepted values are 64–16,384. The canonical
dataset CLI and Python configuration expose the same setting. Use 1024 for
lower Monte Carlo noise when the additional generation cost is acceptable.
An explicitly supplied `IndoorGiSettings` resource takes precedence, and later
resource overrides remain available to benchmark and validation applications.
`BakeScene::bake` is a CPU-only API returning `ProbeData`; it can run on a
preparation worker without a GPU or ECS world. `BakeStatistics` records triangle
and probe counts, ray and bounce budgets, preparation/bake time, probe texture
size, mean radiance, prefetch use, and wait time. GPU transport does not pretend
to know a CPU bake duration or mean radiance: these fields are null. With Bevy
render diagnostics enabled, `indoor_diffuse_bake` measures the actual compute
pass using GPU timestamps. The native validation test explicitly reads back
the volume and compares it with the separate CPU implementation.

The correctness controls include an analytic hemispherical-sky integral,
a closed diffuse/emissive enclosure checked against an independent geometric
series, BVH intersections checked against brute-force triangle intersections,
and exact ambient-cube texture packing. The ignored CPU reference experiment
compares the production budget with an independent sample sequence using
16,384 rays and eight bounces. That high-sample comparison shares the transport
integrator; it is a convergence diagnostic, not an independent renderer.
The ignored native RGB experiment changes only volume intensity between one
and zero, leaving geometry, direct lights, shadows, and exposure fixed. Playback
is stopped and exact camera-matrix equality is asserted for every comparison.

Actual GPU-volume readback at nine spatial points (six lobes each) was compared
with the separate CPU implementation using 4096 rays and the same three-bounce
transport. For seed 6, the aggregate relative mean absolute error was 17.01% at
256 rays and 8.76% at 1024 rays. This measures transport agreement at selected
points against a finite-sample reference; it does not measure agreement with
a photograph or a full path-traced renderer. The captured GPU volume contained
825 probes in 39,600 bytes and about 13.5 MB of transient transport buffers.
See the [256-ray reference](procedural_indoor/gi_gpu_reference.json) and
[1024-ray reference](procedural_indoor/gi_gpu_reference_1024.json).

Matched native generation measurements used 64 scenes, discarded eight warmup
scenes, and returned three 320×240 cameras with RGB, depth, position, normal,
and semantic planes per scene on an NVIDIA RTX PRO 6000 Blackwell / Vulkan.
Preparation and completed readback are included; file encoding is excluded.

| Rays per probe | Selected-probe relative error | Completed views/s | Median scene seconds |
| --- | ---: | ---: | ---: |
| 256 (default) | 17.01% | 5.741 | 0.514 |
| 1024 | 8.76% | 2.877 | 1.001 |

The [256-ray](procedural_indoor/gi_benchmark_256.json) and
[1024-ray](procedural_indoor/gi_benchmark_1024.json) reports record the adapter,
actual settings, and measurement policy. The default retains the more efficient
budget. These short comparisons do not establish long-run memory stability;
see the separate engine qualification for that evidence.

In the two fixed-camera seed-6 images, disabling only the GI volume produced
mean absolute per-pixel linear-luminance differences of 0.000574 and 0.001219; respectively 1.03% and 2.26% of
pixels changed by more than 0.005. The contribution is measurable but modest
in these views, and the ceiling remains dark. The
[controlled RGB report](procedural_indoor/gi_render_ablation.json),
[GI image](procedural_indoor/gi_native.png), and
[same camera without GI](procedural_indoor/gi_native_no_indirect.png) preserve
that limitation. A separate [fixed-camera lighting control](procedural_indoor/lighting_ablation_v3.json)
confirmed both local and sun shadows affect the image: removing local shadows
produced luminance MAEs of 0.00774 and 0.00658; removing all direct lights
produced 0.04545 and 0.04330. The baked GI field stays fixed in the direct-light
control so that the intervention isolates the raster direct-light contribution.

Lighting placement uses a room-size fixture grid instead of four fixed lamps
for every floor area. Per-fixture flux comes from a bounded lumen-method design
estimate (450 lux daytime, 300 lux evening, utilization factor 0.70). These are
design targets, not a claim that every desk measures that illuminance. Up to
eight fixtures cast shadow maps. Screen and diffuser emission responds to
camera exposure; Bevy's default exposure-independent emission had washed out
display content. Diffuser luminance uses flux divided by emitting area and π.
The `shadow_map_size` capability field describes sun/spot maps (native Auto
2048, WebGPU Auto 1024). Floor-lamp point shadows use Bevy's 1024-pixel cube faces.

```sh
cargo test --lib scene::procedural_indoor::gi::tests -- --nocapture
cargo test --lib export_indoor_gi_reference -- --ignored --nocapture
cargo test --test procedural_indoor_gi -- --ignored --nocapture
INDOOR_GI_TEST_RAYS=1024 INDOOR_GI_TEST_OUTPUT=out/indoor_gi_1024 cargo test --test procedural_indoor_gi -- --ignored --nocapture
cargo test --test procedural_indoor_lighting -- --ignored --nocapture
```

This is static diffuse GI with a bounded bake budget, not a full path-traced
renderer. Glossy interreflection, caustics, refraction transport, fine-scale
visibility during probe interpolation, and arbitrary moving-light updates are
not solved. Probes inside solids are replaced by the nearest valid probe;
ambient-cube interpolation can still leak illumination across thin objects.
The generated sky is an analytic approximation, and the albedo reduction and
Lambertian transport differ from the full raster BRDF. Human geometry is baked
at its generated pose. Regenerate/rebake when geometry or lighting changes.

## Experimental Solari preview

Build the native Studio with `cargo run --features solari -- --preview-lighting solari`.
**View & playback → RGB display & anti-aliasing → RGB preview lighting** switches the editor
and camera grid live. The preview uses Bevy's realtime path tracer with its default bounce
and sampling settings. The **Solari + ReSTIR** option (`--preview-lighting solari-restir`)
adds spatial and temporal sample reuse. Hardware ray queries, bindless arrays and a sufficiently large
storage-buffer binding are required. An unsupported device falls back to raster PBR.
Browser, archive, and Burn workers use the standard raster renderer. Burn resolves
a Studio configuration to raster before computing the WGPU or CUDA capture policy fingerprint.

The release supports indirect lighting, glossy reflections, emissive meshes, a directional
sun and the generated environment. Eligible, visible rigid room batches reuse their mesh
and material handles. Hidden prefetched rooms enter the acceleration structure when they
become visible. Glass and skinned or morphing meshes use forward rendering on primary
rays. Secondary rays omit them and use Bevy’s reduced base, metallic/roughness, normal
and emission material representation for traced hits. The traced pass does not evaluate the generator's
analytic point, spot and rectangular lights. The existing diffuse volume and shadow
maps continue to serve forward surfaces. A single RGB image can therefore combine two
lighting models. Sampling noise is visible without a denoiser. The initial random state
includes the renderer frame number, which makes RGB sensitive to rendering history.

Mesh batching, CPU room lookahead and GPU material synthesis still apply. The hardware
acceleration structure is shared within the current room, while each **active physical
view** gets its own lighting cache. Bevy allocates about 88 MiB per view for the world
cache and light tiles alone. Three pinhole cameras start at about 264 MiB, before
geometry, G-buffers and output images. Each active fisheye source face adds another
cache. ReSTIR adds two 48-byte reservoirs per shaded pixel, about 96 MiB extra
per physical 1024×1024 view, along with previous-frame prepasses. A view exceeding
the device’s storage-buffer binding limit uses the ordinary Solari integrator. SSAA increases shaded pixels and attachment sizes. Spatial AA remains available.
The preview writes the ordinary GPU color target before lens composition and sensor
resolve. It introduces no CPU pixel copy. Dataset annotations and WGPU/CUDA tensor
capture keep their existing pipelines.

Zeroverse pins Solari 0.20.0 to its Bevy 0.20.0 workspace. The published Solari 0.20.1
has the same Rust and shader source under `src/`.
[Bevy release notes](https://bevy.org/news/bevy-0-20/#solari-and-dlss) ·
[Solari renderer design](https://jms55.github.io/posts/2026-09-18-solari-bevy-0-20/) ·
[Per-view allocations](https://github.com/bevyengine/bevy/blob/v0.20.0/crates/bevy_solari/src/realtime/prepare.rs).

### Completed three-view benchmark

On 2026-11-23, generator 213 and capture-v243 produced these RGB streams on an RTX
PRO 6000 Blackwell with ongoing GPU training. All modes used the same rooms, camera
calibration, automatic SSAA (2× or 3×, matched per seed), material baking, full native
quality and CPU prefetch. Timing covers room preparation through completed readback
and excludes file encoding. The build used Rust optimization level 1 throughout.

| Native lighting | Three pinholes, rooms/s | Three circular fisheyes, rooms/s | Peak renderer VRAM, pinhole / fisheye |
| --- | ---: | ---: | ---: |
| Raster PBR | 1.53 | 1.30 | 2,046 / 3,008 MiB |
| Solari | 1.64 | 1.13 | 2,230 / 3,802 MiB |
| Solari + ReSTIR | 1.37 | 1.18 | 3,294 / 5,310 MiB |

Each output was 512×512. The pinhole test repeated eight seeds per mode in reversed
block order (16 measured rooms per mode). The circular test measured four additional
seeds per mode. Each process discarded two warmup rooms. Mean whole-GPU activity
ranged from 87% to 96% across measured pinhole blocks. The sample size and
concurrent workload limit inference about other GPUs, idle rendering and training
step time. We used 95% of raster throughput for each lens type as the threshold for
changing the default. Both traced modes fell below it on circular rigs and produced
visible grain in room 93400 at matched exposure. Raster PBR is the default for training.

To reproduce the pinhole workload, run the command once per renderer with a fresh
output directory. Append `--solari-preview` for the ordinary path tracer. Add
`--solari-restir` as well for reservoir resampling. A circular rig uses
`{"lenses":{"probability":1,"circular_probability":1,"elliptical_probability":0}}`.

```sh
cargo run --features solari,human_motion --example indoor_bench -- \
  --scenes 10 --warmup-scenes 2 --seed 93400 --cameras 3 --width 512 --height 512 \
  --rgb-only --preparation-threads 2 --task-pool-threads 3 \
  --indoor-camera '{"lenses":{"probability":0},"ceiling_mount":{"probability":0}}' \
  --output out/lighting-raster
```
