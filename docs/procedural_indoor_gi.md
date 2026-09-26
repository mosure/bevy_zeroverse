# Procedural diffuse global illumination

Native `Auto` now computes diffuse indirect illumination from the generated
scene, including its furniture and procedural people. No lightmap, room scan,
HDR photograph, or imported mesh is required. `Portable` and Wasm keep the
documented environment-light fallback. Bevy 0.17.3 explicitly disables
irradiance-volume shader bindings in Wasm, so the browser profile must not be
described as having this GI implementation.

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

Solari was considered but is not silently enabled: Bevy 0.17 describes its
hardware-raytraced implementation as experimental and its realtime path as
diffuse-only. That would change the existing material and platform contract.
See the official [Bevy 0.17 rendering notes](https://bevy.org/news/bevy-0-17/)
and the pinned [irradiance-volume implementation](https://docs.rs/bevy_pbr/0.17.3/src/bevy_pbr/light_probe/irradiance_volume.rs.html).
