# Device-local material baking

`BevyZeroverseConfig::indoor_gpu_materials` defaults to true and enables a once-per-recipe compute bake
for qualified indoor substrates. The equivalent CLI/query option is
`--indoor-gpu-materials true` / `indoor_gpu_materials=true`. Unsupported devices
and material programs use the existing CPU implementation; this switch never
selects a lower-detail material or changes the sampled recipe. CPU-only
`MaterialRecipe::maps` and `PreparedIndoor::build` keep returning complete CPU
images for export and reference rendering.

The GPU path implements woven fabric, knitted fabric (including shared human
jersey), carpet, leather, metal, chrome, plastic, rubber, paper and the wall
paint/plaster program. Wood, mineral/glaze, botanical, screen, artwork and
skin/hair maps currently retain CPU synthesis. This is a
hybrid backend, not a replacement of every procedural program.

Set `--indoor-gpu-materials false` / `indoor_gpu_materials=false` to use the CPU
reference for a matched performance or numerical comparison. Default Rust, CLI,
URL and legacy JSON configurations all enable the device-capable path.

## Data and scheduling contract

- Recipes retain their full seed, physical scale, phase, weave topology, yarn
  dye program and finish parameters. Texture resolution and PBR parameters do
  not change. Scene structure variants keep independent recipes/maps.
- Compute first writes albedo, metric height, AO and perceptual roughness, then
  derives tangent normals with the same repeating metric derivatives. Every mip
  retains the **unnormalized normal first moment** and mean fourth power of
  perceptual roughness. Unresolved relief broadens the GGX lobe instead of
  disappearing. Color mips use CPU-equivalent quantized sRGB lookup thresholds.
- Packed linear GPU buffers copy into ordinary RGBA8 material textures, including
  an sRGB albedo texture. PBR rendering subsequently samples those cached maps;
  it does not re-evaluate procedural fields per frame or per camera.
- Synthesis, normal derivation and each dependent mip use separate compute
  passes within the same command encoder. Immutable, cached 48-byte uniforms
  supply source/destination offsets from the same padded layout used for
  texture copies. The shader does not reconstruct mip addresses with loops.
  Bindings validate the uniform ABI sizes. This keeps stage dependencies and
  small-mip row pitches explicit across WebGPU/Metal and native backends,
  without a host fence, extra submission, or image readback.
- A device-local scratch allocation is shared sequentially across maps and
  scenes. Inputs and mip uniforms are small; there is no baked-image readback.
  The CPU evaluates the exact 32×32 point samples already used by the indirect
  lighting/reflection transport, preserving that existing lighting contract.
- Current-room baking precedes the camera pass. A bounded resident successor
  can also bake, and promotion retains its textures without uploading blank
  replacements. Capture readiness includes every bake and propagates shader
  failures; neither CPU image export nor GPU tensor capture can finish early.
- The backend uses published Bevy/wgpu APIs and standard compute/storage-buffer
  capabilities. Devices lacking them fall back to CPU. No CUDA dependency,
  wgpu patch, external memory extension or extra logical device is required.

`MaterialBakeStatistics` reports actual per-room GPU coverage, independently of
the requested switch. CPU-only rendering does not compile the bake pipelines or
allocate scratch storage. Compute shader work competes with training on the
shared device: measure completed work before selecting it for a workload.

## Reproducible qualification

Hardware tests read back the actual textures, including every mip, and compare
against the CPU recipe oracle. Tolerances are in encoded 8-bit units (maximum
three codes and RMSE 0.65 codes per map/mip), not a visual similarity score.
They also check exact CPU indirect-lighting samples, legacy recipe defaults,
64-bit seed carry, tiled derivatives, sRGB transfer boundaries and scratch layout.

```sh
cargo test --lib materials::gpu -- --include-ignored --test-threads=1 --nocapture
cargo test -p bevy_zeroverse_burn --features gpu_tensor --lib material_qualification -- --ignored --test-threads=1 --nocapture
```

Browser portability has a separate Chrome/Dawn check. Native Naga validation
can accept expressions that browser WGSL rejects, so the test compiles the
actual combined shaders and all eight compute pipelines. Run it with Node 22+
and Chrome:

```sh
node scripts/validate_material_webgpu.mjs
```

It uses SwiftShader by default to check portable WebGPU validation; it does not
measure hardware performance. Set `CHROME` to the browser executable if needed,
or pass `--native` to test the browser's hardware adapter. The CI test runs this
check independently of the Rust build.

For numerical browser qualification, export the CPU oracles and run:

```sh
cargo test --lib export_webgpu_material_fixtures -- --ignored --nocapture
node scripts/validate_material_webgpu.mjs --fixtures=out/material_webgpu/fixtures
node scripts/validate_material_webgpu.mjs --native --headful --fixtures=out/material_webgpu/fixtures
```

The 33 fixtures cover both map resolutions, woven/knit fabric, leather, carpet,
chrome, coated metal, plastic and paint/plaster. Consecutive recipes reuse one
scratch allocation in a single submission, including cloth after chrome. The
test checks every texture mip against CPU bytes and requires exact metalness.
It also renders GPU-baked and CPU-uploaded textures through repeat/trilinear/4×
anisotropic sampling at LOD -2 through 9.25, including oblique footprints and
texture seams. Shader compilation alone does not establish this contract.
No fixtures or readbacks are added to normal generation.

On macOS set `CHROME` to the Chrome executable (usually
`/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`). `--native`
uses that browser's hardware backend; the native Rust GPU tests also select the
platform backend instead of forcing Vulkan (`WGPU_BACKEND=metal` is supported).
Linux Vulkan, SwiftShader and macOS/Metal are separate qualification targets;
passing the first two does not certify the third.

`gpu_baked_fabric_matches_cpu_when_minified` renders nine paired
StandardMaterial probes: woven fabric, knit and leather, each at close, medium
and distant texture footprints. Both material and reflection baking switch
together, as in the viewer. PNGs and linear-RGB error measurements are written
under `out/material_portability/pbr/` for review.

The second test compares completed three-view, 512×512, two-timestep captures,
with static people, full native lighting, both annotation glass policies, RGB,
depth, world position, geometric normal, semantic ID, optical flow and directed
co-visibility. Geometric annotations must match exactly. RGB allows bounded
floating-point map differences. Procedural mesh ordinals make annotation depth
ties independent of recycled ECS entity IDs; no metric depth bias is added.

For throughput, alternate otherwise identical runs of:

```sh
cargo run -p bevy_zeroverse_burn --features gpu_training_bench --example material_jit_qualification -- --rooms 24 --warmup 4 --output out/material_cpu.json
cargo run -p bevy_zeroverse_burn --features gpu_training_bench --example material_jit_qualification -- --gpu-materials --rooms 24 --warmup 4 --output out/material_gpu.json
```

Repeat both with `--training-steps 4`. The consumer then runs a patch MLP with
forward/backward/Adam updates on all three rendered images, on the same WGPU
device, while the persistent renderer worker prepares the next sample. Both
loss and updated-weight reductions confirm completed training work. Reports
record capture/consumer wall times and completed rooms/views/optimizer steps per
second. This is a reproducible training workload, not a claim about a particular
downstream model. It does not replace testing that model's actual training loop.

Keep matched reports/galleries under ignored `out/`; do not copy local study
results into the project page or publication artifacts automatically. Report all
repetitions and shared-device contention, not only the fastest run. No shadow,
GI, human, geometry or texture quality setting is reduced by the benchmark.

## Window backgrounds and exposure

Procedural interiors include a closed, inward-facing sky proxy beyond the room
and surrounding geometry. Its radius is half the footprint diagonal plus 18 m;
it stays within the existing 50 m capture range for primary-room cameras. RGB
uses the scene's sky radiance and camera exposure. The proxy does not cast
shadows, enter the indirect-lighting BVH, enlarge the reconstruction AABB, or
enter O-voxel export. Existing analytic sky lighting remains authoritative.

The appended `sky` semantic class identifies this coarse background separately
from physical buildings. Its position, depth, normals, flow and co-visibility
refer to the finite proxy surface, **not** an astronomical sky distance. Existing
semantic IDs retain their values. Capture metadata records the proxy radius,
semantic ID, actual GPU material coverage and scene EV100.

Scene Studio → Appearance → Exposure offset controls all capture cameras and
the editor camera. CLI/API: `indoor_appearance` JSON with
`{"exposure_ev100_offset":1.0}` darkens RGB by one stop before tonemapping;
negative values brighten it. The supported offset range is [-2, 2] around the
seeded, daylight-aware exposure. Shared URLs retain this setting. Exposure does
not alter geometry annotations; the actual EV100 is exported with each room.

```sh
cargo test -p bevy_zeroverse_burn --features gpu_tensor --lib finite_exterior_and_exposure -- --ignored --test-threads=1 --nocapture
```

This hardware test samples ten rooms, three 512×512 cameras and two times,
including reported seeds 57541088 and 57491588. It requires positive finite depth
and labelled, unit normals at every pixel through glass, exercises actual sky
pixels, preserves the primary-room bounds, and separately checks RGB exposure
changes against exact depth/position/normal/semantic/flow/co-visibility equality.
