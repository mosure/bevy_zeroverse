# Capture timeout qualification: seed 71727123

Core/FFI 0.63.2, Burn bridge 0.47.2, capture 0.6.22 and publication 0.9.10
contain this repair. Capture identity is `capture-v143`; generator identity is
118 and the tabletop grammar identity is 18. Burn remains 0.22.1.

## Failure and repair

The reported eight-camera capture used three timesteps, 512×512 output, automatic
2× SSAA, mixed camera mounts/lenses, through-glass annotations and full native
lighting. Its renderer lane reached the 180-second deadline before IPC export.

Notebook instance 102 (object seed 2530243425158947287) contained an ink stroke
whose positive parameter interval rounded to identical float32 positions.
Normalizing its collapsed quad produced four NaN normals. CPU geometry checking
reproduced the defect. A cold registry 0.63.1 GPU capture also timed out with a
45-second diagnostic deadline; that replay used surface-glass annotations.

Annotation extraction already detected the invalid mesh. The all-view copy guard
then returned while the annotation was unready, before forwarding its failure.
The sampler therefore waited for a result that could never become valid.

Notebook construction now checks both triangle areas before emitting a quad or
normalizing its normal. Capture readiness forwards annotation, copy and pipeline
errors before scene/warmup barriers. GPU timeout messages also include readiness,
missing assets, pipeline counts, sampler progress and captured view counts.

## Validation

- Twenty-nine focused CPU tests cover the notebook at five scales, 256 notebook
  variants, complete room geometry, early error propagation, annotation validation,
  capture/readback barriers and timeout diagnostics.
- Native captures of 71727123, 71727122 and 71727123 again produce byte-identical
  RGB, depth, normals, semantics and co-visibility for the repeated room. Camera
  calibration, transforms and timestamps also match.
- Direct WGPU and fresh-process CUDA IPC each capture 71727122, 71727123 and
  71727124 with the reported eight-camera/three-timestep policy, through-glass
  annotations, full shadows/GI and 2× SSAA. Every capture contains 24 finite
  RGB/world-depth views. CUDA IPC verifies sRGB output, dimensions and SSAA
  receipts. No image pixels traverse the host in these transport checks.
- The room manifest and resampled cameras match the preceding source exactly,
  apart from the generator identity. Placement and camera sampling are unchanged.

For 71727123, the WGPU producer interval was 1.161 seconds; CUDA IPC completed
capture, import and finite-value validation in 2.408 seconds. These bounded,
optimized development-build checks demonstrate completion, not production
throughput. Cold initialization and shader compilation took longer. Concurrent
training throughput, long-run memory and browser runtime were not requalified.
The WASM library with `web,human_motion`, strict Clippy and formatting passed.

Detailed local evidence is retained under `out/seed_71727123` and
`out/release_0632`. Registry, GitHub workflow and deployment status belong to the
release receipt, separately from these source-level checks.
