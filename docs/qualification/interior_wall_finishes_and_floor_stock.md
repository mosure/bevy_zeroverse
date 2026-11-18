# Interior wall finishes, floor stock and wall lighting

## Construction and appearance

Perimeter walls in the polygonal room generator can carry a lower lining of timber veneer, acoustic fabric or ceramic slabs. A room samples a product and an installation height. Eligible wall runs have enough solid area below that height to accept panels. Nearby walls can share the same stock, with their own starting cut. Kitchens favor washable slabs, teaching spaces favor acoustic finishes, and meeting rooms and lounges favor veneer. Each program also admits other finishes and plain walls.

The installer cuts the field face to the window jambs, heads and sills. The joints have a real recessed bed and beveled returns. All finish geometry lies within the wall section, leaving the interior mounting plane and the window aperture intact. Texture coordinates are metric and follow each physical panel through a cut. The meshes use the existing wall semantic label and the same geometry for shadows, depth, normals, co-visibility and voxelization. Archived envelope walls with no lining field keep their previous construction. The per-wall installation is recorded in the manifest and the population metrics export includes material, installed wall count, height, pitch and joint depth.

Carpets sample a ground yarn, a lighter heather and a darker fleck from coordinated dye lots. Their relative hues can vary independently. Longer pile increases resolved relief and broadens the highlight. Carpet cuts show backing related to the yarn color. The mineral sampler favors limestone, earth and slate colors, with subdued green stone and a smaller tail of colored glazes. Mineral grout can follow the binder or contrast with it. These parameters use the existing base color, tangent-space normal and occlusion/roughness/metallic maps. Both backends sample the same recorded material recipe at the existing texture resolution and filtering settings.

Wall sconces fit their assigned mounting bays with upward, downward and canted assemblies. The backplate remains flush. One construction pose drives the shade, diffuser, physical light aperture, direct spotlight and diffuse transport. Emitted luminance uses the actual diffuser radius. Existing circuits still set power and color temperature.

## Qualification

The consecutive Mixed population comprised seeds 31000–31255, three requested cameras per room and all 20 room programs. All 256 passed layout, camera and full geometry validation. The meshes included 19,454 object bounds, 490 human bounds and 144,739,453 triangles. The triangle scan found zero degenerate triangles. The same seeds previously produced 144,694,999 triangles. Total mesh batches grew from 116,181 to 116,331.

Of the 256 rooms, 105 retained plain perimeter walls, 107 had one lined wall and 44 had two. The installed stock comprised 84 veneer, 56 acoustic and 55 ceramic runs. Height ranged from 0.75 to 2.10 m, module width from 0.37 to 1.51 m, and recess depth from 1.6 to 9.9 mm. Of 150 assembled wall sconces, 58 were uplights and 92 were downlights. An independent 256-product test measures the position of the actual diffuser relative to the light origin, its projected area, construction bounds, attachment and pose replay.

### Matched native captures

The render study reused the same 12 consecutive Mixed seeds (44210–44221) and three OpenOffice seeds (57040–57042) with a mixed-lens policy. Each room was captured from three views at 224 × 224 with 2× linear SSAA, shadows, indirect lighting and GPU material baking on Mesa software Vulkan. All object and human manifests, calibrated intrinsics, extrinsics and exposure settings matched their input counterparts. The sample contains 38 pinhole and seven fisheye views.

| Measurement | 12 mixed rooms | 3 mixed-lens offices |
| --- | ---: | ---: |
| Updated directed primary overlap, 10th / 50th / 90th percentile | 37.85 / 56.34 / 76.61% | 7.92 / 33.78 / 70.95% |
| Input median directed primary overlap | 56.30% | 33.79% |
| Semantic label changes across matched pixels | 2 / 1,806,336 | 0 / 451,584 |
| Exact per-peer mask changes across matched pixels | 429 / 1,806,336 | 91 / 451,584 |
| Updated mean deep-shadow coverage, linear luminance below 0.01 | 1.14% | 0.07% |
| Updated mean near-clipped RGB coverage, any channel over 0.99 | 0.55% | 0.34% |

All 45 exported co-visibility validity images and per-peer bitmasks independently reproduced the reported valid, shared, per-peer and overlap-cardinality counts. The largest per-view 99th percentile reprojection error was 0.000221 px and the largest depth–position discrepancy at that percentile was 3.34 µm. These are internal agreement checks on dataset targets. The mean absolute RGB change on pixels labeled floor in both captures was 15.64 out of 255 in the mixed rooms and 23.27 in the offices. This measures the size of the visible intervention. Similarity to real photographs requires a separate benchmark.

Three additional 512 × 512 showroom views of seed 44211 were checked at 2× linear SSAA. RGB, depth, position, normal and semantic exports had the requested dimensions. Independently decoding the co-visibility masks reproduced all three cameras’ per-peer and overlap-cardinality counts. The largest per-view 99th percentile reprojection error was 0.000252 px.

### Independent lighting reference

The kitchenette at seed 44212 contains a ceramic wall lining and two wall sconces. Its 525,277 exported triangles, material maps, physical lights and calibrated camera supplied a 160 × 160 Blender 5.2.2 LTS Cycles reference. Two CPU renders used distinct sampling seeds, 96 samples per pixel and ten bounces each. Both used raw scene-linear output. The native comparison applied the recorded camera exposure with no exposure fitting. Six analytic controls passed on the same Blender build, including transmission of a sun through a glass slab and absorption in a tinted volume. All 49 geometry probes agreed on the visible surface, with a 2.67 µm 99th-percentile position difference.

The two reference renders agreed to 0.46% in mean luminance and 2.95% in relative RGB mean absolute error when averaged in 32 × 32 pixel blocks. At the same scale the native renderer’s error against Cycles was 26.73%, and its whole-image mean luminance was 15.8% higher. On the floor mask the native mean luminance was 1.64× the reference, while the wall mask measured 0.92×. A uniform exposure correction would move these two surfaces in opposite directions relative to their targets. Individual reference pixels still had substantial sampling noise, so this comparison qualifies coarse light distribution for one camera. Other lighting programs, photographic equivalence and downstream transfer remain unmeasured by this comparison.

### Checks and scope

Seventy-four focused Rust tests passed, plus the opt-in eight-camera regression over eleven previously difficult training seeds. Independent geometric rays across oblique wall runs checked window holes, surface coverage, relief and outward normals. The desk-lamp texture test measures the planar cap and the cylindrical side separately, preserving the metric check on both charts. The native CPU and WGSL material comparison covered 193 recipes and 5,328 map/mip combinations. The maximum encoded difference was one byte level and the worst per-level RMSE was 0.03125 byte levels. Strict native Clippy, Rust formatting, the WASM library check and 15 reference-render contract tests passed. WASM qualification here covers compilation.

### Preparation cost and memory

The native preparation benchmark used 32 identical Mixed room seeds per pass, one request in flight, two pinned physical CPU cores, two preparation workers and four separate warmup rooms. The run order was input, updated, updated, input. Measured rates in order were 0.927, 0.932, 0.943 and 0.940 CPU-prepared rooms/s. The pooled change was +0.43%, within the observed run-to-run variation. Median room preparation took 0.978 s for the input and 0.976 s for the update. Both p95 values were 1.84 s. These figures measure CPU preparation with GPU material recipes in a development build. Production GPU rendering and concurrent training throughput require a dedicated hardware measurement.

Mean consumption per room rose from 400.59 to 401.22 meshes, 163.84 to 164.03 images and 197.59 to 197.88 materials. A lining occasionally requires a stock variant that other objects in the room did not use. Peak charged memory was 0.96 GiB across the timing runs, 3.64 GiB for the mixed captures and 3.97 GiB for the mixed-lens captures. Each run had an 8 GiB hard memory limit and zero swap. Long-duration process stability remains a separate measurement.

The render source digest was `48a7f65e081d9f3c3c427dc7b012710ae0db7458f6130c0ebcd2bf3854543db8`, generator 126, capture v152. The subsequent desk-lamp test correction changed the checkout source digest to `e85828fc83ebe9283180aa1a436b2345e2b0853967df111f9c326e6f9d8a0170`. A hash audit verified that this test body was the only difference between the two generator source trees.

The population and matched mixed views can be regenerated with these validator arguments.

```sh
indoor_validate --seed 31000 --audit-seeds 256 --audit-geometry --layout mixed \
  --cameras 3 --width 224 --height 224 --output out/finish-population
indoor_validate --seed 44210 --audit-seeds 12 --audit-geometry --renders 12 \
  --layout mixed --cameras 3 --width 224 --height 224 --labels --co-visibility \
  --no-raw --output out/finish-views
```

### Native throughput with full-quality lighting

This measurement isolates the cost of preparing and capturing the latest finishes. The input and optimized generators both include the wall linings, floor stock and directional sconces described above. A separate CPU comparison against the generator *before* those finish changes used the same seeds and measured 0.887 versus 0.905 prepared rooms/s (+1.9%).

| Workload | Input | Optimized | Change | Unit |
| --- | ---: | ---: | ---: | --- |
| CPU preparation, two physical cores | 0.890 | 0.933 | +4.9% | rooms/s |
| Headless capture with CPU readback, six modes | 5.28 | 5.46 | +3.2% | views/s |
| Same-device Burn WGPU, five modes | 6.50 | 6.56 | shared GPU | views/s |
| Same-device Burn WGPU, 48 additional rooms | 5.44 | 6.43 | shared GPU | views/s |

The workstation had an Intel i9-12900K and an NVIDIA RTX PRO 6000 Blackwell running Vulkan driver 610.43.02. Both builds used development optimization level 1, including dependencies. One persistent renderer had eight physical performance cores. Each Mixed room supplied three 512 × 512 views at one timestep, 2× linear SSAA, native shadows, GPU material synthesis and three-bounce diffuse lighting with 256 rays per source probe. The capture run included RGB, depth, position, normals, semantics and co-visibility. The Burn run exported the first five attachments on Bevy’s WGPU device and consumed each RGB tensor with a fenced scalar reduction. Capture means completed delivery including validation, readback and housekeeping. The measurements exclude file encoding, startup, four warmup rooms and process shutdown.

The first three workloads used seeds 44210–44233. CPU preparation and Burn used run order A–B–B–A. Headless capture used A–B–B–A–A–B. The independent Burn cohort used seeds 44234–44281 and A–B–B–A–B–A after local compilation finished. Each rate pools completed work over elapsed time.

Other GPU work varied during the Burn runs. Both versions sometimes shared the card with roughly 38 GiB already allocated. The larger cohort’s observed input rates ranged from 4.64 to 6.40 views/s, and the optimized rates ranged from 5.92 to 6.86. In the adjacent B3/A4 pair, both processes started with about 1.6 GiB on the device and peaked near 5.2 GiB. Those runs delivered 6.59 optimized and 6.40 input views/s. One comparable pair supports a capacity check. The full cohort has too much variation in competing GPU work to assign a causal JIT speedup. The separate headless readback runs all started near 1.6 GiB and peaked below 3.6 GiB. Their measured-window GPU busy fraction averaged 23% and 20%, respectively, across the card used for desktop display.

The probe-classification and shadow trees reuse triangle bounds and centroids. CPU shadow traversal and the GPU upload share 32-byte nodes. The material atlas reuses the wrapped lattice coordinate during interpolation. Indirect lighting traces each distinct probe origin and seed once and copies all six irradiance lobes to its relocated cells. Across 24 rooms, this reduced primary paths from 9,237,248 to 8,755,968 and mean lighting transport data from 48.79 to 46.04 MiB/room. The optional `traced_primary_rays` statistic records executed paths alongside the existing budget field.

All 108 float32 planes, six room manifests and six calibration/annotation metadata files matched byte for byte over 11 fisheye and seven pinhole views. Every view had finite outputs and its co-visibility channel reproduced the number of set peer bits, excluding the source camera. Independent CPU replay also matched the content fingerprints of 1,125 meshes and 473 images. On hardware, all 2,880 RGBA16F texels matched the original lighting shader across eight volumes covering unoccupied grids, partial relocation, a single valid source and the all-solid fallback.

The original consecutive Burn stream reproduced a process-exit crash after delivering its samples. A debugger captured a Bevy lighting task allocating a Vulkan buffer while the main thread was unloading the NVIDIA driver. The convenience stream now retires the renderer on ordinary drop, bounded by its capture timeout. Its explicit shutdown method accepts a caller deadline. Both timed Burn builds included this same shutdown fix and all timed processes exited successfully. A retained three-view Burn sample also remained readable after explicit shutdown. The largest measured capture-process memory charge was 3.23 GiB under an 8 GiB hard limit with zero swap. This short run qualifies the native unfused WGPU consumer. Production CUDA training, long-run memory behavior, release-profile rates and browser rendering speed were outside this measurement.

The optimized generator source digest is `e1ca5c1840c6ee863f2761eb00bc26a9829883098966579ffd1a629bb1fc17fe`. Native indoor generation selects these optimizations automatically.
