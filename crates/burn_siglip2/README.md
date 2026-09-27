# burn_siglip2

The canonical source and release home for `burn_siglip2` is the
[bevy_zeroverse workspace](https://github.com/mosure/bevy_zeroverse/tree/main/crates/burn_siglip2).
The crate has its own version and can be used independently of Bevy and the scene
generator. It was originally developed in `burn_loom`; see [ORIGIN.json](ORIGIN.json)
for import provenance and [RELEASING.md](RELEASING.md) for the publication procedure.

Production fixed-resolution SigLIP2 inference for Burn 0.21. The crate implements both image and
text towers, canonical preprocessing/tokenization, calibrated SigLIP scoring, native BPK loading,
incremental BPK-shard loading, and a browser WebGPU interface.

## Supported checkpoints

The importer and production configuration validator intentionally support exactly these variants:

| Variant | Hugging Face model | Image / patch | Hidden / projection | Layers / heads |
| --- | --- | --- | --- | --- |
| `base-patch16-224` | `google/siglip2-base-patch16-224` | 224 / 16 | 768 / 768 | 12 / 12 |
| `large-patch16-256` | `google/siglip2-large-patch16-256` | 256 / 16 | 1024 / 1024 | 24 / 16 |
| `so400m-patch14-224` | `google/siglip2-so400m-patch14-224` | 224 / 14 | 1152 / 1152 | 27 / 16 |

These are the three smallest SigLIP2 architecture scales, represented by their smallest
fixed-resolution checkpoints. Tiny configs remain available for unit tests, but every production
loader rejects them.

## Features

| Feature | Purpose |
| --- | --- |
| `ndarray` (default) | Burn NdArray CPU backend and compatibility aliases |
| `flex` | Burn Flex portable SIMD CPU backend |
| `preprocess` | Encoded image decoding and canonical RGB/bilinear/mean-std preprocessing |
| `tokenizer` | Fixed-length SigLIP2 tokenization from Hugging Face tokenizer assets |
| `pipeline` | `preprocess` + `tokenizer` ergonomic inference APIs |
| `wgpu` | Burn WGPU backend using Vulkan on Linux/Windows and Metal on macOS |
| `webgpu` | Cross-target WebGPU/WGPU feature; includes `wgpu` and Burn's WGSL compiler |
| `wasm` | `wasm-bindgen` browser WebGPU API; also enables `webgpu` and `pipeline` |
| `import` | Hugging Face safetensors importer |
| `cli` | `siglip2_import` plus native bootstrap support |

The default native backend is Burn NdArray with SIMD and multithreading. Enable `flex` and use
`load_backend_flex` for Burn's portable SIMD + Rayon CPU kernels. Enable `webgpu` (or the
lower-level `wgpu` feature) and use
`load_backend_wgpu` for native GPU execution. Browser builds use unfused WebGPU because current
browser validation rejects some fused command scopes; native Vulkan/Metal builds retain fusion.
F16 is an artifact storage format; loaders expand stored values to F32 tensors for the current
backends.

## Native loading and multimodal inference

`LoadRequest::from_bpk` loads a monolithic BPK. `LoadRequest::from_parts_manifest` reads and
applies sibling shards in manifest order, reusing one host buffer and optionally checking each
declared SHA-256. Both paths validate the embedded production model configuration and expected
tensor set.

Enable `pipeline` and score encoded image bytes against raw text like this:

```rust
use std::path::Path;

use burn_siglip2::{LoadRequest, Siglip2Tokenizer, load_backend};

fn main() -> Result<(), String> {
    let runtime = load_backend(LoadRequest::from_parts_manifest(
        "/models/base/siglip2-base-patch16-224.bpk.parts.json",
        true,
    ))?;
    let tokenizer = Siglip2Tokenizer::from_file(
        Path::new("/models/base/siglip2-base-patch16-224.tokenizer.json"),
        &runtime.model.config,
    )?;
    let encoded_image = std::fs::read("cat.jpg").map_err(|err| err.to_string())?;
    let response = runtime.encode_image_bytes_and_text_strings(
        &encoded_image,
        &tokenizer,
        &["a photo of a cat", "a photo of a dog"],
        false,
    )?;
    let probabilities = response
        .probabilities_per_image
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| format!("{err:?}"))?;
    println!("{probabilities:?}");
    Ok(())
}
```

The same pipeline runs on Flex CPU with only the loader changed:

```rust
use burn_siglip2::{LoadRequest, load_backend_flex};

let runtime = load_backend_flex(LoadRequest::from_parts_manifest(
    "/models/base/siglip2-base-patch16-224.bpk.parts.json",
    true,
))?;
# Ok::<(), String>(())
```

Build and test Flex without pulling either NdArray or WGPU into the crate feature set:

```bash
cargo test -p burn_siglip2 --no-default-features --features flex,pipeline --lib --tests
```

For explicit adapter/device selection, use the backend-generic loader. For example, a native WGPU
application can choose the first discrete adapter rather than relying on the platform default:

```rust
use burn_siglip2::{
    DefaultWgpuBackend, LoadRequest, WgpuDevice, load_backend_on_device,
};

let runtime = load_backend_on_device::<DefaultWgpuBackend>(
    LoadRequest::from_parts_manifest(
        "/models/base/siglip2-base-patch16-224.bpk.parts.json",
        true,
    ),
    WgpuDevice::DiscreteGpu(0),
)?;
# Ok::<(), String>(())
```

The multimodal response contains raw and L2-normalized image/text embeddings,
`logits_per_image`, `logits_per_text`, and element-wise sigmoid probabilities. Logits use the
checkpoint's learned `exp(logit_scale)` and `logit_bias`.

### Native CDN bootstrap

With the `bootstrap` feature, the default configuration downloads the verified Base shards and
tokenizer assets from
`https://aberration.technology/model/siglip2/base-patch16-224/` and caches them below
`$HOME/.burn_siglip2/models/`. Select another supported model without hand-building paths:

```rust
use burn_siglip2::{
    Siglip2BootstrapConfig, Siglip2ModelVariant,
    resolve_or_bootstrap_siglip2_weights_with_config,
};

let config = Siglip2BootstrapConfig::for_variant(Siglip2ModelVariant::LargePatch16_256);
let artifacts = resolve_or_bootstrap_siglip2_weights_with_config(&config)?;
println!("{}", artifacts.parts_manifest_path.display());
# Ok::<(), burn_siglip2::ModelBootstrapError>(())
```

The zero-argument `resolve_or_bootstrap_siglip2_weights()` applies environment overrides:
`BURN_SIGLIP2_MODEL_BASE_URL` overrides the CDN root,
`BURN_SIGLIP2_REMOTE_ROOT` overrides the canonical model-size directory, and
`BURN_SIGLIP2_MODEL_STEM` overrides the file stem. Explicit per-file URL overrides remain
available. The `*_with_config` functions instead use their supplied typed configuration verbatim.
The public CDN is intentionally monolith-free: a failed parts download only falls back to a
monolithic model when `BURN_SIGLIP2_BPK_URL` was explicitly configured.

Canonical fixed-resolution behavior is intentionally specific:

- apply encoded EXIF orientation once, convert to RGB, resize directly to the configured square
  with bilinear resampling, rescale by `1/255`, then normalize each channel with mean/std `0.5`;
- lowercase text, append EOS, and right-pad with PAD to exactly 64 tokens using IDs derived from
  the tokenizer artifacts;
- omit the attention mask for Hugging Face checkpoint parity. Explicit `*_with_attention_mask`
  methods remain available for callers that deliberately want different semantics.

Image input is deliberately bounded before large allocations. Encoded images must be non-empty
and at most 64 MiB. Encoded and caller-provided `DynamicImage` values must be at most 16,384 pixels
on either axis and 64 megapixels in total. Decoders receive a 256 MiB allocation budget; because
that decoder limit is best-effort for formats that cannot enforce it, dimensions and area are
independently checked before full decode. The enabled codecs are PNG, JPEG, WebP, and BMP.

Text is lowercased and truncated or padded to exactly 64 tokens. Tokenizer and browser APIs accept
at most 256 strings per call, 64 KiB of UTF-8 per string, and 1 MiB of UTF-8 for the complete
batch. These limits are checked before lowercasing or tokenization allocations. Loaded tokenizer
vocabulary size, PAD/EOS IDs, and every emitted token ID are checked against the model config.
Missing `do_lower_case` metadata uses SigLIP2's canonical `true` behavior; an explicit `false` is
rejected.

`Siglip2Tokenizer::from_file` looks for a correspondingly prefixed sidecar such as
`siglip2-base-patch16-224.tokenizer_config.json`, then falls back to bare
`tokenizer_config.json`. Files and in-memory assets are bounded to 64 MiB for tokenizer JSON and
1 MiB for tokenizer-config JSON before parsing. Every tokenizer instance exposes an artifact
SHA-256 over the exact tokenizer/config bytes it parsed. Include this identity in any
persistent embedding schema. `Siglip2Tokenizer::from_bytes` provides the same validation for
already-fetched browser assets.

## Import F16 weights and create CDN bundles

The importer accepts a local Hugging Face directory and never needs to download weights. It
rejects missing, malformed, unexpected, or unused source tensors by default. Use a full immutable
Hugging Face commit in artifact metadata:

```bash
cargo run --release -p burn_siglip2 --features cli --bin siglip2_import -- \
  --hf-dir /models/siglip2-base-patch16-224 \
  --output assets/models/siglip2/base-patch16-224/siglip2-base-patch16-224 \
  --model-variant base-patch16-224 \
  --upstream-revision <full-40-character-hugging-face-commit> \
  --precision f16 \
  --parts true \
  --parts-max-mib 64 \
  --parts-overwrite \
  --copy-tokenizer-assets true \
  --reject-unused-source-tensors true
```

Repeat into matching per-variant directories for `large-patch16-256` and
`so400m-patch14-224`. The importer requires one local `model.safetensors` plus `config.json` and
reads the model file in full, so allow substantial transient host RAM. A complete image/text CDN
bundle also requires `tokenizer.json`, `tokenizer_config.json`, and `preprocessor_config.json`;
`tokenizer.model` is copied when present. Imported sidecars are prefixed with the output stem.
The supplied revision is format-validated and recorded, but the importer does not independently
prove that the local checkpoint came from that revision.

To re-shard an already imported BPK without re-reading the Hugging Face checkpoint, run:

```bash
cargo run --release -p burn_siglip2 --features cli --bin siglip2_parts -- \
  --bpk assets/models/siglip2/siglip2-base-patch16-224.bpk \
  --max-mib 64 --overwrite
```

Create strict, monolith-free upload directories after all three imports. From the workspace root,
use the crate-local script that is also shipped in the crate package:

```bash
BURN_SIGLIP2_CDN_BUNDLE_STRICT=1 \
  bash crates/burn_siglip2/scripts/bundle_siglip2_assets.sh \
  assets/models/siglip2 dist/cdn/siglip2
```

From an unpacked `burn_siglip2` crate package, invoke
`bash scripts/bundle_siglip2_assets.sh <source-root> <destination-root>` instead. Both source and
destination arguments are resolved from the caller's working directory.

The output layout is:

```text
dist/cdn/siglip2/
├── index.json
├── SHA256SUMS
├── base-patch16-224/
│   ├── bundle.manifest.json
│   ├── SHA256SUMS
│   ├── siglip2-base-patch16-224.bpk.parts.json
│   ├── siglip2-base-patch16-224.bpk.part-00000.bpk
│   ├── ...
│   ├── siglip2-base-patch16-224.tokenizer.json
│   ├── siglip2-base-patch16-224.tokenizer_config.json
│   └── siglip2-base-patch16-224.preprocessor_config.json
├── large-patch16-256/
└── so400m-patch14-224/
```

Upload the contents of `dist/cdn/siglip2/` to
`https://aberration.technology/model/siglip2/` while preserving this hierarchy. Source `.bpk`
monoliths are intentionally excluded from every variant directory.

### Portable bounded shards

The part writer converts the 256,000-row text embedding into 16 canonical row chunks before
packing BPK parts. Each chunk maps directly to one backend embedding buffer, so the loader never
has to materialize the original 375--562.5 MiB F16 tensor. Other safetensors entries remain atomic.
The packer reserves header space below the requested target, and the strict CDN bundler rejects
any serialized part over 64 MiB or any manifest whose requested limit was not enforced.

Production loaders assemble the model through an uninitialized, completeness-checked builder:
each decoded tensor is moved into its final backend slot exactly once, and the full runtime is only
constructed after every expected key has been accepted. They do not allocate a complete F32
zero-filled model before applying the real shards. `Siglip2Model::zeros` remains available for
small deterministic tests.

The upload-ready artifacts generated in this repository contain 14 Base, 35 Large, and 42 So400m
model parts. Their largest files are respectively 64,981,055, 65,086,475, and 64,001,677 bytes,
all below 64 MiB. The manifest records the `portable_text_embedding_chunks_v1` strategy, exact
part sizes/digests, source provenance, aggregate tensor count, and whether the requested limit was
enforced. Native monolithic BPK loading remains compatible with the original full-table key.

Successful loaders expose two distinct checksums in `PartLoadStats`.
`weight_payload_sha256` preserves the source artifact's declared monolithic payload checksum for
provenance; a parts manifest can only assert that value. `loaded_weight_sha256` is the cache-safe
model identity: the loader derives it from every accepted tensor's canonical name, storage dtype,
shape, and raw bytes and publishes it only after the complete tensor set passes validation. It is
independent of shard/fetch order and gives the full text table and its canonical row chunks the
same identity. Embedding databases should always key on `loaded_weight_sha256`.

## Browser/WebGPU

The crate emits both `rlib` and `cdylib` artifacts. Its `wasm` feature includes WebGPU,
preprocessing, tokenization, and the `WasmSiglip2` bindings. A verified release build is:

```bash
cargo build -p burn_siglip2 --lib --target wasm32-unknown-unknown --release \
  --no-default-features --features wasm
```

This creates `target/wasm32-unknown-unknown/release/burn_siglip2.wasm`. Generate the JavaScript
glue with the host application's `wasm-bindgen` packaging workflow. The repository's real-model
browser harness uses:

```bash
wasm-bindgen --target web --out-dir artifacts/wasm/burn_siglip2 \
  target/wasm32-unknown-unknown/release/burn_siglip2.wasm
wasm-tools validate artifacts/wasm/burn_siglip2/burn_siglip2_bg.wasm
```

A dependency-free application is shipped in [`examples/web`](examples/web/README.md). It selects
Base, Large, or So400m, verifies the loaded model identity, previews an uploaded image, accepts a
batch of candidate texts, and renders ranked calibrated scores plus embedding diagnostics. Its
README contains package-relative build and serving commands that also work from an unpacked
crates.io archive.

Once packaged by the host application, load a generated bundle and run image/text scoring:

```javascript
import init, { WasmSiglip2 } from "./pkg/burn_siglip2.js";

await init();
const model = await WasmSiglip2.createDefault();
// Or: await WasmSiglip2.createFromModelSize("large");
// Or: await WasmSiglip2.createFromBundleManifestUrl(
//   "https://aberration.technology/model/siglip2/base-patch16-224/bundle.manifest.json",
// );
const encodedImage = new Uint8Array(
  await (await fetch("cat.jpg")).arrayBuffer(),
);
const imageEmbedding = JSON.parse(
  await model.encodeImageBytesJson(encodedImage),
);
const textEmbeddings = JSON.parse(
  await model.encodeTextsJson(["a photo of a cat", "a photo of a dog"]),
);
const scores = JSON.parse(
  await model.scoreImageTextsJson(encodedImage, [
    "a photo of a cat",
    "a photo of a dog",
  ]),
);
console.log(imageEmbedding.normalized_embedding);
console.log(textEmbeddings.normalized_embedding);
console.log(scores.probabilities_per_image);
console.log(model.loadedWeightSha256, model.tokenizerSha256);
console.log(model.upstreamModelId, model.upstreamRevision);
```

All response arrays are row-major and flattened according to their shape fields. Standalone
`encodeImageBytesJson` and `encodeTextsJson` return:

```jsonc
{
  "schema_version": 1,
  "method": "encodeImageBytesJson",
  "shape": [1, 768],
  "raw_embedding": [/* 768 row-major f32 values */],
  "normalized_embedding": [/* 768 row-major f32 values */]
}
```

For text, `method` is `encodeTextsJson` and the first shape dimension is the text batch size.
`scoreImageTextsJson` preserves its original shape/logit/probability fields and adds
`schema_version`, `method`, `raw_image_embedding`, `normalized_image_embedding`,
`raw_text_embedding`, and `normalized_text_embedding`. Thus browser callers can use either tower
independently or obtain embeddings and calibrated scores from one combined execution.

`createDefault` selects `base-patch16-224`. `createFromModelSize` accepts `base`, `large`,
`so400m`, or a complete canonical variant name and resolves it below
`https://aberration.technology/model/siglip2/`. `createFromBundleManifestUrl` validates the outer
schema, F16 storage declaration, safe unique
inventory paths, non-zero declared lengths, SHA-256 syntax, and inventory byte total. It fetches the
parts manifest with the outer inventory's exact byte length and SHA-256, validates the inner
manifest, requires exact outer/inner variant, model ID, revision, and storage provenance, and
cross-checks every inner shard path, size, and digest against the outer inventory.
URL constructors prefer and preflight a high-performance WebGPU adapter before transferring model
metadata, tokenizer files, or shards. When that preference yields no usable adapter, they retry
the low-power/integrated path and retain the selected device for subsequent constructors.
Every fetched shard is checked again against the inner manifest before its tensors are applied to
WebGPU. `Response.body` is consumed incrementally with a running SHA-256 into one bounded Wasm
buffer; no whole-response JavaScript `ArrayBuffer` copy is created, and the Wasm buffer is dropped
before the next request. The runtime fetches and verifies both `tokenizer.json` and
`tokenizer_config.json` against the outer inventory.
`tokenizer.model` and `preprocessor_config.json` remain checksummed distribution/provenance assets,
but this fixed-resolution runtime does not fetch or consume them. `loadedPartCount` and
`loadedBytes` expose load evidence. The `modelSize`, `imageSize`, and `embeddingSize` getters
expose the validated loaded profile. `loadedWeightSha256` identifies the complete canonical tensor
set independently of monolithic versus sharded transport, while `tokenizerSha256` identifies the
exact tokenizer/config bytes. `upstreamModelId` and `upstreamRevision` expose validated artifact
provenance. `tokenizerSha256` is `undefined` for the deliberately image-only
`createFromManifestUrl` runtime; production parts manifests always provide the two upstream
getters. Browser embedding caches should include both weight and tokenizer fingerprints in their
keys.

Browser HTTP and model limits are constructor-specific:

| Resource | Limit |
| --- | ---: |
| Root bundle manifest | 8 MiB |
| Standalone parts manifest passed to `createFromManifestUrl` | 8 MiB |
| Parts-manifest JSON passed to `createFromParts` | 8 MiB |
| Standalone tokenizer JSON URL | 64 MiB |
| Standalone tokenizer-config URL | 1 MiB |
| Bundle tokenizer / optional tokenizer model | 64 MiB each |
| Bundle tokenizer-config / image preprocessor | 1 MiB each |
| Parsed model manifest | 256 parts |
| Parsed model-shard bytes | 4 GiB aggregate |
| One model shard | 64 MiB |
| One encoded image | 64 MiB |
| One text / text batch | 64 KiB / 1 MiB |
| Texts per call | 256 |

The same role-specific ceilings apply to verified files declared by the outer bundle and to
caller-provided arrays passed to `createFromParts`. For identity-encoded responses,
an exposed `Content-Length` is used as an early upper-bound check when no exact decoded length is
known. Because Fetch exposes decoded bytes while `Content-Length` can describe a compressed HTTP
representation—and `Content-Encoding` may not be CORS-exposed—the decoded stream remains the
authority. Every response is bounded while streaming and its final decoded length and SHA-256 are
checked against the inventory before parsing or applying it.
These checks establish internal consistency relative to the fetched bundle manifest; serve that
root manifest over trusted HTTPS or pin its digest to establish authenticity.

`createFromManifestUrl` is intentionally image-only; it rejects unverified tokenizer URLs.
`createFromManifestUrlWithTokenizerHashes` enables text inference when the caller supplies the
tokenizer and tokenizer-config URLs plus both expected SHA-256 values.
`createFromParts` accepts already-fetched `Uint8Array` shards for service-worker or IndexedDB
caches and requires tokenizer/config arrays as a pair, but retaining all arrays has a higher peak
host-memory cost. Prefer the complete bundle constructor when possible. URL loading currently
uses `window.fetch` and therefore targets a browser window rather than a Web Worker.

F16 is the network/storage representation; the current Burn WebGPU backend expands parameters to
F32 device tensors. Bounded shards keep transient host memory predictable, but complete-model
device residency is still substantial—especially for Large and So400m. Keep headroom for
activations and browser/driver allocations. Depending on the browser and driver, exhaustion can
surface as a rejected constructor, WebGPU device loss, or renderer termination; it cannot always
be converted into a Rust error after the device is lost.

Tanh-GELU clamps its `tanh` argument to the exactly saturated F32 interval `[-10, 10]`. This is
bitwise-equivalent to PyTorch F32 for the observed checkpoint activations and avoids non-finite
WGSL results on adapters whose `tanh` implementation overflows its internal exponentials.

The CDN must allow browser CORS requests for JSON, tokenizer, and BPK files.

## Numerical correctness

The real-checkpoint oracle generator is local-only and supports the same three variants:

```bash
python3 crates/burn_siglip2/scripts/siglip2_reference.py \
  --hf-dir /models/siglip2-base-patch16-224 \
  --variant base-patch16-224 \
  --weight-precision f16 \
  --upstream-revision <full-40-character-hugging-face-commit> \
  --output-base /tmp/siglip2-base-patch16-224-f16-reference
```

Compare the resulting immutable reference against a BPK or parts manifest:

```bash
BURN_SIGLIP2_NUMERICAL_REFERENCE=/tmp/siglip2-base-patch16-224-f16-reference.json \
BURN_SIGLIP2_NUMERICAL_BUNDLE=/models/base/siglip2-base-patch16-224.bpk.parts.json \
cargo test -p burn_siglip2 --features pipeline --test real_numerical_parity \
  opt_in_real_hf_reference_matches_imported_bundle --release -- --nocapture
```

This comparison first covers the exact reference pixel/token tensors and all eight model outputs.
With `pipeline` enabled it additionally decodes the checksummed PNG, preprocesses it from encoded
bytes, tokenizes the original strings, requires exact Hugging Face token IDs, and compares all
eight byte/string pipeline outputs. F16 mode rounds upstream parameters through IEEE F16 before
F32 compute, matching the loader's F16-storage behavior.

The final F16 CDN parts bundles produced these end-to-end maximum absolute errors:

| Variant | Pixels | Token IDs | Raw image | Raw text | Logits | Probabilities |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Base/16-224 | `1.192093e-7` | exact | `5.245209e-6` | `7.629395e-6` | `4.768372e-6` | `1.000444e-11` |
| Large/16-256 | `1.192093e-7` | exact | `3.814697e-6` | `7.629395e-6` | `3.814697e-6` | `1.136868e-11` |
| So400m/14-224 | `1.192093e-7` | exact | `2.861023e-6` | `5.722046e-6` | `3.814697e-6` | `1.159606e-11` |

Each row is a separate real-checkpoint run against its upload-ready parts manifest. All eight
encoded-image/string outputs passed their configured parity thresholds for every variant.

The opt-in live-CDN regression starts from the production default URL, verifies every downloaded
Base shard, loads both towers and compares the deterministic image/text outputs with the same
reference values. Set a persistent cache directory to avoid downloading roughly 790 MB again:

```bash
BURN_SIGLIP2_LIVE_CDN_CACHE=/tmp/burn_siglip2-live-cdn-base \
cargo test --release -p burn_siglip2 --all-features --test live_cdn_base \
  public_default_base_bundle_matches_reference -- --ignored --nocapture
```

Fast checks that do not load a checkpoint are also available:

```bash
python3 crates/burn_siglip2/scripts/siglip2_reference.py --self-test
cargo test -p burn_siglip2 --test real_numerical_parity \
  comparison_math_and_schema_smoke_are_deterministic
cargo check -p burn_siglip2 --target wasm32-unknown-unknown \
  --no-default-features --features wasm
```

The tensor phase isolates model math and weight mapping, while the `pipeline` phase validates
Burn's encoded-image preprocessing and tokenizer against Hugging Face before running combined
inference. Unit tests additionally pin tokenizer EOS/padding/lowercasing and Pillow-compatible
bilinear pixels.

Native WGPU has both a small dispatch smoke and an opt-in real Base dual-tower parity test. The
real test validates raw image/text embeddings, logits, and probabilities against the same F16
Hugging Face oracle:

```bash
BURN_SIGLIP2_WGPU_REFERENCE=/references/base-patch16-224-f16-reference.json \
BURN_SIGLIP2_WGPU_BUNDLE=/models/base/siglip2-base-patch16-224.bpk.parts.json \
cargo test --release -p burn_siglip2 --features wgpu,pipeline \
  --test real_wgpu_parity opt_in_real_wgpu_dual_tower_matches_reference_logits \
  -- --nocapture
```

An additional real-model matrix loads every selected bundle independently on NdArray, Flex CPU,
and native WGPU; runs encoded-image preprocessing, tokenization, both towers, and calibrated
scoring; and compares every returned embedding, logit, and probability with finite/shape/sigmoid
checks. Omitting `BURN_SIGLIP2_BACKEND_MATRIX_VARIANTS` runs all three model sizes:

```bash
BURN_SIGLIP2_BACKEND_MATRIX_ROOT="$PWD/dist/cdn/siglip2" \
BURN_SIGLIP2_BACKEND_MATRIX_IMAGE=/tmp/siglip2-reference.input.png \
cargo test --release -p burn_siglip2 --features flex,webgpu,pipeline \
  --test real_backend_matrix opt_in_all_model_sizes_match_across_ndarray_flex_and_wgpu \
  -- --nocapture
```

The checked-in browser test is also opt-in because it loads the selected 750 MB--2.27 GB shard
payload. It self-hosts the built Wasm and CDN tree, launches headless Chrome with selectable
SwiftShader or native WebGPU (SwiftShader by default), creates the deterministic oracle image
in-browser, and tests each tower independently plus calibrated combined scoring:

```bash
BURN_SIGLIP2_WASM_E2E=1 \
BURN_SIGLIP2_WASM_E2E_MODEL_SIZE=base \
node crates/burn_siglip2/tests/wasm_browser_e2e.mjs
```

Set `BURN_SIGLIP2_WASM_E2E_MODEL_SIZE` to `base`, `large`, or `so400m`. Every selection performs
real standalone image inference, standalone text inference, and combined scoring with shape,
finite-value, unit-norm, standalone/combined embedding-parity, sigmoid-calibration, and fixed F16
reference logit/probability assertions. The references use the same deterministic image and texts
as the per-variant Hugging Face parity workflow above. Final native-adapter Chrome runs produced:

| Variant | Standalone/combined embedding error | Logit max abs | Probability max abs |
| --- | ---: | ---: | ---: |
| Base/16-224 | `0` | `1.735950e-5` | `7.821403e-11` |
| Large/16-256 | `0` | `1.400000e-5` | `1.630000e-11` |
| So400m/14-224 | `0` | `2.300000e-5` | `7.890000e-11` |

Each run also matched the pinned weight, tokenizer, model-ID, and revision identities. The harness
requires Node 22 or newer and accepts `BURN_SIGLIP2_WASM_E2E_CHROME`,
`BURN_SIGLIP2_WASM_E2E_ADAPTER=swiftshader|native`, and
`BURN_SIGLIP2_WASM_E2E_TIMEOUT_MS` overrides.

See `scripts/README.md` for the reference schema, tolerance policy, and exact preprocessing
semantics.

## Crates.io package verification

The package includes source, tests, licenses, reference/bundling scripts, fixtures, and the web
example. Multi-gigabyte model bundles, generated Wasm glue, workspace applications, and local
caches are excluded. Verify the exact archive and its clean-room build before publishing:

```bash
cargo package --list -p burn_siglip2 --allow-dirty --locked
cargo publish --dry-run -p burn_siglip2 --allow-dirty --locked
```

The crate's default build is NdArray CPU. The following consumer-facing feature checks cover the
independent CPU, native GPU, and browser configurations:

```bash
cargo check -p burn_siglip2 --no-default-features --features flex,pipeline
cargo check -p burn_siglip2 --no-default-features --features webgpu,pipeline
cargo check -p burn_siglip2 --target wasm32-unknown-unknown \
  --no-default-features --features wasm
```
