# burn_siglip2 browser example

This is a dependency-free static application for the crate's `wasm-bindgen` WebGPU API. It can
load Base, Large, or So400m from the public SigLIP2 CDN, preview an uploaded image, tokenize text
candidates, execute both model towers, and display calibrated image-to-text scores.

The application is package-relative: the commands below work from this directory in either the
repository checkout or an unpacked `burn_siglip2` crate package.

## Prerequisites

- a recent stable Rust toolchain;
- the `wasm32-unknown-unknown` target;
- `wasm-bindgen-cli` compatible with the crate's resolved `wasm-bindgen` dependency;
- a WebGPU browser and enough GPU memory for the selected model;
- Python 3, or another static HTTP server.

Use the `wasm-bindgen-cli` version matching the workspace or packaged lockfile:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126 --locked
```

## Build and serve

Run these commands from `examples/web/`:

```bash
export BURN_SIGLIP2_WEB_TARGET="$PWD/target"

CARGO_TARGET_DIR="$BURN_SIGLIP2_WEB_TARGET" \
  cargo build --manifest-path ../../Cargo.toml --lib --release \
  --target wasm32-unknown-unknown --no-default-features --features wasm

wasm-bindgen --target web --out-dir pkg \
  "$BURN_SIGLIP2_WEB_TARGET/wasm32-unknown-unknown/release/burn_siglip2.wasm"

python3 -m http.server 8080
```

Open <http://127.0.0.1:8080/>. Do not open `index.html` directly with a `file:` URL: WebGPU and
JavaScript module loading require a secure context, and localhost receives that treatment in
modern browsers.

The page's Content Security Policy permits model requests only to
`https://aberration.technology`. To self-host a bundle, add its HTTPS origin to `connect-src` in
`index.html` and call `WasmSiglip2.createFromBundleManifestUrl(...)` from `app.js`.

## Resource expectations

The public F16 bundles are approximately 0.79 GB, 1.80 GB, and 2.31 GB for Base, Large, and
So400m. Burn's current WebGPU backend expands weights to F32 device tensors, before activations,
so many browser adapters can run Base but cannot allocate Large or So400m. The application reports
allocation and network failures instead of silently selecting a smaller model.

The current API exposes final `loadedPartCount` and `loadedBytes` values but no in-flight callback.
Accordingly, the page shows an indeterminate progress bar, elapsed time, and expected transfer
size while the constructor streams and verifies shards. It deliberately does not prefetch all
parts in JavaScript merely to calculate a percentage, because that would defeat bounded sequential
loading and can exhaust browser memory.

## What the example exercises

- `WasmSiglip2.createFromModelSize("base" | "large" | "so400m")`;
- encoded-image decoding and canonical fixed-resolution preprocessing;
- tokenizer-backed text inference;
- combined image/text embeddings, learned-scale logits, and sigmoid probabilities;
- exact loaded-weight, tokenizer, upstream-model, and upstream-revision identity checks;
- explicit cleanup of the previous Wasm model when switching sizes.

The model response contains complete embedding vectors. The page renders only shapes, norms, and a
short vector preview so that the DOM does not retain thousands of formatted numbers. It also shows
the complete canonical weight/tokenizer fingerprints in the embedding diagnostics; use both as
part of any persistent browser embedding-cache key.
