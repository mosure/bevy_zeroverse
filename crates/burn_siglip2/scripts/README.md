# SigLIP2 numerical correctness harness

`siglip2_reference.py` produces an immutable numerical oracle from a local
Hugging Face fixed-resolution SigLIP2 checkpoint. It writes:

- `<output-base>.json`: source/config/preprocessing metadata and tolerances;
- `<output-base>.input.png`: the exact decoded RGB image re-encoded for the
  Rust byte-input pipeline;
- `<output-base>.safetensors`: exact pixel/token inputs, raw and normalized
  image/text embeddings, logits, probabilities, and learned logit scalars.

The generator is local-only. It never downloads a checkpoint, and it has no
default model directory. The three supported variants and example local paths
are:

| Variant | Example `--hf-dir` |
| --- | --- |
| `base-patch16-224` | `/models/siglip2-base-patch16-224` |
| `large-patch16-256` | `/models/siglip2-large-patch16-256` |
| `so400m-patch14-224` | `/models/siglip2-so400m-patch14-224` |

## Generate the F16 CDN oracle

Use a Python environment containing PyTorch, Transformers, Pillow,
Safetensors, and Tokenizers:

```sh
python3 crates/burn_siglip2/scripts/siglip2_reference.py \
  --hf-dir /models/siglip2-base-patch16-224 \
  --variant base-patch16-224 \
  --weight-precision f16 \
  --upstream-revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 \
  --output-base /tmp/siglip2-base-patch16-224-f16-reference
```

`--weight-precision f16` rounds every floating-point checkpoint parameter
through IEEE F16 and then computes in F32. That matches an F16 CDN artifact
which the Burn loader expands into F32 tensors. Use `--weight-precision f32`
to measure the unquantized upstream checkpoint. `--hash-model` additionally
hashes the multi-gigabyte upstream model file; sidecar hashes are always
recorded.

If `--image` is omitted, the script uses a deterministic 83x61 RGB pattern.
Pass `--image` and repeat `--text` to create a semantic end-to-end case.

## Compare an imported Burn artifact

The Rust test accepts either a monolithic `.bpk` or its `.parts.json` manifest:

```sh
BURN_SIGLIP2_NUMERICAL_REFERENCE=/tmp/siglip2-base-patch16-224-f16-reference.json \
BURN_SIGLIP2_NUMERICAL_BUNDLE=/path/to/siglip2-base-patch16-224.bpk.parts.json \
cargo test -p burn_siglip2 --test real_numerical_parity \
  opt_in_real_hf_reference_matches_imported_bundle --release -- --nocapture
```

The real model test is opt-in: with neither environment variable set it
returns immediately. Supplying only one variable is an error. The test feeds
the exact reference pixel tensor and token IDs into Burn, deliberately passes
no text attention mask, and compares all eight public multimodal outputs.

Enable the existing `pipeline` feature to additionally decode the PNG
sidecar, load the tokenizer copied next to the BPK, tokenize the original text
strings, and call `encode_image_bytes_and_text_strings`. The harness first
requires the Rust token IDs to match Hugging Face exactly, then compares all
eight end-to-end outputs:

```sh
BURN_SIGLIP2_NUMERICAL_REFERENCE=/tmp/siglip2-base-patch16-224-f16-reference.json \
BURN_SIGLIP2_NUMERICAL_BUNDLE=/path/to/siglip2-base-patch16-224.bpk.parts.json \
cargo test -p burn_siglip2 --features pipeline --test real_numerical_parity \
  opt_in_real_hf_reference_matches_imported_bundle --release -- --nocapture
```

The importer must retain its default tokenizer-copy behavior so the adjacent
`<variant>.tokenizer.json` and `<variant>.tokenizer_config.json` assets are
available. Older reference manifests without `.input.png` remain usable for
tensor-level parity, but the pipeline portion reports that it was skipped.

Fast smoke tests do not load a model:

```sh
python3 crates/burn_siglip2/scripts/siglip2_reference.py --self-test
cargo test -p burn_siglip2 --test real_numerical_parity \
  comparison_math_and_schema_smoke_are_deterministic
```

## Canonical preprocessing semantics

- Explicit lowercase before tokenization. The fixed-resolution repositories
  advertise `GemmaTokenizer`, which does not itself apply SigLIP2 lowercasing.
- Append EOS (`1`), then right-pad with PAD (`0`) to exactly 64 tokens.
- Do not forward an attention mask. The official fixed-resolution checkpoint
  metadata exposes only `input_ids`, and padding participates in the encoder
  before the final sequence position is pooled.
- Apply encoded EXIF orientation once, convert to RGB, directly resize to the
  configured square with PIL bilinear resampling (`resample = 2`), rescale by
  `1/255`, and normalize each channel with mean/std `0.5`. The slow Hugging
  Face image processor is pinned because fast processor implementations may
  differ slightly.
- L2-normalize both pooled embeddings, then compute
  `exp(logit_scale) * image @ text.T + logit_bias`; probabilities are an
  element-wise sigmoid, never a softmax.

## Tolerance policy

These are same-storage-dtype parity limits embedded in every manifest:

| Output | F32 | F16 storage |
| --- | ---: | ---: |
| embedding max absolute error | `2e-4` | `5e-4` |
| embedding RMSE | `5e-5` | `1e-4` |
| minimum row cosine | `0.999995` | `0.99999` |
| logit max absolute error | `0.02` | `0.05` |
| logit RMSE | `0.01` | `0.03` |
| probability max absolute error | `0.002` | `0.005` |
| probability RMSE | `0.001` | `0.003` |

F16 fidelity relative to upstream F32 must additionally retain embedding
cosine `>= 0.9999`, embedding max error `<= 5e-4`, logit max error `<= 0.05`,
and probability max error `<= 0.005`. On the base checkpoint and the canonical
COCO cats image, measured F16-storage rounding yielded image/text embedding
max errors `1.56e-4`/`1.01e-4`, cosine above `0.999999`, and logit max error
`0.0241`.

Do not loosen these thresholds to accommodate ranking-only success. Incorrect
padding masks, pooling, GELU, or tensor remapping can preserve a top-1 label
while producing numerically incompatible embeddings.
