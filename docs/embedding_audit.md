# SigLIP2 capture spacing audit

The optional `embedding_audit` feature adds `indoor_embed`, an offline image
embedding CLI. It uses the first-party Burn 0.21 [burn_siglip2 crate](../crates/burn_siglip2),
maintained and released from this workspace. Historical import provenance is
retained in `ORIGIN.json`; the crate is independently publishable. Normal scene
generation and the viewer do not initialize or download this model. This CLI is native; the shared
scene changes retain the separate WebGPU viewer build. It does not add browser
embedding inference to the viewer.

```sh
python scripts/indoor_embedding_report.py index \
  --cohort before=out/domain9_qualified --cohort after=out/domain10_verified \
  --output out/embedding_audit/index.json
cargo run --features embedding_audit --bin indoor_embed -- \
  --index out/embedding_audit/index.json --output out/embedding_audit/base \
  --variant base --batch-size 16
OPENBLAS_NUM_THREADS=8 python scripts/indoor_embedding_report.py report \
  out/embedding_audit/base/embeddings.json
```

Python reports require NumPy, Pillow and Matplotlib. Indexing requires completed
`indoor_validate` cohorts and verifies their run/selection identities. The encoder
checks image hashes, writes a bounded batch at a time, and produces unit-normalized
float32 embeddings. A completion JSON records tensor/image hashes, actual loaded
weight identity, preprocessing, verified shards and runtime evidence. Existing
outputs are not overwritten. Failed runs have no completion marker.

`--variant base`, `large`, and `so400m` select the fixed-resolution
`base-patch16-224`, `large-patch16-256`, and `so400m-patch14-224` checkpoints.
The model loader downloads shards from
`https://aberration.technology/model/siglip2/`, verifies SHA-256 checksums, and
reuses its cache under `~/.burn_siglip2/models`. `--cache` overrides that root;
`--parts /path/to/model.bpk.parts.json` selects an already downloaded manifest
and still verifies its shards. No model weights are stored in this repository.
Only the image tower executes, although the current loader reads the complete
dual-tower bundle. Batches use one embedding readback each; intermediate
activations stay on the GPU. Model load/verification and inference/IO times are
reported separately. Download, decoding and readback can still leave the GPU idle.

The image processor converts RGB, resizes directly to the model's square size
using Pillow-compatible bilinear sampling, and scales channels with mean/std 0.5.
This follows the fixed-resolution [Google SigLIP2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224).
Compare runs using the same checkpoint and preprocessing.

The report exports:

- Exact nearest cosine distances to **other scenes**, excluding all cameras and
  timesteps of the same scene; contact sheets show the closest pairs.
- Separate trajectory and within-room camera distances, plus exact-file duplicate
  checks when duplicate inputs are present.
- Nearest scene-centroid distances and the effective rank of the centered scene
  centroid covariance. Each scene contributes once to these statistics.
- Exploratory small-distance fractions and quantile plots; no universal spacing
  threshold is treated as an acceptance criterion.

Exact reporting is bounded to 4,096 views per cohort and 50,000 indexed images
overall. For a 10M-image dataset, audit declared, equal-size samples of scenes.
This bounded implementation deliberately does not claim scalable all-pairs search.

Spacing is sensitive to cohort size, camera policy and subject composition.
Increasing spacing can also mean generating irrelevant content. A global semantic
encoder can miss local geometry defects, bad materials or incorrect annotations.
Use the nearest-pair images alongside the geometry/visibility metrics and matched
lighting references. Embedding spread alone establishes neither photographic
realism nor downstream pretraining utility.

The generator-10 evidence was recorded before the crate moved from its original
import directory into `crates/burn_siglip2`. Historical provenance paths are kept
unchanged; the inference source and model artifacts were unchanged by that move.
