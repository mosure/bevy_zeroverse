# SigLIP2 capture spacing audit

The optional `embedding_audit` feature adds `indoor_embed`, an offline image
embedding CLI. It uses the first-party Burn 0.22 [burn_siglip2 crate](../crates/burn_siglip2),
maintained and released from this workspace. Historical import provenance is
retained in `ORIGIN.json`; the crate is independently publishable. Normal scene
generation and the viewer do not initialize or download this model. This CLI is native; the shared
scene changes retain the separate WebGPU viewer build. It does not add browser
embedding inference to the viewer.

```sh
cargo run --locked -p bevy_zeroverse_publication -- compare-indoor \
  --before out/study/before --after out/study/after --output out/study/report
cargo run --features embedding_audit --bin indoor_embed -- \
  --index out/study/report/embedding-index.json --output out/study/embeddings \
  --variant base --batch-size 16
cargo run --locked -p bevy_zeroverse_publication -- compare-indoor \
  --before out/study/before --after out/study/after --output out/study/report \
  --embeddings out/study/embeddings
cargo run --locked -p bevy_zeroverse_publication -- verify-study \
  --output out/study/report
```

The Rust report requires completed, matched `indoor_validate` cohorts and verifies
their run/selection identities, all images and actual-mesh qualification records.
Matched studies remain in ignored `out/` directories, outside `out/publication`.
They carry `local_only` receipts and cannot become published reference galleries.
It defaults to 64 consecutive rooms, three 512×512 views and one timestep;
`--recipe` selects a different single-timestep study protocol. The encoder
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
- Exact cross-room RGB duplicate checks, without treating cameras from the same
  room as independent room samples.
- Nearest scene-centroid distances and the effective rank of the centered scene
  centroid covariance. Each scene contributes once to these statistics.
- Exploratory small-distance fractions and distributions; paired-room bootstrap
  confidence intervals. No universal spacing threshold is an acceptance criterion.
- Conditional covariance-ellipsoid volume ratios in a common pooled PCA basis,
  at supported ranks 4, 8 and 16. Per-axis spread and paired bootstrap intervals
  accompany the ratios; degenerate cohorts are reported explicitly. This is a
  fixed-sample diagnostic, not an estimate of global perceptual manifold volume.

The paired gallery is bounded to 256 rooms and 4,096 views per cohort. The encoder
accepts at most 50,000 indexed images. For a 10M-image dataset, audit declared,
equal-size samples of scenes.
This bounded implementation deliberately does not claim scalable all-pairs search.

Spacing is sensitive to cohort size, camera policy and subject composition.
Increasing spacing can also mean generating irrelevant content. A global semantic
encoder can miss local geometry defects, bad materials or incorrect annotations.
Use the nearest-pair images alongside the geometry/visibility metrics and matched
lighting references. Embedding spread alone establishes neither photographic
realism nor downstream pretraining utility.

The standalone Python utilities remain available for exploratory trajectory
analyses, but the paired gallery, tensor binding and qualification receipt use
the first-party Rust publication crate.
