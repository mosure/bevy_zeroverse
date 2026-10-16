# Configuration-conditioned capacity evidence

Capacity is an explicit, optional audit of the **accepted, captured population**.
It does not run during normal generation, viewer edits or tensor capture. The
default observation is a complete multi-view room, not each view independently.
Mixed/All samples all twenty concrete room programs. Geometry, accepted objects,
people, assigned materials, illumination and scheduled cameras enter one joint
descriptor; their marginal counts are never multiplied.

## Rust API

```rust,no_run
use bevy_zeroverse::scene::procedural_indoor::capacity::{
    CapacityRequest, CapturePlan, CapacityReceipt,
};
# fn audit(config: &bevy_zeroverse::app::BevyZeroverseConfig,
#          capture: CapturePlan, receipt: &CapacityReceipt) -> Result<(), String> {
let request = CapacityRequest::default();
let plan = config.capacity_plan(&request, capture.clone())?;
// A pure query: no downloads, GPU initialization or implicit audit.
let estimate = config.estimate_capacity(&request, capture, receipt)?;
# Ok(()) }
```

The capture contract explicitly specifies image dimensions, cameras, delivered
progress values, optional physical seconds, annotation modes and glass policy.
`capacity::generate` and `capacity::observe` are the opt-in CPU adapter and
accepted-program observer. `capacity::observe_rendered` consumes a completed
`Sample`, checking source identity, full K, poses, delivered times and finite
RGB/position/normal/semantic buffers. It uses the same primary-region membership
as reconstruction diagnostics. Caller-declared mesh checks must have actually
completed before setting `mesh_checked`.

The schema is in `bevy_zeroverse_capture::capacity`; offline estimators and
reporting are in `bevy_zeroverse_publication::capacity`. Portable consumers do
not need Bevy or a GPU. Generator queries reject mismatched source, configuration,
seed law, definition or capture schedule. Exact root seed/source ancestry groups
interventions and frames; this identity is never counted as an appearance class.

## Explicit local workflow

```sh
cargo run --bin indoor_capacity -- plan --output out/capacity-plan.json
cargo run --bin indoor_capacity -- audit \
  --plan out/capacity-plan.json --output out/capacity-audit --workers 2
cargo run -p bevy_zeroverse_publication -- capacity-qualify \
  --plan out/capacity-plan.json --records out/capacity-audit/records.jsonl \
  --output out/capacity-receipt.json
cargo run -p bevy_zeroverse_publication -- capacity-report \
  --receipt out/capacity-receipt.json --output out/capacity-report
cargo run --bin indoor_capacity -- estimate \
  --plan out/capacity-plan.json --receipt out/capacity-receipt.json
```

These commands neither publish nor alter training. Outputs are immutable; a
failed/interrupted run cannot masquerade as a completed receipt. Every requested
draw, including failures, remains in the ordered JSONL. The worker pool has at
most twice the worker budget in flight and deterministic ordering. No retry
resamples a failed seed. The initial exact in-memory qualifier is capped at
65,536 rooms; a 10M draw request is an estimand, not a request to allocate or
render 10M rooms. External sorting remains unimplemented. The optional exact
image-space kernel below has its own smaller, explicit memory budget.

For native rendered qualification, resolve a plan with a `rendered_grid`
definition, equal manifest/render budgets, and `color`, `normal`, `position`,
`semantic` modes. Then use `indoor_capacity capture --plan … --output …` instead
of `audit`. One production headless app/device survives across the cohort. It
waits for construction, material baking and sampler readiness, preserves
production lighting/shadows, and records completed-capture and audit time
separately. The current runner captures at most 2,048 rooms and saves manifests,
provenance and RGB previews. Raw annotation planes are hashed at observation but
not persisted by this runner; use the normal dataset export when retaining every
plane is required. The observer API can also consume those completed samples.
Each PNG preview has a checksum bound to the original JSONL record bytes.
Indexed captures reset the scene sequence even for repeated seeds. Explicit
clips use `SamplerState::start_progress`; its default remains zero, and invalid
values fail before changing playback or posing actors. Later times remain in
`SamplerState::timesteps`. Physical seconds are checked when requested; they are
not invented from the renderer's frame rate.

## Optional SigLIP2 sample audit

Freeze the image index, loaded-weight SHA256 and kernel bandwidth **before**
running inference. The index builder validates the capacity receipt, original
records, preview checksums and complete view/time coverage. All successful rooms
enter; failed draws stay in the binding. The existing optional `indoor_embed`
tool uses checksum-verified cached shards and bounded batches.

```sh
cargo run -p bevy_zeroverse_publication -- capacity-embedding-index \
  --receipt out/render-receipt.json --records out/render-audit/records.jsonl \
  --captures out/render-audit --weights-sha256 LOADED_WEIGHT_SHA256 \
  --bandwidth 0.35 --max-rooms 512 --output out/embedding-index.json
cargo run --features embedding_audit --bin indoor_embed -- \
  --index out/embedding-index.json --output out/embeddings --batch-size 16
cargo run -p bevy_zeroverse_publication -- capacity-embedding-report \
  --index out/embedding-index.json --embeddings out/embeddings \
  --output out/kernel-report
```

This is a whole-image diagnostic, including exterior context. Normalized view
embeddings are averaged and normalized within each room; every room gets equal
weight. The uncentered RBF kernel reports Vendi, participation, the complete
spectrum, numerical residual and exact nearest rooms. Its 512-room ceiling is a
hard limit, not silent subsampling. A frozen bandwidth defines a sample statistic;
it is not automatically a calibrated perceptual threshold. Repeated modes and
analytically known spectra test the implementation. No image-space score is
converted to a global population size or an expected 10M-draw distinct count.

## Paper and page integration

`capacity-paper --receipt RECEIPT... --output DIRECTORY` generates a local
source-checked paper section, plots and report pages. It does not publish.
The canonical publication recipe optionally accepts `capacity_receipts`, a list
of up to four receipt paths under `docs/evidence`. Refresh checks their exact
generator identity before starting captures and includes their hashes in the
publication attestation. Each configuration retains a separate table and plot;
removing the configured evidence removes the staged section and figures. No
unqualified capacity count appears on the default page.

## What the numbers mean

- **Observed classes**: exact distinct fixed-bin joint descriptors in this cohort.
  Accepted-program bins are construction evidence, not perceptual equivalence.
  Rendered grids add view-dependent evidence at their stated grid/bin resolution;
  neither is a calibrated human-judgment metric.
- **Collision diversity**: reciprocal collision probability. The all-pairs point
  is separate from exact confidence bounds using prespecified disjoint room pairs.
  Bounds are adjusted across all declared definitions. Zero collisions yields a
  lower bound and an unknown upper bound, never infinity.
- **Requested distinct bound**: a conservative population bound conditional on
  successful independent draws. No extrapolated point estimate is provided.
  Fixed/consecutive sweeps have observed statistics only, without IID confidence.
- **Rarefaction**: exact without-replacement occupancy within the observed cohort.
  Prefix/holdout novelty diagnostics do not automatically qualify an unseen-tail
  forecast. The current qualified extrapolation horizon is zero.
- **Factor diagnostics**: counts, relationships, material construction area and
  pose/lighting coverage include absent categories as zero. Marginal block counts
  identify collapse but are not independent multiplicative capacity factors.

The default single-timestep capture observes initial camera transforms only.
Paths affect admission but cannot multiply temporal capacity. Disabled human or
object motion is inactive. Repeated identical complete frames canonicalize to one
state. Multiple modalities annotate the same sample. Explicit motion capture must
deliver completed trajectories; a CPU audit cannot substitute static humans.

The viewer's Advanced panel accepts a qualification receipt for read-only review.
It shows sample unit, resolution, delivered counts and forecast horizon, marks
configuration/source mismatches, and keeps the receipt out of scene URLs. Pasting
or changing evidence never regenerates the scene or starts a workload.

## Scientific promotion gates

Current reports make no global perceptual-capacity, photographic-accuracy or
learning-utility claim. Calibrated partitions need independent intervention
cohorts. Full tinted/frosted-light transport, a sustained native memory plateau,
Metal/Chrome image qualification, uncontended completed-capture benchmarks and a
fixed learner/real evaluation dataset remain separate experiments. A large
construction-class count cannot pass those gates. Source-bound results and the
execution status of the broader plan live in `procedural_capacity_plan.md`.
