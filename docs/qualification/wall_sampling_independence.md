# Wall placement and content sampling qualification

This audit addresses accidental cross-category predictors in masked-image and
cross-view reconstruction data. Seeded replay remains deterministic: all views
of a scene must share the same geometry and content. Replacing deterministic
seeds with wall-clock randomness would damage replay without demonstrating
better training data.

## Findings and sampling contract

The supplied seed, **1000000010**, resolves to a Library under Mixed at density
0.65. In the recorded released source, two small pictures and a fire alarm share
a chamfered wall. Their instance seeds are different; there is no explicit rule
placing a fire alarm next to a picture or selecting picture content from an
alarm. A single room cannot establish a population-level association.

The source review nevertheless found removable coupling:

| Decision | Current contract |
| --- | --- |
| Ordinary surface dressing | Separate from the existing architecture/furniture cursor; dressing is keyed by support seed rather than a shared cursor across all tables. The main furniture sampler is retained. |
| Public-program supported contents | Support-local streams; retries on one counter do not advance the next counter's proposals. |
| Standalone art, displays, boards and clocks | Independent category counts, preserving each former marginal distribution. They no longer consume one shared category budget. |
| Wall-equipment proposals, pendant proposals and wall groups | Separate count/proposal streams. Rejection draws for one proposal do not shift the next proposal. |
| Alarms, speakers, exit signs, detectors, string lights and electrical fittings | Separate domains. A difficult exit-sign placement cannot shift the alarm/speaker random choices. Speaker products can still form a matched installation. |
| Wall admission order | Twelve installation passes have a seedable permutation. Artwork is neither always first nor always after safety/electrical fittings. Functional furniture, projector/screen pairs and ceiling reservations precede these passes. |
| Electrical/service/person clearance | Both admission directions reserve 25 mm. Late sockets and human poses cannot invalidate an earlier fixture by using a smaller broad-phase margin. The lecture-hall seed 62 regression includes the previously missed 22 mm human/speaker gap. |
| Low-ceiling alarm/speaker height | Sample the feasible interval, accounting for the actual floor/ceiling and product height. Do not clamp many samples onto an identical roof-relative height. |
| Power-strip standard | Sample once per room, independent of consecutive-seed modulo. Realistic within-room coordination is retained. |
| Displayed documents | Object identity selects the content slot; furniture-finish identity does not. Desktop, phone, television, whiteboard, artwork and printed paper use separate content namespaces. |
| Rendering and content metadata | Baking, construction labels, metrics, capacity descriptors and the furniture review example use the same surface-aware content key. |

`layout/sampling.rs` owns explicitly numbered random domains. Counts and
per-proposal child keys use distinct streams. Child keys are addressed directly
by ordinal, so retry consumption, early exits and instance numbering cannot
shift later proposal keys. All retry budgets remain finite. This is
counterfactual isolation, not a claim of mathematically independent unlimited
entropy: scene generation is still a finite seeded program.

Admission still depends on actual geometry. A window takes wall area away from
pictures; a gallery can have coordinated frames; worktop outlets follow
worktops; exit signs and pull stations follow doors. Collision exclusion can
produce accepted-placement correlations even with independent proposals. These
physical relationships are retained. In archived rectangular manifests, the
three exterior attachment walls remain distinct from the internal +Z partition;
current polygonal envelopes use their actual wall set.

## Evidence and reproduction

The local matched audit uses the frozen source at
`2bd5c20bf190350194aa4f97f9e2c212a0431d4e` and the revised working tree. It samples
seeds 0–511 separately for Mixed, Gallery, Conference and Cafe: **2,048 rooms per
sampler**, cycling densities 0.35, 0.65 and 0.90. The population audit has zero
cameras and zero people to isolate placement cost. Every room runs layout and
public-program validation; the first eight per profile also run full mesh
validation. No failed seeds are discarded.

The raw records include requested/resolved profiles, room dimensions, density,
object counts, accepted positions/yaws/sizes/variants, artwork content keys and
content parameters. The sample unit for correlation tests is a **room**, not an
object pair. Six exploratory probes per profile cover art/display/board counts
and mean art brightness/density/horizon versus alarm position/height. Permutations
preserve resolved room profile and density; 999 shuffles use a fixed analysis
seed. The 24 probes are also reported with Benjamini–Hochberg correction. These
small probes are diagnostics, not an exhaustive nonlinear shortcut detector.

```sh
cargo run --locked --example audit_placement_sampling -- \
  --rooms 512 --mesh-rooms 8 --output out/placement_sampling/after
cargo run --locked --example audit_placement_sampling -- \
  --seed 1000000010 --rooms 1 --density 0.65 --human-density 0.25 --cameras 3 \
  --mesh-rooms 1 --save-scenes --output out/placement_sampling/after-seed
cargo test --locked --lib scene::procedural_indoor::layout:: -- --test-threads=2
cargo test --locked --lib materials::content_diversity_tests
cargo test --locked --lib materials::variants
cargo test --locked --lib metrics::content
cargo test --locked --lib capacity::
```

`--save-scenes` exports full manifests, actual API schematics and artwork
albedo maps. Schematics are diagrams, not photographic captures. Local raw
records, input hashes, analysis scripts, qualification logs and seed diagrams
are retained under `out/placement_sampling/`; this study is not injected into
the project page or paper.

## Observed results

Both 2,048-room populations passed layout/public-program validation with no
rejected seeds; both 32-room mesh subsets passed. These are consecutive samples,
not a gallery selected for favorable outcomes.

| Requested profile | Rooms | Mean total objects | Mean wall art | Alarm present | Art and alarm on the same wall, conditional on both being present |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 512 | 76.62 | 2.95 | 98.2% | 23.9% (350 rooms) |
| Gallery | 512 | 38.78 | 4.68 | 99.0% | 34.3% (431 rooms) |
| Conference | 512 | 97.73 | 1.97 | 98.0% | 23.0% (291 rooms) |
| Cafe | 512 | 57.71 | 2.86 | 98.8% | 23.4% (337 rooms) |

The released population also showed weak associations on the selected probes:
maximum absolute within-layout/density correlation 0.074, minimum adjusted
q-value 0.840. Revised values are 0.095 and 0.528. **The measured picture/alarm
association was not established in either population.** These results support
removing the concrete coupling found in source; they do not demonstrate a
reduction in learned shortcuts or stronger real-world transfer.

The four independent standalone decoration proposal counts were checked on
4,096 seeds. Means for art/display/board/clock are 1.368/1.372/2.784/1.403,
against intended values 1.4/1.4/2.8/1.4. Maximum absolute pair correlation is
0.0505. Across 2,048 seeds, art is ahead of alarms in 1,003 cases (49.0%),
electrical fittings in 983 (48.0%), exit signs in 1,036 (50.6%) and wall groups
in 1,053 (51.4%). The counterfactual test exercises picture/group generation,
removes its geometry, then verifies unchanged alarm and outlet proposals in 64
rooms. Actual collisions are intentionally not removed from production.

There are 6,381 accepted artwork instances in the revised population. Of these,
184 reuse another artwork's content key within their room (2.88%). The bounded
cache therefore still permits repetition; the audit does not mistake seed/key
uniqueness for perceptual diversity.

Seed **1000000010** passes full layout, public-program, mesh and camera checks
with three cameras and human density 0.25, for Mixed, Gallery, Conference and
Cafe. The Mixed room has four admitted people. Its local diagnostic compares
wall placement and includes a separate full API schematic and actual artwork
albedo exports. No inference is made from one aesthetically unusual placement.

The native checks pass **127 distinct tests**, including 100 layout tests,
1,024 consecutive Mixed rooms with three cameras and human density 0.7,
material/content selection and PBR-map byte parity, construction-to-metadata
bindings, support/contact/cable/clearance checks, camera contracts, capacity
metadata, and seeded replay. Three optional manual category-gallery diagnostics
remain ignored; they are not counted as passes. Strict native all-target Clippy,
formatting and diff checks also pass.

Wall decoration, electrical and group passes cache unchanged collision bounds
locally, extending them only for accepted objects. Cache lifetime is one pass;
there is no persistent scene cache or additional texture budget. The optimization
preserved **32/32 byte-identical complete manifests** (including people and three
cameras) and **2,048/2,048 identical population records** excluding the timing
field. The changed admission passes were retested after the optimization.

Three interleaved processes per sampler each generated 512 rooms (128 seeds ×
four profiles), following short warmups. Both builds used the same pinned
nightly, optimized dev profile (`opt-level=1`, dependencies also 1, debug off).
Training remained active on the shared host.

| CPU-only measure | Released sampler | Revised sampler |
| --- | ---: | ---: |
| Median of three run-mean manifest latencies | 4.380 ms/room | 3.921 ms/room |
| Range of run-mean manifest latencies | 3.923–4.778 ms/room | 3.866–4.786 ms/room |
| Median whole-audit process CPU time, including validation/JSON export | 5.320 ms/room | 4.730 ms/room |

Run ranges overlap and adjacent paired results vary. This short shared-host
measurement found no consistent preparation slowdown; it is not evidence of a
10% production capture speedup. Full runs and process CPU measurements are in
`out/placement_sampling/timing.json`.

## Interpretation and limits

Deterministic content cannot be made impossible to memorize by adding random
jitter. Factor holdouts and downstream comparisons with correct, repeated and
mismatched reference views are needed to evaluate whether the model uses
cross-view geometry. The audit does not change the training process or assert
an improvement in MAE/CroCo transfer.

The content cache still has 120 slots per screen/art/board surface and 12 for
printed paper. Only consumed slots are baked. Same-room slot reuse remains
possible; a unique key is not proof that two images are perceptually distinct.
PBR shaders, image sizes, material-map channel conventions, semantic categories,
camera calibration and rendering algorithms are unchanged. Geometry and content
sampling change. The matched audit used generator **113**, **capture-v138**
(`indoor=56`, `finishes=23`). The combined 0.62.3 release includes the subsequent
[seed recovery qualification](rejected_seeds_20261106.md) and uses generator
**115**, **capture-v140**, including the subsequent
[furnishing expansion](furnishing_diversity_20261106.md); scene caches must use
that combined identity.

CPU timings measure manifest construction only. They do not qualify completed
capture throughput, GPU utilization, photographic realism, browser rendering or
concurrent-training performance. No local GPU renders or training changes are
part of this audit.
