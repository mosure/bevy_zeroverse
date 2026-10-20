# Seating construction space

Seating uses seeded, conditional construction programs rather than selecting a
finished mesh and changing its scale. The long-standing chair family IDs still
select appropriate task, conference, visitor, dining, stool, lounge and executive
contexts. Independent streams sample the mechanisms and continuously vary their
dimensions. No assets, models, texture atlases or new per-frame systems are loaded.

## Realized geometry

Task and conference chairs combine straight, arched or cast-blade radial bases
and disc pedestals with caster or glide feet. Visitor, shell and dining chairs
use tapered legs, sleds, cantilevers or trestles. Casters require a radial base.
Eligible unoccupied chairs can slide or roll as a complete body; swivel chairs
retain a separate upper-seat mechanism. The motion planner checks the actual
swept geometry and support clearance before admitting either movement.

Back structures include tension mesh, spindles, shaped shells, segmented cushions,
slats, split ergonomic lobes, lumbar openings, upholstered ribs and vertical
formed slats. Arms have post, loop, fork, branching, arch and covered-panel
supports. Seats include crowned cushions, smoothly deformed waterfall fronts
with transformed normals, and separate replaceable front bolsters. Headrests
include formed shells, oval pads and split pads. Continuous parameters control
curvature, width, taper, crown, lumbar shaping, recline, rib pitch, front-bolster
depth and spoke arch height.

Armchairs, loveseats, sofas and sectionals combine track, rolled, swept,
open-bentwood, flared or absent arms; tapered feet, hairpins, sleds, plinths,
pedestals/bridge frames or block feet; loose, lumbar-split, channel, bolster-crest
or exposed-frame backs; and individual, bench or split-front seat cushions.
Bench cushions are restricted to straight seating. Cushion widths, asymmetry,
thickness, crown, gaps, arm rise and sweep remain continuous. Loose pillows use
the actual seat construction for support and avoid the front-bolster seam.

Some lounge chairs are recliners, with a reserved larger footprint, a supported
reclining back, padded leg rest, paired linkage bars and a recessed release
lever. Extension is sampled as a static continuous mechanism position. These
chairs do not receive the existing upright human pose: that pose does not supply
a reclined leg/torso support fit. This feature does not add animated reclining.

## Contracts and qualification

The shared 0.47 m chair seating datum, semantic classes, object IDs and annotation
ownership are retained. A recliner's complete posed envelope is reserved before
placement, human population and camera selection. New structural controls are
available through `objects::audit::program`, `chairs::parameters`,
`seating::parameters` and the dataset distribution metrics. Generator revision
56 distinguishes these samples for deterministic replay.

The ignored Rust tests `seating_space_qualification` and
`seating_population_qualification` write local artifacts under
`out/seating_space/current`. The former checks 96 fixed-envelope samples in each
of eleven categories, records construction time and triangle counts, validates
mesh attributes, semantics and bounds, and checks exact replay. It also samples
4,096 parameter programs per category. The recliner category conditions on
recliner selection; the other categories use consecutive seeds. The latter
validates 64 complete rooms, with cameras and people, across mixed, lounge,
conference and coworking layouts.

Geometric distances use six 24-by-24 first-surface depth maps, with color,
materials, scale and world pose excluded. Coverage is reported at several
thresholds rather than selecting one favorable threshold. Active discrete
construction signatures are a separate diagnostic: they do not measure
independent training examples, SigLIP volume or photographic realism.
Continuous parameter ranges must not be multiplied into a claimed dataset size.

The contact sheets are flat-shaded CPU geometry previews. CPU construction
timings exclude uploads, rendering, readback and concurrent training. Native
and browser GPU appearance and end-to-end throughput require separate runtime
qualification.

## Local sample results

The 1,056-mesh sweep passed attribute, bounds, winding, semantic and replay
checks. The 64-room sweep passed with 560 seating objects and 280 people; two
accepted chairs were recliners. The dedicated recliner fixture is conditional
coverage, not a claim that recliners occur frequently in mixed rooms.

| Fixed-envelope category | Observed construction signatures / 4,096 programs | Shape representatives / 96 at distance 0.15 | Median build, ms | Largest mesh, triangles |
| --- | ---: | ---: | ---: | ---: |
| Task chair | 2,214 | 96 | 0.172 | 9,764 |
| Conference chair | 1,151 | 90 | 0.151 | 8,580 |
| Visitor chair | 502 | 84 | 0.123 | 6,758 |
| Dining chair | 372 | 90 | 0.178 | 8,580 |
| Shell chair | 251 | 84 | 0.168 | 8,628 |
| Executive chair | 2,099 | 96 | 0.174 | 10,388 |
| Lounge chair | 360 | 17 | 0.173 | 7,216 |
| Loveseat | 539 | 24 | 0.311 | 13,192 |
| Sofa | 1,011 | 16 | 0.454 | 21,432 |
| Sectional | 1,455 | 34 | 0.502 | 26,360 |
| Recliner | 18 | 11 | 0.201 | 7,852 |

These counts exclude material choices, continuous numerical fields and loose
pillows; inactive decisions are omitted. The one-module bench/individual-seat
equivalence is collapsed. Distances 0.015, 0.03, 0.08 and 0.25 and nearest-neighbor
shape distances are also recorded in the JSON report. The per-mesh budgets are
24,000 triangles for chairs and 32,000 for multi-seat sofas/sectionals.

The local matched study exceeds the 10× construction-program target in each
existing category, using conservative bounds on the preceding construction
choices. **It does not establish a 10× increase in perceptual volume or useful
training samples.** Shape-distance gains are smaller and depend on category and
resolution. Build timings were collected under concurrent training load; they
are indicative object costs, not a controlled end-to-end throughput comparison.
The matched comparison and previews remain in ignored `out/seating_space/`;
they are not project-page or paper assets.
