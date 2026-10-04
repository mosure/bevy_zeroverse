# bevy_zeroverse 0.30

Core and FFI **0.30.0**, Burn wrapper **0.13.0**, capture contract **0.1.1** and
publication tool **0.1.2** release the accumulated indoor geometry, appearance,
placement and motion improvements. The renderer contract is capture-v47,
generator 29, wardrobe 4 and morphology 2.

Continuous ceramic, paint, concrete and textile programs extend correlated
color, roughness and relief variation. Furnishings, plants, supported props,
architecture, garments, footwear and long-hair grooms gain geometric variation
and tighter placement/mesh checks. Motion navigation and admission include
kinematic diagnostics; static people remain supported without loading motion
models.

Adult Anny shape programs diversify height, build, age, muscle and proportions.
Height anchors stay within the calibrated adult range; shoulder and hip spans
scale with stature. Torso retargeting preserves shoulder yaw, and shared-vertex
skin/cloth fields prevent inconsistent sleeve boundaries. Collision reservations
remain conservative independently of rendered anatomy. See the
[body-fit qualification](procedural_indoor_humans.md),
[hair review](hair_quality.md), [geometry review](geometry_quality.md),
[material review](material_quality.md) and
[placement qualification](procedural_space_robustness.md) for bounded evidence
and separate source identities.

The formal Rust publisher refreshes the project page and paper from 512 current
room programs, 32 rendered rooms, four views and two trajectory samples, retaining
all six aligned annotation modes and exact co-visibility membership masks.
These instantaneous measurements do not establish photographic realism,
representative demographics or ten-million-sample downstream training utility.

## Rust compatibility

New optional body, hair and garment programs and material controls add public
struct fields. Exhaustive Rust literals must initialize them; use defaults or
sampling constructors where available. Missing optional serialized fields retain
legacy replay behavior. The minor release prevents these source-level changes
from arriving through a patch update. Both wrappers require core 0.30.0.

Numeric annotation layouts are unchanged. Generator and compiled-source identity
checks distinguish new captures from older runs; captures are not silently mixed.

Renderer and publisher now share the population-metrics schema constant. The
release gate rejects stale or future schemas; a regression test covers this
contract alongside missing annotations and mixed-source captures.
