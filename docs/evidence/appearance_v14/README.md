# Appearance v14 evidence

See [the qualification report](../../appearance_v14.md) for methods, exact scope and limitations. `summary.json` explicitly records `photographic_realism_established: false`.

- `hair_before_after.png`: eight front portrait controls and two rear tied-hair controls; matching subjects, lighting and cameras.
- `glass_sampling.json`: all 32 views of consecutive seeds 44–59, including insufficient-coverage views and per-view filter bias.
- `glass.png`: fixed-display comparisons for two visible frosted-glass cases.
- `cycles/*.json` and `native_cycles.png`: all eight physical-reference comparisons; difficult/low-light views retained.
- `radiometry.json`: six independent analytic bridge controls.
- `browser.json`, `browser_auto.png`, `browser_portable.png`: actual occupied WebGPU scenes, readiness and validation checks.
- `logs/`: successful final checks. The Rust future-incompatibility message belongs to the published Burn CubeCL dependency.
- `provenance.json`: exact source hashes, original local raw-data root, and verified optical-reference reuse.

Large EXR, RGBA32F and geometry/map exports remain under the recorded ignored `out/` directory. These reports are local work; no CI, commits, push or publication was performed.
