# Generator 10 local evidence

Final consecutive and stratified stress cohorts are complete. Read the
[review](../../procedural_domain_v10.md) for measurements, bounds and failed
intermediate diagnostics. [Provenance](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/provenance.json) records source/binary
hashes and commands; no selected final captures were skipped.

- [Consecutive distributions](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/capture_distribution.json), [dashboard](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/capture_dashboard.svg), [heatmaps](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/capture_placement.svg).
- [Scene counts](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/captured_scenes.csv), [camera/visibility rows](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/captured_views.csv), [object angles](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/captured_object_rotations.csv).
- [10,000-room CPU metrics](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/metrics.json), [stress distributions](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/stress/capture_distribution.json), [rotation comparison](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/rotation_comparison.json).
- [SigLIP2 spacing](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/embeddings/spacing.svg), [measurements](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/embeddings/spacing.json), [new closest pairs](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/embeddings/nearest_v10.jpg), [previous closest pairs](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/embeddings/nearest_v9.jpg), [encoder controls](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/embeddings/control_validation.json).
- [Cycles agreement](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/cycles/comparison.json), [view 0](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/cycles/view_00_native_cycles.png), [view 1](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/cycles/view_01_native_cycles.png). Left native, right Cycles; common exposure/display transform, no exposure fit. The 512-sample images retain per-pixel noise. Export material/light mapping limits are recorded in the [reference report](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/cycles/reference_report.json).

The 3 MiB `embeddings/embeddings.f32` tensor and its completion metadata allow
numerical spacing recomputation without the model or original image files:

```sh
python scripts/indoor_embedding_report.py report \
  docs/evidence/domain10/embeddings/embeddings.json --no-figures
```

Image contact sheets are self-contained. Metadata retains original local capture
paths for regenerating sheets. Full RGB/semantic PNGs and source meshes remain in
`out/domain10_verified`, `out/domain10_verified_stress` and `out/domain10_linear`.
No model weights are included.

## Every final captured view

### Consecutive

[Page 1](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_00.jpg) [Page 2](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_01.jpg) [Page 3](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_02.jpg) [Page 4](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_03.jpg) [Page 5](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_04.jpg) [Page 6](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_05.jpg) [Page 7](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_06.jpg) [Page 8](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_07.jpg) [Page 9](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_08.jpg) [Page 10](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_09.jpg) [Page 11](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/contact_10.jpg)

[Darkest views](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/darkest_views.jpg) · [Image-hash closest pairs](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/consecutive/closest_pairs.jpg)

### Stress

[Page 1](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/stress/contact_00.jpg) [Page 2](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/stress/contact_01.jpg)

[Darkest views](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/stress/darkest_views.jpg) · [Image-hash closest pairs](https://github.com/mosure/bevy_zeroverse/blob/6cc9ee235004b0e71b7eeb69bc5990375f77d48c/docs/evidence/domain10/stress/closest_pairs.jpg)
