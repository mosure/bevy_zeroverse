# Generator 10 local evidence

Final consecutive and stratified stress cohorts are complete. Read the
[review](../../procedural_domain_v10.md) for measurements, bounds and failed
intermediate diagnostics. [Provenance](provenance.json) records source/binary
hashes and commands; no selected final captures were skipped.

- [Consecutive distributions](consecutive/capture_distribution.json), [dashboard](consecutive/capture_dashboard.svg), [heatmaps](consecutive/capture_placement.svg).
- [Scene counts](consecutive/captured_scenes.csv), [camera/visibility rows](consecutive/captured_views.csv), [object angles](consecutive/captured_object_rotations.csv).
- [10,000-room CPU metrics](consecutive/metrics.json), [stress distributions](stress/capture_distribution.json), [rotation comparison](rotation_comparison.json).
- [SigLIP2 spacing](embeddings/spacing.svg), [measurements](embeddings/spacing.json), [new closest pairs](embeddings/nearest_v10.jpg), [previous closest pairs](embeddings/nearest_v9.jpg), [encoder controls](embeddings/control_validation.json).
- [Cycles agreement](cycles/comparison.json), [view 0](cycles/view_00_native_cycles.png), [view 1](cycles/view_01_native_cycles.png). Left native, right Cycles; common exposure/display transform, no exposure fit. The 512-sample images retain per-pixel noise. Export material/light mapping limits are recorded in the [reference report](cycles/reference_report.json).

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

[Page 1](consecutive/contact_00.jpg) [Page 2](consecutive/contact_01.jpg) [Page 3](consecutive/contact_02.jpg) [Page 4](consecutive/contact_03.jpg) [Page 5](consecutive/contact_04.jpg) [Page 6](consecutive/contact_05.jpg) [Page 7](consecutive/contact_06.jpg) [Page 8](consecutive/contact_07.jpg) [Page 9](consecutive/contact_08.jpg) [Page 10](consecutive/contact_09.jpg) [Page 11](consecutive/contact_10.jpg)

[Darkest views](consecutive/darkest_views.jpg) · [Image-hash closest pairs](consecutive/closest_pairs.jpg)

### Stress

[Page 1](stress/contact_00.jpg) [Page 2](stress/contact_01.jpg)

[Darkest views](stress/darkest_views.jpg) · [Image-hash closest pairs](stress/closest_pairs.jpg)
